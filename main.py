import os, pathlib, tempfile, shutil, subprocess
import json, base64
from typing import cast, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langsmith import traceable
from vosk import KaldiRecognizer

from utils.nlu import classify
from utils.config import SYSTEM_PROMPT
from graph import build_graph
from state import ChatStateModel, MessageModel, ChatStateTD, to_graph_state, from_graph_state
from utils import db
from nodes.voice_hooks import stt_listen, tts_speak, _get_vosk_model

# ---- startup / env -----------------------------------------
load_dotenv()
print("[CFG] USE_WAITER_LLM =", os.getenv("USE_WAITER_LLM"))
print("[CFG] WAITER_LLM_BACKEND =", os.getenv("WAITER_LLM_BACKEND"))
print("[CFG] VOSK_MODEL_PATH =", os.path.abspath(os.path.expanduser(os.getenv("VOSK_MODEL_PATH", "./models/vosk"))))

DATA_DIR = pathlib.Path("./data_audio")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def _save_upload_to_tmp(upload: UploadFile) -> str:
    suffix = ""
    if upload.filename and "." in upload.filename:
        suffix = "." + upload.filename.split(".")[-1].lower()
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return tmp_path

def _maybe_to_wav_16k_mono(input_path: str) -> str:
    """Ensure audio is 16kHz mono WAV for Vosk."""
    norm_path = input_path + ".norm.wav"
    if _has_ffmpeg():
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ac", "1", "-ar", "16000",
            "-f", "wav", norm_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return norm_path
    return input_path

def _pack_state(session_id: str):
    model = db.load_state(session_id)
    if not model:
        return {"cart": [], "total": 0, "stage": None, "last_order": {}}
    md = model.metadata or {}
    cart = md.get("cart", [])
    total = sum(int(it.get("price", 0)) * int(it.get("qty", 1)) for it in cart)
    return {"cart": cart, "total": total, "stage": md.get("stage"), "last_order": md.get("last_order") or {}}

def ensure_system_prompt(model: ChatStateModel) -> None:
    if not any(m.role == "system" for m in model.messages):
        model.messages.insert(0, MessageModel(role="system", content=SYSTEM_PROMPT))

# ---- app / graph -------------------------------------------
app = FastAPI()
graph_app = build_graph()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    user_message: str

@traceable(name="POST /chat", tags=["api", "phase1"])
def _handle_turn(model: ChatStateModel, user_text: str) -> ChatStateModel:
    user_text = (user_text or "").strip()
    if not user_text:
        return model
    ensure_system_prompt(model)
    stage = model.metadata.get("stage")
    _ = classify(user_text, stage=stage, history=[m.model_dump() for m in model.messages][-4:])
    model.messages.append(MessageModel(role="user", content=user_text))
    model.metadata["_awaiting_worker"] = True
    out_dict = graph_app.invoke(to_graph_state(model))
    return from_graph_state(cast(ChatStateTD, out_dict))

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    model = db.load_state(req.session_id) or ChatStateModel(session_id=req.session_id)
    ensure_system_prompt(model)
    model = _handle_turn(model, req.user_message)
    db.save_state(req.session_id, model.model_dump())
    return {"response": model.messages[-1].content if model.messages else ""}

# ---- Voice HTTP endpoint (graceful ASR errors) --------------
@app.post("/voice")
async def voice_endpoint(
    audio: UploadFile = File(...),
    session_id: str = Form("voice"),
    return_json: bool = Form(False),
):
    # 1) Save & normalize
    src_path = _save_upload_to_tmp(audio)
    wav_path = _maybe_to_wav_16k_mono(src_path)

    # 2) STT with guard
    try:
        user_text = stt_listen(wav_path).strip()
    except Exception as e:
        return JSONResponse({"error": "asr_unavailable", "detail": str(e)}, status_code=503)

    if not user_text:
        return JSONResponse({"error": "empty_transcript"}, status_code=400)

    # 3) Turn
    model = db.load_state(session_id) or ChatStateModel(session_id=session_id)
    ensure_system_prompt(model)
    model = _handle_turn(model, user_text)
    db.save_state(session_id, model.model_dump())

    reply_text = next((m.content for m in reversed(model.messages) if m.role == "assistant"), "Okay.")

    # 4) TTS (robust tts_speak returns path or None)
    out_wav = tempfile.mktemp(suffix=".wav")
    wav_out = tts_speak(reply_text, out_wav)

    # 5) Return
    if return_json or not wav_out:
        return {"transcript": user_text, "reply": reply_text}

    def _iterfile(path):
        with open(path, "rb") as f:
            yield from f
    headers = {"X-Transcript": user_text[:2000], "X-Reply": reply_text[:2000]}
    return StreamingResponse(_iterfile(wav_out), media_type="audio/wav", headers=headers)

# ---- Chat WebSocket ----------------------------------------
@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    session_id = None
    try:
        # First message must include session_id
        hello = await ws.receive_json()
        session_id = (hello.get("session_id") or "web") if isinstance(hello, dict) else "web"
        model = db.load_state(session_id) or ChatStateModel(session_id=session_id)
        ensure_system_prompt(model)

        await ws.send_json({"type": "hello", "ok": True, "state": _pack_state(session_id)})

        while True:
            msg = await ws.receive_json()
            text = (msg.get("text") or "").strip()
            if not text:
                continue

            model = _handle_turn(model, text)
            db.save_state(session_id, model.model_dump())

            reply = next((m.content for m in reversed(model.messages) if m.role == "assistant"), "Okay.")
            await ws.send_json({
                "type": "reply",
                "text": reply,
                "state": _pack_state(session_id)
            })

    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass

# ---- Voice WebSocket (graceful Vosk init) -------------------
@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    await ws.accept()

    if os.getenv("USE_ASR","1") != "1":
        await ws.send_json({"error":"asr_disabled","detail":"Set USE_ASR=1 to enable Vosk."})
        await ws.close()
        return

    # Optional hello with session_id
    session_id = "ws-voice"
    try:
        hello = await ws.receive_json()
        if isinstance(hello, dict) and hello.get("session_id"):
            session_id = str(hello["session_id"])
    except Exception:
        # client might send bytes first; that's fine
        pass

    # Try to init recognizer; if it fails, tell the client and close cleanly
    try:
        rec = KaldiRecognizer(_get_vosk_model(), 16000)
        rec.SetWords(True)
    except Exception as e:
        await ws.send_json({"error": "asr_unavailable", "detail": str(e)})
        await ws.close()
        return

    model = db.load_state(session_id) or ChatStateModel(session_id=session_id)
    ensure_system_prompt(model)

    try:
        while True:
            packet = await ws.receive()
            if packet["type"] != "websocket.receive":
                continue

            # audio frames are sent as binary
            if "bytes" in packet and packet["bytes"]:
                data = packet["bytes"]

                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result() or "{}")
                    text = (result.get("text") or "").strip()
                    await ws.send_json({"partial": False, "text": text})

                    if text:
                        model = _handle_turn(model, text)
                        db.save_state(session_id, model.model_dump())

                        reply = next((m.content for m in reversed(model.messages)
                                      if m.role == "assistant"), "Okay.")

                        # TTS -> wav (tts_speak may return None if TTS disabled)
                        tmp_wav = tempfile.mktemp(suffix=".wav")
                        wav_path = tts_speak(reply, tmp_wav)
                        b64 = None
                        if wav_path:
                            with open(wav_path, "rb") as f:
                                b64 = base64.b64encode(f.read()).decode("ascii")

                        await ws.send_json({
                            "reply": reply,
                            "audio_wav_b64": b64,
                            "content_type": "audio/wav" if b64 else None,
                            "final": True,
                            "state": _pack_state(session_id),
                        })
                else:
                    part = json.loads(rec.PartialResult() or "{}")
                    await ws.send_json({"partial": True, "text": part.get("partial", "")})

            # ignore stray text frames here (used only for hello above)

    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass

@app.get("/state")
def get_state(session_id: str = Query(...)):
    model = db.load_state(session_id)
    if not model:
        return {"cart": [], "total": 0, "stage": None, "last_order": {}}
    md = model.metadata or {}
    cart = md.get("cart", [])
    total = sum(int(it.get("price", 0)) * int(it.get("qty", 1)) for it in cart)
    return {
        "cart": cart,
        "total": total,
        "stage": md.get("stage"),
        "last_order": md.get("last_order") or {}
    }

# ---- CLI helpers -------------------------------------------
def _read_user() -> Optional[str]:
    try:
        return input("You: ").strip()
    except EOFError:
        return None

@traceable(name="terminal_turn", tags=["cli", "phase1"])
def _cli_turn(m: ChatStateModel, text: str) -> ChatStateModel:
    return _handle_turn(m, text)



if __name__ == "__main__":
    model = ChatStateModel(session_id="cli", messages=[], metadata={"cart": [], "candidates": [], "confirmed": False})
    ensure_system_prompt(model)
    while True:
        text = _read_user()
        if not text: 
            continue
        if text.lower() in {"quit", "exit", "bye"}:
            print("AI: 👋 Thanks for visiting! See you next time.")
            break
        model = _cli_turn(model, text)
        if model.messages and model.messages[-1].role == "assistant":
            print("AI:", model.messages[-1].content)
        