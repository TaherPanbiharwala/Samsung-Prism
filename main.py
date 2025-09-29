# main.py
import os, pathlib, tempfile, shutil, subprocess, uuid
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import cast, Optional
from dotenv import load_dotenv
from langsmith import traceable

from utils.nlu import classify
from utils.config import SYSTEM_PROMPT
from graph import build_graph
from state import (
    ChatStateModel, MessageModel, ChatStateTD,
    to_graph_state, from_graph_state,
)
from utils import db
from nodes.voice_hooks import stt_listen, tts_speak

# ---- startup / env -----------------------------------------
load_dotenv()
print("[CFG] USE_WAITER_LLM =", os.getenv("USE_WAITER_LLM"))
print("[CFG] WAITER_LLM_BACKEND =", os.getenv("WAITER_LLM_BACKEND"))

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

def ensure_system_prompt(model: ChatStateModel) -> None:
    if not any(m.role == "system" for m in model.messages):
        model.messages.insert(0, MessageModel(role="system", content=SYSTEM_PROMPT))

# ---- app / graph -------------------------------------------
app = FastAPI()
graph_app = build_graph()

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

# ---- Voice endpoint ----------------------------------------
@app.post("/voice")
async def voice_endpoint(
    audio: UploadFile = File(...),
    session_id: str = Form("voice"),
    return_json: bool = Form(False),
):
    # 1) Save & normalize audio
    src_path = _save_upload_to_tmp(audio)
    wav_path = _maybe_to_wav_16k_mono(src_path)

    # 2) STT → text
    user_text = stt_listen(wav_path).strip()
    if not user_text:
        return JSONResponse({"error": "empty_transcript"}, status_code=400)

    # 3) Run graph turn
    model = db.load_state(session_id) or ChatStateModel(session_id=session_id)
    ensure_system_prompt(model)
    model = _handle_turn(model, user_text)
    db.save_state(session_id, model.model_dump())

    reply_text = next((m.content for m in reversed(model.messages) if m.role == "assistant"), "Okay.")

    # 4) TTS → wav
    out_wav = tempfile.mktemp(suffix=".wav")
    tts_speak(reply_text, out_wav)

    if return_json:
        return {"transcript": user_text, "reply": reply_text}

    # 5) Stream wav
    def _iterfile(path):
        with open(path, "rb") as f:
            yield from f

    headers = {"X-Transcript": user_text[:2000], "X-Reply": reply_text[:2000]}
    return StreamingResponse(_iterfile(out_wav), media_type="audio/wav", headers=headers)

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
        if not text: continue
        if text.lower() in {"quit", "exit", "bye"}:
            print("AI: 👋 Thanks for visiting! See you next time.")
            break
        model = _cli_turn(model, text)
        if model.messages and model.messages[-1].role == "assistant":
            print("AI:", model.messages[-1].content)