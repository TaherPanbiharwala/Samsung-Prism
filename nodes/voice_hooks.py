# nodes/voice_hooks.py
# nodes/voice_hooks.py
from __future__ import annotations
import os, glob, wave, subprocess
from typing import Any, Optional

try:
    from vosk import Model as VoskModel, KaldiRecognizer  # type: ignore
except Exception:
    VoskModel = Any            # type: ignore
    KaldiRecognizer = Any      # type: ignore

# Resolve and normalize paths
def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

# Read env (may be relative), but don't trust it blindly
_ENV_VOSK = os.getenv("VOSK_MODEL_PATH", "./models/vosk")

# Cached model
_VOSK_MODEL: Optional[VoskModel] = None

def _find_vosk_model_dir() -> str:
    """
    Returns a valid model directory or raises an Exception with a helpful message.
    Tries:
      1) VOSK_MODEL_PATH (absolute/expanded)
      2) ./models/vosk-model-*/ (first match)
      3) ./models/vosk (symlink/dir)
    """
    # 1) env path
    cand = _abs(_ENV_VOSK)
    if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "conf")):
        return cand

    # 2) auto-detect a vosk-model-* folder under ./models
    here = _abs(".")
    models_dir = os.path.join(here, "models")
    if os.path.isdir(models_dir):
        matches = sorted(glob.glob(os.path.join(models_dir, "vosk-model-*")))
        for m in matches:
            if os.path.isdir(os.path.join(m, "conf")):
                return m

        # 3) ./models/vosk (symlink/dir) if it has "conf"
        fallback = os.path.join(models_dir, "vosk")
        if os.path.isdir(fallback) and os.path.isdir(os.path.join(fallback, "conf")):
            return fallback

    raise Exception(
        "Vosk model not found. Set VOSK_MODEL_PATH to your model folder "
        "(e.g. /absolute/path/to/models/vosk-model-small-en-us-0.15) "
        "or create a symlink ./models/vosk -> vosk-model-... ."
    )

def _get_vosk_model():
    global _VOSK_MODEL
    if _VOSK_MODEL is None:
        model_dir = _find_vosk_model_dir()
        _VOSK_MODEL = VoskModel(model_dir)  # type: ignore
    return _VOSK_MODEL

# --- the rest of your stt_listen/tts_speak stays the same ---

def stt_listen(wav_path: str) -> str:
    """Transcribe 16kHz mono WAV with Vosk → plain text."""
    mdl = _get_vosk_model()
    with wave.open(wav_path, "rb") as wf:
        rec = KaldiRecognizer(mdl, wf.getframerate())  # type: ignore
        parts = []
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                parts.append(rec.Result())
        parts.append(rec.FinalResult())
    raw = "".join(parts).strip()
    # extract "text" if JSON, else return raw
    try:
        import json
        obj = json.loads(raw) if raw.startswith("{") else None
        return (obj.get("text") or "").strip() if isinstance(obj, dict) else raw
    except Exception:
        return raw

# ---------- TTS (Coqui -> Edge fallback) ----------
COQUI_MODEL = os.getenv("COQUI_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")
EDGE_VOICE  = os.getenv("EDGE_TTS_VOICE", "en-US-JennyNeural")

def _coqui_say(text: str, out_path: str) -> bool:
    try:
        from TTS.api import TTS  # pip install TTS
        tts = TTS(COQUI_MODEL)
        # Most single-speaker models don’t need speaker_lang/speaker_id
        tts.tts_to_file(text=text, file_path=out_path)
        return True
    except Exception as e:
        print(f"[TTS] Coqui failed: {e}")
        return False

async def _edge_say_async(text: str, out_path: str) -> bool:
    try:
        import edge_tts  # pip install edge-tts
        communicate = edge_tts.Communicate(text, EDGE_VOICE)
        await communicate.save(out_path)
        return True
    except Exception as e:
        print(f"[TTS] Edge failed: {e}")
        return False

def _edge_say(text: str, out_path: str) -> bool:
    try:
        return asyncio.run(_edge_say_async(text, out_path))
    except RuntimeError:
        # If already in an event loop (rare for FastAPI worker), use a new loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_edge_say_async(text, out_path))
        finally:
            loop.close()
    except Exception as e:
        print(f"[TTS] Edge runtime failed: {e}")
        return False

def tts_speak(text: str, out_path: str) -> Optional[str]:
    """
    Try Coqui first, then Edge. Return out_path if successful, else None.
    Never raises — lets the caller fall back to text reply.
    """
    text = (text or "").strip()
    if not text:
        return None

    # Coqui first
    if _coqui_say(text, out_path):
        return out_path

    # Edge fallback
    if _edge_say(text, out_path):
        return out_path

    print("[TTS] All engines failed; returning None (use text fallback).")
    return None