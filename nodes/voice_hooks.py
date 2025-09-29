# nodes/voice_hooks.py
# ASR (Vosk) + TTS (Piper), with lazy init and type-checker friendly imports.

from __future__ import annotations

import os
import subprocess
import wave
from typing import Any, Optional

# --- Import Vosk types in a type-checker-safe way ---
try:  # keep Pylance/mypy happy even if vosk isn't installed at analysis time
    from vosk import Model as VoskModel, KaldiRecognizer  # type: ignore
except Exception:  # pragma: no cover
    VoskModel = Any            # type: ignore
    KaldiRecognizer = Any      # type: ignore

# --- Config (use absolute paths if possible) ---
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "./models/vosk")
PIPER_BIN       = os.getenv("PIPER_BIN", "./models/piper/piper")
PIPER_VOICE     = os.getenv("PIPER_VOICE", "./models/piper/en_US-amy-medium.onnx")

# --- Lazy, cached Vosk model ---
_VOSK_MODEL: Optional[VoskModel] = None

def _get_vosk_model() -> VoskModel:
    """
    Lazily load and cache the Vosk model.
    Avoids constructing at import time (prevents crashes if env var not set yet).
    """
    global _VOSK_MODEL
    if _VOSK_MODEL is None:
        # Pylance used to complain: "Variable not allowed in type expression"
        # by annotating with the runtime Model directly. Using alias fixes it.
        _VOSK_MODEL = VoskModel(VOSK_MODEL_PATH)  # type: ignore[call-arg]
    return _VOSK_MODEL

# --- ASR (speech -> text) ---
def stt_listen(wav_path: str) -> str:
    """
    Transcribe a 16 kHz mono WAV file with Vosk.
    Returns lowercased plain text (Vosk's default).
    """
    mdl = _get_vosk_model()
    with wave.open(wav_path, "rb") as wf:
        rec = KaldiRecognizer(mdl, wf.getframerate())  # type: ignore[call-arg]
        text_fragments: list[str] = []
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                text_fragments.append(rec.Result())
        text_fragments.append(rec.FinalResult())

    # Join and extract the "text" field if present
    raw = "".join(text_fragments)
    try:
        import json
        obj = json.loads(raw) if raw.strip().startswith("{") else None
        if isinstance(obj, dict) and "text" in obj:
            return str(obj["text"]).strip()
        return raw.strip()
    except Exception:
        return raw.strip()

# --- TTS (text -> wav) ---
def tts_speak(text: str, out_path: str) -> str:
    """
    Synthesize speech with Piper to out_path (wav).
    """
    # Make sure we call the subprocess module, not a shadowed name.
    run = getattr(subprocess, "run")  # avoids Pylance “Object of type None cannot be called”
    cmd = [PIPER_BIN, "--model", PIPER_VOICE, "--output_file", out_path]
    # Piper reads the text from stdin
    run(cmd, input=text.encode("utf-8"), check=False)
    return out_path