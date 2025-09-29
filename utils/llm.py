# utils/llm.py
from __future__ import annotations
from typing import Dict, Any
import os, json, requests, re

# Always use Ollama backend
_MODEL = os.getenv("OLLAMA_MODEL", "waiter-gguf:latest")
_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

_TEMP = float(os.getenv("WAITER_TEMP", "0.3"))
_TOP_P = float(os.getenv("WAITER_TOP_P", "0.9"))

# --------------------------------------------------
# Core Ollama call
# --------------------------------------------------
def _call_ollama(prompt: str, max_tokens: int) -> str:
    try:
        resp = requests.post(
            f"{_OLLAMA_URL}/api/generate",
            json={
                "model": _MODEL,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": _TEMP,
                    "top_p": _TOP_P,
                },
            },
            timeout=60,
        )
        resp.raise_for_status()

        text = ""
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = line.decode("utf-8")
                obj = json.loads(data)
                if "response" in obj:
                    text += obj["response"]
                if obj.get("done", False):
                    break
            except Exception:
                continue
        return text.strip()
    except Exception as e:
        print("[LLM Ollama Error]", e)
        return ""

# --------------------------------------------------
# JSON Generation
# --------------------------------------------------
def generate_json(system: str, user: str, schema_hint: str, max_tokens: int = 128) -> dict:
    """
    Ask Ollama to output strict JSON. If it fails, fallback to schema_hint.
    """
    prompt = (
        f"{system.strip()}\n\n"
        f"Schema example: {schema_hint}\n\n"
        f"User: {user}\nAssistant:"
    )

    raw = _call_ollama(prompt, max_tokens)
    if not raw:
        return json.loads(schema_hint)

    # Extract JSON object from raw text
    try:
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print("[generate_json parse error]", e, "raw:", raw)

    # Fallback heuristics for yes/no
    low = raw.lower()
    if "yes" in low and "no" not in low:
        return {"decision": "yes"}
    if "no" in low and "yes" not in low:
        return {"decision": "no"}
    if "unclear" in low:
        return {"decision": "unclear"}

    return json.loads(schema_hint)

# --------------------------------------------------
# Natural-language waiter responses
# --------------------------------------------------
def generate_waiter(system: str, context_menu: str, user: str, max_tokens: int = 128) -> str:
    prompt = (
        f"{system}\n\n"
        f"CONTEXT_MENU:\n{context_menu}\n\n"
        f"User: {user}\nWaiter:"
    )
    out = _call_ollama(prompt, max_tokens)
    return out if out else "Here are some options. Please choose by number."

# Back-compat
def chat(system: str, context_menu: str, user: str, max_tokens: int = 128) -> str:
    return generate_waiter(system, context_menu, user, max_tokens)