# utils/llm.py
from __future__ import annotations
from typing import Dict, Any
import os, json, requests, re

_BACKEND    = os.getenv("WAITER_LLM_BACKEND", "ollama").lower()
_TEMP       = float(os.getenv("WAITER_TEMP", "0.3"))
_TOP_P      = float(os.getenv("WAITER_TOP_P", "0.9"))
_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "waiter-gguf:latest")

def _ollama_generate(prompt: str, max_tokens: int, *, json_mode: bool = False) -> str:
    """
    Call Ollama /api/generate and accumulate the streamed 'response' field.
    If json_mode=True, we set format='json' so the model MUST return a single JSON object.
    """
    try:
        payload: Dict[str, Any] = {
            "model": _OLLAMA_MODEL,
            "prompt": prompt,
            "options": {"num_predict": max_tokens, "temperature": _TEMP, "top_p": _TOP_P},
        }
        if json_mode:
            payload["format"] = "json"  # <-- forces JSON output

        resp = requests.post(f"{_OLLAMA_URL}/api/generate", json=payload, timeout=60, stream=True)
        resp.raise_for_status()

        text = ""
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
                if "response" in obj:
                    text += obj["response"]
                if obj.get("done", False):
                    break
            except Exception:
                # ignore malformed partials
                continue
        return text.strip()
    except Exception as e:
        print("[LLM Ollama Error]", e)
        return ""

def generate_waiter(system: str, context_menu: str, user: str, max_tokens: int = 128) -> str:
    prompt = f"{system}\n\nCONTEXT_MENU:\n{context_menu}\n\nUser: {user}\nWaiter:"
    out = _ollama_generate(prompt, max_tokens, json_mode=False) or ""
    # scrub common end tokens
    out = re.sub(r"(?:</s>|<\|eot\|>|<eos>)+", "", out, flags=re.I).strip()
    return out or "Here are some menu options. Please choose by number."


def generate_json(system: str, user: str, schema_hint: str, max_tokens: int = 128) -> dict:
    """
    Ask the LLM for strict JSON output via Ollama. We set format='json' so
    the response is a single valid JSON object (no code fences / prose).
    """
    prompt = (
        f"{system.strip()}\n\n"
        f"Schema example: {schema_hint}\n\n"
        f"User: {user}\n"
        f"Assistant: Reply ONLY with a valid JSON object."
    )

    # Use the function you actually defined
    raw = _ollama_generate(prompt, max_tokens, json_mode=True).strip()

    # First, try straight JSON (preferred path with format='json')
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Fallback: extract first {...} block if the model still wrapped anything
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            return json.loads(m.group(0))
    except Exception as e:
        print("[generate_json parse error]", e, "raw:", raw)

    # Last resort: return the provided schema hint
    return json.loads(schema_hint)