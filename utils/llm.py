# utils/llm.py
from __future__ import annotations
from typing import Optional
import os

# simple, resilient wrapper around llama.cpp
_BACKEND = os.getenv("WAITER_LLM_BACKEND", "llamacpp")   # "llamacpp" or "disabled"
_MODEL   = os.getenv("WAITER_LLM_MODEL", "/Users/taherpanbiharwala/Desktop/Win/ollama/models/gemma-ft/gemma-ft-q8_v2.gguf")
_CTX     = int(os.getenv("WAITER_LLM_CTX", "2048"))
_TEMP    = float(os.getenv("WAITER_TEMP", "0.3"))
_TOP_P   = float(os.getenv("WAITER_TOP_P", "0.9"))

_llm = None

def _get_llm():
    global _llm
    if _llm is not None:
        return _llm
    if _BACKEND != "llamacpp":
        _llm = None
        return _llm
    try:
        from llama_cpp import Llama
        _llm = Llama(model_path=_MODEL, n_ctx=_CTX, n_threads=max(2, (os.cpu_count() or 4)))
        print(f"[LLM] loaded: {_MODEL} ctx={_CTX}")
        print(f"[LLM] backend={_BACKEND} model={_MODEL}")
    except Exception as e:
        print(f"[LLM] load failed: {e}")
        _llm = None
    return _llm

def generate_waiter(system: str, context_menu: str, user: str, max_tokens: int = 128) -> str:
    """
    One-shot completion style for speed; avoids passing backend-specific args that may not exist.
    """
    prompt = (
        f"{system}\n\nCONTEXT_MENU:\n{context_menu}\n\n"
        f"User: {user}\nWaiter:"
    )

    llm = _get_llm()
    if not llm:
        # fast fallback if local model is unavailable
        return "Here are a few options from the menu. Tell me the numbers you like and quantities."

    try:
        # preferred call
        out = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=_TEMP,
            top_p=_TOP_P,
            stop=["User:", "</s>", "\n\n"]
            # DO NOT pass cache_prompt – not all versions support it
        )
        return out["choices"][0]["text"].strip()
    except TypeError:
        # ultra-minimal fallback for older versions
        out = llm.create_completion(prompt=prompt, max_tokens=max_tokens)
        return out["choices"][0]["text"].strip()
    except Exception:
        return "Let me know: starters, mains, beverages or desserts? I’ll suggest a few quick picks."

def chat(system: str, context_menu: str, user: str, max_tokens: int = 128) -> str:
    """Back-compat alias for old imports."""
    return generate_waiter(system, context_menu, user, max_tokens)