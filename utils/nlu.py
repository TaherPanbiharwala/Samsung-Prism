# utils/nlu.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from typing_extensions import TypedDict
import os, re, json
import numpy as np
from langsmith import traceable
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ========= Config =========
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EXEC_THRESHOLD = float(os.getenv("NLU_EXEC_THRESHOLD", "0.65"))      # execute if >=
CLARIFY_THRESHOLD = float(os.getenv("NLU_CLARIFY_THRESHOLD", "0.60"))# clarify if <
USE_LOCAL_LLM = os.getenv("NLU_USE_LOCAL_GEMMA", "0") == "1"         # optional
LOCAL_LLM_MODEL = os.getenv("NLU_LOCAL_GEMMA_ID", "google/gemma-2-2b-it")  # example HF id

_embed = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

# Cache: stage -> (labels, vec_syns, protos)
_PROTO_CACHE: Dict[str, Tuple[List[str], np.ndarray, List[str]]] = {}

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", s.lower()).strip()

# ========= Vocab (English-only) =========
VOCAB: Dict[str, Dict[str, List[str]]] = {
    "none": {
        "ordering.lookup": ["menu", "starters", "mains", "beverages", "desserts", "i want to order"],
        "payment": ["pay", "payment", "bill", "split"],
        "chitchat": ["hi", "hello", "thanks", "who are you"],
    },
    "take": {
        "ordering.more": ["more", "more options", "show more", "next", "another", "others"],
        "ordering.take": ["add", "i will have", "i'll have", "i want", "order", "take"],
        "ordering.lookup_refine": ["except", "without", "something else", "other cuisine"],
        "chitchat": ["hi", "thanks", "ok"],
    },
    "confirm": {
        "confirm.yes": ["yes","y","yeah","yep","yup","sure","ok","okay","confirm","place","definitely","absolutely","go ahead"],
        "confirm.no": ["no","n","nope","nah","change","edit","later","cancel","stop"],
        "confirm.ambiguous": ["maybe","not sure","idk","let me think"],
    },
}

# ========= Types =========
class NLUResultBase(TypedDict):
    label: str
    confidence: float
    slots: Dict[str, str]
    needs_clarification: bool

class NLUResult(NLUResultBase, total=False):
    rationale: str  # optional

# ========= Helpers =========
def _stage_space(stage: Optional[str]) -> Dict[str, List[str]]:
    return VOCAB.get(stage or "none", VOCAB["none"])

def _get_proto_cache(stage: Optional[str]):
    key = stage or "none"
    if key in _PROTO_CACHE:
        return _PROTO_CACHE[key]
    space = _stage_space(stage)
    labels, protos = [], []
    for label, syns in space.items():
        for s in syns:
            labels.append(label)
            protos.append(_norm(s) or s)
    vec_syns = np.array([_embed([p])[0] for p in protos], dtype=float) if protos else np.zeros((0, 384))
    _PROTO_CACHE[key] = (labels, vec_syns, protos)
    return _PROTO_CACHE[key]

def _last_assistant(history: Optional[List[Dict[str, str]]]) -> str:
    if not history:
        return ""
    for m in reversed(history):
        if m.get("role") == "assistant":
            return (m.get("content") or "").lower()
    return ""

def _bias_from_history(stage: Optional[str], history: Optional[List[Dict[str, str]]]) -> Optional[str]:
    last_a = _last_assistant(history)
    if not last_a:
        return None
    if "here are some" in last_a or "options:" in last_a or "tell me the numbers" in last_a:
        return "ordering"
    if "place the order" in last_a or "reply with 'yes' or 'no'" in last_a or "shall i place" in last_a:
        return "confirm"
    return None

def _extract_slots_take(text: str) -> Dict[str, str]:
    t = _norm(text)
    has_nums = bool(re.fullmatch(r"\d+(?:\s*(?:,|and)\s*\d+)*", t))
    has_add  = any(k in t for k in ["add ", "i will have", "ill have", "order ", "take "])
    return {"has_numbers": str(has_nums), "has_add_verb": str(has_add)}

# ========= Local/API fallback (optional stubs) =========
def _classify_local_llm(text: str, stage: Optional[str], candidates: List[str]) -> Optional[Tuple[str, float, str]]:
    if not USE_LOCAL_LLM:
        return None
    try:
        from transformers import pipeline
        pipe = pipeline("text-generation", model=LOCAL_LLM_MODEL, device_map="auto")
        prompt = (
            "You are a precise intent classifier. "
            f"Current stage: {stage or 'none'}. "
            f"Valid labels: {', '.join(candidates)}. "
            "Given the user text, output strict JSON with keys {label, confidence, rationale}. "
            "Confidence in [0,1]. Only use the provided labels.\n\n"
            f"USER: {text}\nJSON: "
        )
        out = pipe(prompt, max_new_tokens=128)[0]["generated_text"].split("JSON:")[-1].strip()
        j = json.loads(out)
        label = j.get("label")
        conf = float(j.get("confidence", 0.5))
        rationale = j.get("rationale", "")
        if label in candidates:
            return (label, conf, rationale)
    except Exception:
        return None
    return None

def _classify_api_llm(text: str, stage: Optional[str], candidates: List[str]) -> Optional[Tuple[str, float, str]]:
    return None

# ========= Main =========
@traceable(name="NLU.classify", tags=["nlu"])
def classify(text: str, stage: Optional[str], history: Optional[List[Dict[str, str]]] = None) -> NLUResult:
    t = _norm(text)
    space = _stage_space(stage)
    bias = _bias_from_history(stage, history)

    # --- 1) Rules fast-paths ---
    if stage == "confirm":
        yes_set = set(map(_norm, VOCAB["confirm"]["confirm.yes"]))
        no_set  = set(map(_norm, VOCAB["confirm"]["confirm.no"]))
        if t in yes_set:
            return {"label": "confirm.yes", "confidence": 0.97 if bias == "confirm" else 0.95, "slots": {}, "needs_clarification": False}
        if t in no_set:
            return {"label": "confirm.no", "confidence": 0.97 if bias == "confirm" else 0.95, "slots": {}, "needs_clarification": False}

    if stage == "take":
        # pure numeric picks like "1 and 3"
        if re.fullmatch(r"\d+(?:\s*(?:,|and)\s*\d+)*", t):
            return {"label": "ordering.take", "confidence": 0.92, "slots": _extract_slots_take(text), "needs_clarification": False}

        # qty + name like "1 dal tadka", "2 garlic naan"
        if re.search(r"\b\d+\s+[a-z]", t):
            return {"label": "ordering.take", "confidence": 0.91, "slots": _extract_slots_take(text), "needs_clarification": False}

        # category/diet refine BEFORE verbs/embeddings
        cat_keys = ["starter","starters","main","mains","main course","dessert","desserts","beverage","beverages","drink","drinks","veg","non veg","non-veg"]
        mentions_cat = any(k in t for k in cat_keys)
        has_addverb  = any(k in t for k in ["add ", "i will have", "ill have", "order ", "take "])

        if mentions_cat and not has_addverb:
            return {"label": "ordering.lookup_refine", "confidence": 0.92 if bias == "ordering" else 0.88, "slots": _extract_slots_take(text), "needs_clarification": False}

        # "more", "another", "next"
        if any(k in t for k in ["more", "more options", "show more", "next", "another", "others"]):
            return {"label": "ordering.more", "confidence": 0.92 if bias == "ordering" else 0.90, "slots": _extract_slots_take(text), "needs_clarification": False}

        # add/verb patterns
        if has_addverb:
            return {"label": "ordering.take", "confidence": 0.90, "slots": _extract_slots_take(text), "needs_clarification": False}

    # --- 2) Embedding similarity vs. synonym prototypes (cached) ---
    labels, vec_syns, protos = _get_proto_cache(stage)
    if len(protos):
        vec_text = np.array(_embed([t])[0], dtype=float)
        denom = (np.linalg.norm(vec_syns, axis=1) * (np.linalg.norm(vec_text) + 1e-12))
        sims = (vec_syns @ vec_text) / np.where(denom == 0, 1e-12, denom)
        best_idx = int(np.argmax(sims))
        best_label = labels[best_idx]
        best_sim = float(sims[best_idx])

        if best_sim >= EXEC_THRESHOLD:
            if bias == "confirm" and best_label.startswith("confirm."):
                best_sim = max(best_sim, EXEC_THRESHOLD + 0.02)
            if bias == "ordering" and best_label.startswith("ordering."):
                best_sim = max(best_sim, EXEC_THRESHOLD + 0.02)
            return {"label": best_label, "confidence": min(0.89, best_sim), "slots": (_extract_slots_take(text) if stage == "take" else {}), "needs_clarification": False}

    # --- 3) Local LLM fallback (optional) ---
    candidates = sorted(space.keys())
    llm_out = _classify_local_llm(text, stage, candidates)
    if llm_out:
        lbl, conf, rat = llm_out
        if conf >= EXEC_THRESHOLD:
            return {"label": lbl, "confidence": conf, "slots": (_extract_slots_take(text) if stage=="take" else {}), "needs_clarification": False, "rationale": rat}
        if conf < CLARIFY_THRESHOLD:
            return {"label": lbl, "confidence": conf, "slots": {}, "needs_clarification": True, "rationale": rat}

    # --- 4) API LLM fallback (optional) ---
    api_out = _classify_api_llm(text, stage, candidates)
    if api_out:
        lbl, conf, rat = api_out
        if conf >= EXEC_THRESHOLD:
            return {"label": lbl, "confidence": conf, "slots": (_extract_slots_take(text) if stage=="take" else {}), "needs_clarification": False, "rationale": rat}
        if conf < CLARIFY_THRESHOLD:
            return {"label": lbl, "confidence": conf, "slots": {}, "needs_clarification": True, "rationale": rat}

    # --- Default fallbacks ---
    if stage == "take":
        cat_keys = ["starter","starters","main","mains","main course","dessert","desserts","beverage","beverages","drink","drinks","veg","non veg","non-veg"]
        mentions_cat = any(k in t for k in cat_keys)
        has_nums    = re.fullmatch(r"\d+(?:\s*(?:,|and)\s*\d+)*", t) is not None
        has_addverb = any(k in t for k in ["add ", "i will have", "ill have", "order ", "take "])
        if mentions_cat and not has_nums and not has_addverb:
            return {"label": "ordering.lookup_refine", "confidence": 0.88, "slots": _extract_slots_take(text), "needs_clarification": False}
        return {"label": "ordering.take", "confidence": 0.45, "slots": _extract_slots_take(text), "needs_clarification": True}

    if stage == "confirm":
        return {"label": "confirm.ambiguous", "confidence": 0.45, "slots": {}, "needs_clarification": True}

    return {"label": "chitchat", "confidence": 0.40, "slots": {}, "needs_clarification": True}