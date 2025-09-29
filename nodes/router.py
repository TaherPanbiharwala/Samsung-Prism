# router.py
from typing import List, Dict, Any, Optional
import os, re, json
from pydantic import BaseModel, Field

from state import ChatStateTD, ChatStateModel, MessageModel
from utils.validation import validate_node
from utils.llm import generate_waiter, generate_json


class RouterInput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]] = Field(default_factory=list)

class RouterOutput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]]


# 🔹 Strong signals for add/remove in free text
_ADD_VERBS = re.compile(
    r"\b(add|get\s+me|give\s+me|i(?:'| wi)ll\s+have|i\s+want|we(?:'ll| wi)ll\s+have|order|take)\b",
    re.I
)
_REMOVE_VERBS = re.compile(
    r"\b(remove|delete|cancel|drop|take\s*out)\b|(?:\bno\s+more\b)|\bwithout\b",
    re.I
)
# numbers like "1 and 3" or "2, 5" or quantified name "2 paneer"
_NUM_LIST = re.compile(r"^\s*\d+(?:\s*(?:,|and)\s*\d+)*\s*$", re.I)
_QTY_NAME = re.compile(r"\b\d+\s*x?\s+[a-z]", re.I)


def _facet_from_text(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    cat = None
    if "starter" in t: cat = "starter"
    elif "main course" in t or "main" in t: cat = "main"
    elif "dessert" in t or "sweet" in t: cat = "dessert"
    elif "drink" in t or "beverage" in t: cat = "beverage"
    return {
        "veg": ("veg" in t and "non veg" not in t and "non-veg" not in t),
        "nonveg": ("non veg" in t or "non-veg" in t),
        "category": cat,
    }


@validate_node(name="NLU_Router", tags=["router"], input_model=RouterInput, output_model=RouterOutput)
def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    msgs = state.get("messages", [])
    md: Dict[str, Any] = state.setdefault("metadata", {})

    last_user = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
    lu = (last_user or "").strip().lower()
    stage: Optional[str] = md.get("stage")
    cands = md.get("candidates", []) or []

    # --- Step 1: LLM JSON classification (your existing approach) ---
    schema_hint = '{"intent": "chitchat", "slots": {}}'
    system = """
    You are an NLU classifier for a restaurant ordering assistant.
    Valid intents:
      - ordering.lookup   (see menu items)
      - ordering.more     (ask for more options)
      - ordering.take     (add or remove items to/from cart)
      - confirm.yes       (confirm order)
      - confirm.no        (reject order)
      - chitchat          (small talk, greetings, unrelated chat)
    Output ONLY a JSON object with fields: intent (string), slots (object).
    """
    parsed = generate_json(system=system, user=last_user, schema_hint=schema_hint, max_tokens=64)
    label, slots = parsed.get("intent", "chitchat"), parsed.get("slots", {})

    # --- Step 2: Strong regex overrides (authoritative) ---
    # numeric picks like "1 and 3"
    if _NUM_LIST.fullmatch(lu):
        label = "ordering.take"
        slots = {"has_numbers": "True"}

    # quantified-name ("2 paneer 65") or explicit add verbs
    elif _QTY_NAME.search(lu) or _ADD_VERBS.search(lu):
        label = "ordering.take"
        slots = {**slots, "has_add": "True"}

    # deletions: "remove X", "delete X", "no more X", "without onions"
    elif _REMOVE_VERBS.search(lu):
        label = "ordering.take"
        slots = {**slots, "has_remove": "True"}

    # if we are already showing options (have candidates), and the user mentions any candidate-ish words,
    # treat as take (covers: "garlic naan and butter chicken", "add tikka", etc.)
    elif stage in {None, "take"} and cands:
        cand_names = [str(c.get("name", "")).lower() for c in cands]
        tokens = set([t for t in re.split(r"[^a-z0-9]+", lu) if len(t) >= 3])
        if any(any(tok in nm for tok in tokens) for nm in cand_names):
            label = "ordering.take"

    # yes / no confirmations (quick path)
    if lu in {"yes","y","ok","okay","yeah","yep","confirm","place"}:
        label = "confirm.yes"
    elif lu in {"no","n","nope","nah","cancel","change"}:
        label = "confirm.no"

    # --- Step 3: update state & route ---
    md["last_intent"] = label
    md["last_slots"] = slots
    md["route"] = "order" if label.startswith(("ordering.", "confirm.")) else None

    if label in {"ordering.lookup","ordering.more"}:
        md["stage"] = "menu"
        if label == "ordering.more":
            md["page"] = int(md.get("page", 0)) + 1
    elif label == "ordering.take":
        md["stage"] = "take"
    elif label.startswith("confirm."):
        md["stage"] = "confirm"
        md["confirm_signal"] = label
    else:
        md["stage"] = None

    return state