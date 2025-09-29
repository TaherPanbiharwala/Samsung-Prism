# router.py
from typing import List, Dict, Any, Optional
import os, re, json
from pydantic import BaseModel, Field

from state import ChatStateTD, ChatStateModel, MessageModel
from utils.validation import validate_node
from utils.llm import generate_json


class RouterInput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]] = Field(default_factory=list)

class RouterOutput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]]


# --- Regex helpers ---
_ADD_VERBS = re.compile(r"\b(add|get\s+me|give\s+me|i(?:'| wi)ll\s+have|i\s+want|we(?:'ll| wi)ll\s+have|order|take)\b", re.I)
_REMOVE_VERBS = re.compile(r"\b(remove|delete|cancel|drop|take\s*out)\b|(?:\bno\s+more\b)|\bwithout\b", re.I)
_NUM_LIST = re.compile(r"^\s*\d+(?:\s*(?:,|and)\s*\d+)*\s*$", re.I)
_QTY_NAME = re.compile(r"\b\d+\s*x?\s+[a-z]", re.I)
_QUIT_WORDS = re.compile(r"\b(quit|exit|goodbye|bye|done|finished|thanks)\b", re.I)
_EXIT_WORDS = re.compile(r"\b(quit|exit|bye|goodbye)\b", re.I)

# Payment regex
_BILL_WORDS = re.compile(r"\bbill|check|receipt|total\b", re.I)
_SPLIT_WORDS = re.compile(r"\bsplit\b", re.I)
_PAY_WORDS = re.compile(r"\bpay|payment|checkout|settle\b", re.I)
_FEEDBACK_WORDS = re.compile(r"\bfeedback|review|rating|comment\b", re.I)


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
    # Exit intent short-circuit (optional)
    if _EXIT_WORDS.search(lu):
        md["last_intent"] = "app.exit"
        md["route"] = None
        md["stage"] = None
        return state
    stage: Optional[str] = md.get("stage")
    cands = md.get("candidates", []) or []

    # --- Step 1: LLM classification ---
    schema_hint = '{"intent": "chitchat", "slots": {}}'
    system = """
    You are an NLU classifier for a restaurant assistant.
    Valid intents:
      - ordering.lookup   (see menu items)
      - ordering.more     (ask for more options)
      - ordering.take     (add or remove items to/from cart)
      - confirm.yes       (confirm order)
      - confirm.no        (reject order)
      - payment.bill      (ask to show/generate bill)
      - payment.split     (ask to split bill)
      - payment.pay       (proceed to payment)
      - payment.feedback  (give feedback)
      - chitchat          (small talk, greetings, unrelated chat)
    Output ONLY a JSON object with fields: intent (string), slots (object).
    """
    parsed = generate_json(system=system, user=last_user, schema_hint=schema_hint, max_tokens=64)
    label, slots = parsed.get("intent", "chitchat"), parsed.get("slots", {})

    # --- Step 2: Regex overrides ---
    
        # --- Step 2: Regex overrides ---

    # --- Step 2: Regex overrides ---

    # 🔸 Payment FIRST, and prioritize SPLIT over BILL
    if _SPLIT_WORDS.search(lu):
        label = "payment.split"
    elif _PAY_WORDS.search(lu):
        label = "payment.pay"
    elif _BILL_WORDS.search(lu):
        label = "payment.bill"
    elif _FEEDBACK_WORDS.search(lu):
        label = "payment.feedback"
    elif _QUIT_WORDS.search(lu):
        label = "quit"

    # 🔹 Ordering AFTER payment checks
    elif _NUM_LIST.fullmatch(lu):
        label = "ordering.take"; slots = {"has_numbers": "True"}
    elif _QTY_NAME.search(lu) or _ADD_VERBS.search(lu):
        label = "ordering.take"; slots = {**slots, "has_add": "True"}
    elif _REMOVE_VERBS.search(lu):
        label = "ordering.take"; slots = {**slots, "has_remove": "True"}

    # Yes/No confirmations
    if lu in {"yes","y","ok","okay","yeah","yep","confirm","place"}:
        label = "confirm.yes"
    elif lu in {"no","n","nope","nah","cancel","change"}:
        label = "confirm.no"

    # --- Step 3: Update state & route ---
    md["last_intent"] = label
    md["last_slots"] = slots

    if label.startswith("ordering.") or label.startswith("confirm."):
        md["route"] = "order"
    elif label.startswith("payment."):
        md["route"] = "payment"
    else:
        md["route"] = None

    # Stage handling
    if label == "quit":
        md["stage"] = None
        return state
    if label in {"ordering.lookup","ordering.more"}:
        md["stage"] = "menu"
        if label == "ordering.more":
            md["page"] = int(md.get("page", 0)) + 1
    elif label == "ordering.take":
        md["stage"] = "take"
    elif label.startswith("confirm."):
        md["stage"] = "confirm"; md["confirm_signal"] = label
    elif label == "payment.bill":
        md["stage"] = "payment.bill"
    elif label == "payment.split":
        md["stage"] = "payment.split"
    elif label == "payment.pay":
        md["stage"] = "payment.gateway"
    elif label == "payment.feedback":
        md["stage"] = "payment.feedback"
    else:
        md["stage"] = None

    return state