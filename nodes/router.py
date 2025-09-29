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
    stage: Optional[str] = md.get("stage")

    # --- Step 1: try strict LLM JSON classification ---
    schema_hint = '{"intent": "ordering.take", "slots": {"items": []}}'
    system = """
    You are an NLU classifier for a restaurant ordering assistant.
    Valid intents:
      - ordering.lookup   (see menu items)
      - ordering.more     (ask for more options)
      - ordering.take     (add items to cart)
      - confirm.yes       (confirm order)
      - confirm.no        (reject order)
      - chitchat          (small talk, greetings, unrelated chat)

    Output ONLY a JSON object with fields: intent (string), slots (object).
    No explanations, no text outside JSON.
    """

    parsed = generate_json(system=system, user=last_user, schema_hint=schema_hint, max_tokens=64)
    label, slots = parsed.get("intent", "chitchat"), parsed.get("slots", {})

    # --- Step 2: fallback regex overrides ---
    lu = last_user.lower().strip()
    if re.fullmatch(r"\d+(?:\s*(?:,|and)\s*\d+)*", lu):
        label = "ordering.take"
        slots = {"has_numbers": "True"}

    if lu in {"yes", "ok", "okay", "y", "yeah"}:
        label = "confirm.yes"
    elif lu in {"no", "n", "nope", "nah"}:
        label = "confirm.no"

    # --- Step 3: update state ---
    md["last_intent"] = label
    md["last_slots"] = slots
    md["route"] = "order" if label.startswith(("ordering.", "confirm.")) else None

    if label in {"ordering.lookup","ordering.more"}:
        md["stage"] = "menu"
        if label == "ordering.more":
            md["page"] = int(md.get("page",0)) + 1
    elif label == "ordering.take":
        md["stage"] = "take"
    elif label.startswith("confirm."):
        md["stage"] = "confirm"
        md["confirm_signal"] = label
    else:
        md["stage"] = None

    return state