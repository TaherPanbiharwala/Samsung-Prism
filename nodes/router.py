from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
# from Flow import state          # ← REMOVED
from state import ChatStateTD, ChatStateModel, MessageModel
from utils.validation import validate_node
from utils.nlu import classify
import os, re

# --- helper -------------------------------------------------
_MENU_WORDS = {
    "starter","starters","main","mains","main course",
    "beverage","beverages","drink","drinks",
    "dessert","desserts","sweet","sweets",
    "veg","non veg","non-veg"
}
def _looks_like_menu_query(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in _MENU_WORDS)

def _facet_from_text(text: str) -> Dict[str, Any]:
    t = text.lower()
    cat = None
    if "starter" in t: cat = "starter"
    elif "main" in t or "main course" in t: cat = "main"
    elif "dessert" in t or "sweet" in t: cat = "dessert"
    elif "drink" in t or "beverage" in t: cat = "beverage"
    return {
        "veg": ("veg" in t and "non veg" not in t and "non-veg" not in t),
        "nonveg": ("non veg" in t or "non-veg" in t),
        "category": cat,
    }
# ------------------------------------------------------------

class RouterInput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]] = Field(default_factory=list)

class RouterOutput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]]

def _last_assistant(msgs: List[Dict[str,str]]) -> str:
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            return (m.get("content") or "").lower()
    return ""

@validate_node(name="NLU_Router", tags=["router"], input_model=RouterInput, output_model=RouterOutput)
def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    msgs: List[Dict[str, str]] = state.get("messages", [])
    md: Dict[str, Any] = state.setdefault("metadata", {})

    if "_awaiting_worker" not in md:
        md["_awaiting_worker"] = True

    # 🔹 Debug override
    forced = os.getenv("ROUTER_FORCE")
    if forced:
        md["last_intent"] = forced
        if forced in {"ordering.lookup", "ordering.lookup_refine", "ordering.more"}:
            md["stage"] = "menu"
        elif forced == "ordering.take":
            md["stage"] = "take"
        elif forced.startswith("confirm."):
            md["stage"] = "confirm"
        else:
            md["stage"] = "menu"
        return state

    # pull once
    last_user = next((m.get("content","") for m in reversed(msgs) if m.get("role")=="user"), "")
    stage: Optional[str] = md.get("stage")
    lu = (last_user or "").strip().lower()

    # ✅ short-circuit numeric picks like "1 and 3"
    if re.fullmatch(r"\d+(?:\s*(?:,|and)\s*\d+)*", lu):
        md["last_intent"] = "ordering.take"
        md["route"] = "order"
        md["stage"] = "take"
        md["last_slots"] = {"has_numbers":"True","has_add_verb":"False"}
        return state

    # ✅ early confirm short-circuit on yes/no
    _yes = {"yes","y","yeah","yep","yup","sure","ok","okay","confirm","place","definitely","absolutely","go ahead"}
    _no  = {"no","n","nope","nah","change","edit","later","cancel","stop"}
    last_a = _last_assistant(msgs)
    asked_to_confirm = any(
        key in last_a
        for key in [
            "shall i place the order",
            "place the order",
            "reply with 'yes' or 'no'",
            "please reply with 'yes'",
            "shall i place",
        ]
    )

    # NEW: allow confirmation if stage==confirm OR the last assistant asked to confirm
    if (stage == "confirm" or asked_to_confirm) and lu in _yes and (md.get("cart") or []):
        md["last_intent"] = "confirm.yes"
        md["route"] = "order"
        md["stage"] = "confirm"
        md["confirm_signal"] = "confirm.yes"
        return state

    if (stage == "confirm" or asked_to_confirm) and lu in _no and (md.get("cart") or []):
        md["last_intent"] = "confirm.no"
        md["route"] = "order"
        md["stage"] = "confirm"
        md["confirm_signal"] = "confirm.no"
        return state

    # ✅ quantified dish names like "2 paneer 65", "3 dal tadka" → take
    # (works even without explicit "add" verb)
    if re.search(r"\b\d+\s*x?\s+[a-z]", lu):
        md["last_intent"] = "ordering.take"
        md["route"] = "order"
        md["stage"] = "take"
        # slots hint optional; take_order_node also regex-checks
        md["last_slots"] = {"has_numbers":"False","has_add_verb":"True"}
        return state

    # ---- base NLU
    res = classify(last_user, stage=stage, history=msgs)
    label = res.get("label","chitchat")
    conf  = float(res.get("confidence",0.0))
    slots = res.get("slots") or {}

    # ---- context overrides while in 'take'
    if stage == "take" and md.get("candidates"):
        low = lu
        cand_names = [str(c.get("name","")).lower() for c in md["candidates"]]
        def mentions_any_name(text,names):
            for n in names:
                if not n: continue
                if n in text: return True
                toks = [t for t in re.split(r"[^a-z0-9]+", n) if len(t)>=3]
                if any(t in text for t in toks): return True
            return False
        if re.search(r"\b\d+(?:\s*(?:,|and)\s*\d+)*\b", low):
            label = "ordering.take"
        elif mentions_any_name(low, cand_names):
            label = "ordering.take"

    # ---- obvious menu-like → refine (unless already take)
    if label != "ordering.take" and _looks_like_menu_query(last_user):
        label = "ordering.lookup_refine"
        conf = max(conf, 0.9)
        md["facets"] = _facet_from_text(last_user)

    # persist outcome
    md["last_intent"] = label
    md["last_conf"]   = conf
    md["last_slots"]  = slots

    # routing
    if label in {"ordering.lookup","ordering.lookup_refine","ordering.more"}:
        md["route"] = "order"; md["stage"] = "menu"
        md.setdefault("facets", _facet_from_text(last_user))
        if label == "ordering.more":
            md["page"] = int(md.get("page",0)) + 1
        return state

    if label == "ordering.take":
        md["route"] = "order"; md["stage"] = "take"
        has_nums = str(slots.get("has_numbers","False")).lower()=="true"
        has_add  = str(slots.get("has_add_verb","False")).lower()=="true"
        if not (md.get("candidates") or []) and not (has_nums or has_add):
            md["last_intent"] = "ordering.lookup_refine"
            md["stage"] = "menu"
            md["facets"] = _facet_from_text(last_user)
        return state

    if label.startswith("confirm."):
        md["route"] = "order"; md["stage"] = "confirm"
        md["confirm_signal"] = label
        return state

    md["route"] = None
    md["last_intent"] = "chitchat"
    return state