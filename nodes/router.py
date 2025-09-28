# nodes/router.py
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any
from utils.validation import validate_node

class RouterInput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]] = Field(default_factory=list)

class RouterOutput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]]

def route_intent_text(text: str) -> str:
    t = text.lower()
    if "order" in t or "starter" in t or "pizza" in t or "menu" in t:
        return "order"
    if "pay" in t or "payment" in t or "bill" in t:
        return "payment"
    return "chitchat"

@validate_node(name="NLU_Router", tags=["router"], input_model=RouterInput, output_model=RouterOutput)
def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    msgs = state.get("messages", [])
    last = msgs[-1]["content"] if msgs else ""
    intent = route_intent_text(last)

    # Route into ordering flow by setting a flag instead of replying a placeholder
    # nodes/router.py  (inside router_node)
    if intent == "order":
        md = state.setdefault("metadata", {})
        md["route"] = "order"
        md["stage"] = md.get("stage") or "menu"
        msgs.append({"role": "assistant",
                    "content": "Of course! What kind of dishes are you in the mood for? (starters / mains / beverages / desserts) Any dietary pref (veg / non-veg)?"})
        state["messages"] = msgs
        return state

    # not order -> clear route/stage so we don't persist into ordering by mistake
    md = state.setdefault("metadata", {})
    md["route"] = None
    md["stage"] = None

    if intent == "chitchat":
        msgs.append({"role":"assistant","content":"Hi! I’m your AI waiter. Ask me about the menu, or say what you’d like."})
    else:
        msgs.append({"role":"assistant","content":"[PAYMENT flow coming soon]"})
    state["messages"] = msgs
    return state