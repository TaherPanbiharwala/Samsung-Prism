# nodes/router.py
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any
from utils.validation import validate_node
from langsmith import traceable
class RouterInput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]] = Field(default_factory=list)

class RouterOutput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]]

def route_intent_text(text: str) -> str:
    t = text.lower()
    if "order" in t:
        return "order"
    if "pay" in t or "payment" in t or "bill" in t:
        return "payment"
    return "chitchat"

@validate_node(
    name="NLU_Router",
    tags=["router"],
    input_model=RouterInput,
    output_model=RouterOutput,
)
def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    msgs = state.get("messages", [])
    last = msgs[-1]["content"] if msgs else ""
    intent = route_intent_text(last)
    reply = f"[{intent.upper()} flow coming soon]" if intent != "chitchat" else "Hello! I'm your AI waiter. How can I help you today?"
    msgs.append({"role":"assistant","content": reply})
    state["messages"] = msgs
    return state