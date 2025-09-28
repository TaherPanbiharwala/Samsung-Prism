# nodes/management.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from utils.validation import validate_node
from langsmith import traceable
from utils.context import format_context_menu
from state import MessageModel  # import if not already


class ChitchatInput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]] = Field(default_factory=list, min_length=1)

class ChitchatOutput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]] = Field(default_factory=list, min_length=1)

@validate_node(name="Chitchat", tags=["management","chitchat"], input_model=ChitchatInput, output_model=ChitchatOutput)
def chitchat_node(state: Dict[str, Any]) -> Dict[str, Any]:
    msgs = state.get("messages", [])

    # Inject dynamic CONTEXT_MENU as a system message for grounding
    candidates = state.get("metadata", {}).get("candidates", [])
    context_msg = MessageModel(role="system", content=format_context_menu(candidates))
    # Avoid duplication if already present this turn
    if not any(m.get("content","").startswith("CONTEXT_MENU") for m in msgs if m.get("role")=="system"):
        msgs.insert(1, context_msg.model_dump())

    last = msgs[-1]["content"] if msgs else ""
    reply = "Sure—how can I help you further?" if last else "Hello! I'm your AI waiter. How can I help you today?"
    msgs.append({"role":"assistant","content": reply})
    state["messages"] = msgs
    return state