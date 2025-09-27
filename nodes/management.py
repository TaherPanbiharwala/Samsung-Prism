# nodes/management.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from utils.validation import validate_node
from langsmith import traceable

class ChitchatInput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]] = Field(default_factory=list, min_length=1)

class ChitchatOutput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]] = Field(default_factory=list, min_length=1)

@validate_node(
    name="Chitchat",
    tags=["management","chitchat"],
    input_model=ChitchatInput,
    output_model=ChitchatOutput,
)
def chitchat_node(state: Dict[str, Any]) -> Dict[str, Any]:
    msgs = state.get("messages", [])
    last = msgs[-1]["content"] if msgs else ""
    reply = "Hello! I'm your AI waiter. How can I help you today?" if not last else "Sure—how can I help you further?"
    msgs.append({"role":"assistant","content": reply})
    state["messages"] = msgs
    return state