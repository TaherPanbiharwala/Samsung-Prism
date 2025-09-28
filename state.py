# state.py
from typing import List, Dict, Any, Literal, TypedDict, cast
from pydantic import BaseModel, Field
from pydantic.type_adapter import TypeAdapter

class MessageModel(BaseModel):
    # 🔹 Added "system" so persona/context messages are valid
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)

class ChatStateModel(BaseModel):
    session_id: str
    messages: List[MessageModel] = []
    metadata: Dict[str, Any] = {}

class MessageTD(TypedDict):
    # 🔹 Allow "system" in TypedDict as well
    role: Literal["system", "user", "assistant"]
    content: str

class ChatStateTD(TypedDict, total=False):
    session_id: str
    messages: List[MessageTD]
    metadata: Dict[str, Any]
    _error: Dict[str, Any]

state_adapter = TypeAdapter(ChatStateModel)

def to_graph_state(model: ChatStateModel) -> ChatStateTD:
    return cast(ChatStateTD, model.model_dump())

def from_graph_state(state: ChatStateTD) -> ChatStateModel:
    return state_adapter.validate_python(state)