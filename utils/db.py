# utils/db.py
import json
from typing import Any, Optional, Union, Dict, cast
import redis
from state import ChatStateModel, ChatStateTD, from_graph_state

# Type it as plain Redis (not generic)
r: redis.Redis = redis.Redis(host="localhost", port=6379, db=0)

def save_state(session_id: str, state_dict: Dict[str, Any]) -> None:
    r.set(session_id, json.dumps(state_dict))

def load_state(session_id: str) -> Optional[ChatStateModel]:
    raw: Optional[Union[bytes, bytearray, memoryview, str]] = r.get(session_id)  # Redis returns bytes-like
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray, memoryview)):
        s = bytes(raw).decode("utf-8")
    else:
        s = str(raw)
    as_dict: Dict[str, Any] = json.loads(s)
    # Cast for the type checker; runtime validation still happens in from_graph_state()
    return from_graph_state(cast(ChatStateTD, as_dict))