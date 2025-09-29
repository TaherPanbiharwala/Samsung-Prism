# main.py
import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import cast, Optional
from dotenv import load_dotenv
from langsmith import traceable

from utils.nlu import classify
from utils.config import SYSTEM_PROMPT
from graph import build_graph
from state import (
    ChatStateModel,
    MessageModel,
    ChatStateTD,
    to_graph_state,
    from_graph_state,
)
from utils import db

# ---- startup / env -----------------------------------------
load_dotenv()  # ensure .env is loaded for tracing, chroma, etc.
print("[CFG] USE_WAITER_LLM =", os.getenv("USE_WAITER_LLM"))
print("[CFG] WAITER_LLM_BACKEND =", os.getenv("WAITER_LLM_BACKEND"))

def ensure_system_prompt(model: ChatStateModel) -> None:
    """Insert system prompt once per session if missing (covers old stored sessions too)."""
    if not any(m.role == "system" for m in model.messages):
        model.messages.insert(0, MessageModel(role="system", content=SYSTEM_PROMPT))

# ---- app / graph -------------------------------------------
app = FastAPI()
graph_app = build_graph()

class ChatRequest(BaseModel):
    session_id: str
    user_message: str

@traceable(name="POST /chat", tags=["api", "phase1"])
def _handle_turn(model: ChatStateModel, user_text: str) -> ChatStateModel:
    # normalize & ignore empty user_text
    user_text = (user_text or "").strip()
    if not user_text:
        # no change to model; caller can decide how to respond
        return model

    # make sure system prompt exists
    ensure_system_prompt(model)

    # (optional) quick NLU call for trace visibility
    stage = model.metadata.get("stage")
    _ = classify(user_text, stage=stage, history=[m.model_dump() for m in model.messages][-4:])

    # record user msg
    model.messages.append(MessageModel(role="user", content=user_text))

    # start-of-turn guard for router/graph
    model.metadata["_awaiting_worker"] = True

    # run graph
    out_dict = graph_app.invoke(to_graph_state(model))
    model = from_graph_state(cast(ChatStateTD, out_dict))
    return model

# ---- HTTP endpoint -----------------------------------------
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    model = db.load_state(req.session_id) or ChatStateModel(session_id=req.session_id)
    ensure_system_prompt(model)

    # Handle turn (will ignore blank user_message safely)
    model = _handle_turn(model, req.user_message)

    # If blank input, reply with a gentle nudge
    if not (req.user_message or "").strip():
        # only add a reply if the graph didn’t already respond
        if not model.messages or model.messages[-1].role != "assistant":
            model.messages.append(
                MessageModel(role="assistant", content="(please type a message)")
            )

    db.save_state(req.session_id, model.model_dump())
    return {"response": model.messages[-1].content if model.messages else ""}

# ---- CLI helpers -------------------------------------------
# ---- CLI helpers -------------------------------------------
def _read_user() -> Optional[str]:
    try:
        s = input("You: ")
    except EOFError:
        return None
    s = (s or "").strip()
    if not s:
        print("(ignored empty input)")
        return None
    return s

@traceable(name="terminal_turn", tags=["cli", "phase1"])
def _cli_turn(m: ChatStateModel, text: str) -> ChatStateModel:
    return _handle_turn(m, text)

# ---- entrypoint --------------------------------------------
if __name__ == "__main__":
    # Always start clean for CLI/dev runs
    model = ChatStateModel(session_id="cli", messages=[], metadata={})
    model.metadata["cart"] = []
    model.metadata["candidates"] = []
    model.metadata["confirmed"] = False
    model.metadata.pop("confirm_signal", None)
    ensure_system_prompt(model)

    while True:
        text = _read_user()
        if text is None:
            continue
        model = _cli_turn(model, text)
        if model.messages and model.messages[-1].role == "assistant":
            print("AI:", model.messages[-1].content)
    # To run API instead:
    # uvicorn.run(app, host="0.0.0.0", port=8000)