# main.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import cast
from dotenv import load_dotenv
from langsmith import traceable
from nodes.router import route_intent_text

from graph import build_graph
from state import ChatStateModel, MessageModel, ChatStateTD, to_graph_state, from_graph_state
from utils import db

load_dotenv()  # make sure .env is loaded for tracing
app = FastAPI()
graph_app = build_graph()

class ChatRequest(BaseModel):
    session_id: str
    user_message: str

@traceable(name="POST /chat")
def _handle_turn(model: ChatStateModel, user_text: str) -> ChatStateModel:
    # detect intent early
    intent = route_intent_text(user_text)
    # append validated message
    model.messages.append(MessageModel(role="user", content=user_text))
    # run graph
    out_dict = graph_app.invoke(to_graph_state(model))
    model = from_graph_state(cast(ChatStateTD, out_dict))

    # attach dynamic tags
    traceable.set_run_tags([f"intent:{intent}", "api","phase1"])
    return model

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    model = db.load_state(req.session_id) or ChatStateModel(session_id=req.session_id)
    model = _handle_turn(model, req.user_message)
    db.save_state(req.session_id, model.model_dump())
    return {"response": model.messages[-1].content}

if __name__ == "__main__":
    session = "test-session"
    model = db.load_state(session) or ChatStateModel(session_id=session)

    @traceable(name="terminal_turn")
    def _cli_turn(m: ChatStateModel, text: str) -> ChatStateModel:
        intent = route_intent_text(text)
        out = _handle_turn(m, text)
        traceable.set_run_tags([f"intent:{intent}", "cli","phase1"])
        return out

    while True:
        try:
            text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit","exit"}:
            break
        model = _cli_turn(model, text)
        db.save_state(session, model.model_dump())
        print("AI:", model.messages[-1].content)

    # To run API instead, comment the loop and:
    # uvicorn.run(app, host="0.0.0.0", port=8000)