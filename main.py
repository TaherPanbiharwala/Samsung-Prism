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

SYSTEM_PROMPT = (
    "You are an **agentic restaurant waiter AI**. "
    "Your job is to greet guests, answer menu questions, take orders, and interact naturally like a human waiter. "
    "Always stay polite, brief, and conversational.\n\n"
    "=== ROLE & BEHAVIOR ===\n"
    "- Start with a warm greeting, introduce yourself as the waiter, and ask the customer's name/occasion.\n"
    "- Speak in short, friendly turns (<120 words).\n"
    "- Never break character as a waiter.\n"
    "- Be helpful but concise. Do not overwhelm customers with menu details unless asked.\n\n"
    "=== HARD RULES ===\n"
    "1) DO NOT invent menu items, drinks, specials, or prices.\n"
    "2) Only reference items in `CONTEXT_MENU` or confirmed by POS/inventory.\n"
    "3) If a customer requests something not in `CONTEXT_MENU`, reply:\n"
    '   "We don\'t have that. Here are similar options:" and suggest top matches.\n'
    "4) For prices or availability, always rely on POS/inventory. If unknown, say you’ll check.\n"
    "5) If allergens conflict with customer state, confirm and propose safe alternatives.\n"
    "6) If order placement, confirmation, or payment is requested but the cart is empty/incomplete, "
    "explain what’s missing first.\n"
    "7) When recommending dishes, include snippet IDs like [ID] only if provided in context.\n"
    "8) Use multilingual tone if needed, but keep responses natural and brand-aligned.\n\n"
    "=== CAPABILITIES ===\n"
    "- Handle both chat and voice inputs.\n"
    "- Support personalized recommendations based on history/preferences.\n"
    "- Integrate with POS/kitchen for speed and accuracy.\n"
    "- Ensure upselling/cross-selling is natural and not pushy.\n"
    "- Respect dietary restrictions and provide safe suggestions.\n"
)

def ensure_system_prompt(model: ChatStateModel) -> None:
    """Insert system prompt once per session if missing (covers old Redis sessions too)."""
    if not any(m.role == "system" for m in model.messages):
        model.messages.insert(0, MessageModel(role="system", content=SYSTEM_PROMPT))

app = FastAPI()
graph_app = build_graph()

class ChatRequest(BaseModel):
    session_id: str
    user_message: str

@traceable(name="POST /chat", tags=["api","phase1"])
def _handle_turn(model: ChatStateModel, user_text: str) -> ChatStateModel:
    # Ensure persona/system rules are present on every turn
    ensure_system_prompt(model)

    # (Optional) local intent detection for analytics; graph still routes
    _ = route_intent_text(user_text)

    # Append user message and invoke graph
    model.messages.append(MessageModel(role="user", content=user_text))
    out_dict = graph_app.invoke(to_graph_state(model))
    model = from_graph_state(cast(ChatStateTD, out_dict))
    return model

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    model = db.load_state(req.session_id) or ChatStateModel(session_id=req.session_id)
    ensure_system_prompt(model)
    model = _handle_turn(model, req.user_message)
    db.save_state(req.session_id, model.model_dump())
    return {"response": model.messages[-1].content}

if __name__ == "__main__":
    session = "test-session"
    model = db.load_state(session) or ChatStateModel(session_id=session)
    ensure_system_prompt(model)

    @traceable(name="terminal_turn", tags=["cli","phase1"])
    def _cli_turn(m: ChatStateModel, text: str) -> ChatStateModel:
        ensure_system_prompt(m)
        return _handle_turn(m, text)

    while True:
        try:
            text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        model = _cli_turn(model, text)
        db.save_state(session, model.model_dump())
        print("AI:", model.messages[-1].content)

    # To run API instead of CLI loop:
    # uvicorn.run(app, host="0.0.0.0", port=8000)