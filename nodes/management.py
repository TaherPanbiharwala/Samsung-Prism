# nodes/management.py
from utils.validation import validate_node
from utils.llm import chat as llm_chat
from utils.config import USE_WAITER_LLM
import traceback

@validate_node(name="Chitchat", tags=["chitchat"])
def chitchat_node(state):
    print(f"[NODE ENTER] menu_lookup: stage={state.get('metadata',{}).get('stage')}, "
      f"intent={state.get('metadata',{}).get('last_intent')}, "
      f"_awaiting={state.get('metadata',{}).get('_awaiting_worker')}")
    msgs = state.get("messages", [])
    md = state.setdefault("metadata", {})
    user = msgs[-1]["content"] if msgs else ""

    reply = None
    if USE_WAITER_LLM:
        try:
            reply = llm_chat(
                system="You are a friendly restaurant waiter. Keep replies brief.",
                context_menu="",
                user=user,
                max_tokens=60
            )
        except Exception:
            traceback.print_exc()
            reply = None

    if not reply or not isinstance(reply, str):
        reply = "Hi there! I’m your AI waiter. Would you like starters, mains, beverages or desserts?"

    msgs.append({"role": "assistant", "content": reply})
    md["_awaiting_worker"] = False
    state["messages"] = msgs
    print(f"[NODE EXIT]  menu_lookup: stage={state.get('metadata',{}).get('stage')}, "
      f"cands={len(state.get('metadata',{}).get('candidates', []))}, "
      f"_awaiting={state.get('metadata',{}).get('_awaiting_worker')}")
    if not state.get("messages") or state["messages"][-1]["role"] != "assistant":
        raise RuntimeError("Worker returned without replying")
    if state["metadata"].get("_awaiting_worker") is not False:
        raise RuntimeError("Worker did not set _awaiting_worker=False")
    return state