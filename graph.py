# graph.py
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from state import ChatStateTD
from nodes.router import router_node
from nodes.ordering import menu_lookup_node, take_order_node, confirm_order_node
from nodes.management import chitchat_node
from utils.validation import validate_node

ERROR_KEY = "_error"

@validate_node(name="ErrorNode", tags=["error"])
def error_node(state: Dict[str, Any]) -> Dict[str, Any]:
    err = state.get(ERROR_KEY, {})
    node = err.get("node", "unknown")
    where = err.get("where", "unknown")
    msgs = state.get("messages", [])
    msgs.append({
        "role": "assistant",
        "content": f"Sorry, something went wrong in {node} ({where}). A human will assist you shortly."
    })
    state["messages"] = msgs
    state.pop(ERROR_KEY, None)
    return state

# graph.py
def _route_from_router(state: Dict[str, Any]) -> str:
    md    = state.get("metadata", {})
    intent = (md.get("last_intent") or "").lower()
    stage  = (md.get("stage") or "").lower()

    # ✅ If a worker set an error, go to the error node immediately
    if state.get(ERROR_KEY):
        print("GRAPH ROUTE:", {"intent": intent, "stage": stage, "next": "error"})
        return "error"

    # ✅ End only AFTER a worker set the flag to False this invoke
    if md.get("_awaiting_worker") is False:
        md.pop("_awaiting_worker", None)  # reset for next turn
        print("GRAPH ROUTE:", {"intent": intent, "stage": stage, "next": "__end__"})
        return "__end__"

    # otherwise choose a worker

    elif intent in {"ordering.lookup", "ordering.more", "ordering.lookup_refine"}:
        nxt = "menu_lookup"
    elif intent == "ordering.take":
        nxt = "take_order"
    elif intent.startswith("confirm."):
        nxt = "confirm_order"
    else:
        nxt = "chitchat"

    print("GRAPH ROUTE:", {"intent": intent, "stage": stage, "next": nxt})
    return nxt

def build_graph():
    g = StateGraph(ChatStateTD)

    # nodes
    g.add_node("router",        router_node)
    g.add_node("menu_lookup",   menu_lookup_node)
    g.add_node("take_order",    take_order_node)
    g.add_node("confirm_order", confirm_order_node)
    g.add_node("chitchat",      chitchat_node)
    g.add_node("error",         error_node)

    # entry
    g.set_entry_point("router")

    # graph.py — keep exactly one of these blocks in the file
    g.add_conditional_edges(
        "router",
        _route_from_router,
        {
            "menu_lookup":   "menu_lookup",
            "take_order":    "take_order",
            "confirm_order": "confirm_order",
            "chitchat":      "chitchat",
            "error":         "error",   # ← add this
            "__end__":       END,
        },
    )


    # each worker -> router (router will then immediately return __end__)
    for node in ["menu_lookup","take_order","confirm_order","chitchat"]:
        g.add_edge(node, "router")

    # error -> END
    g.add_edge("error", END)

    return g.compile()