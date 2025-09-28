# graph.py
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from state import ChatStateTD
from nodes.router import router_node
from nodes.ordering import menu_lookup_node, take_order_node, confirm_order_node, pos_submit_node
from utils.validation import validate_node

@validate_node(name="ErrorNode")
def error_node(state: Dict[str, Any]) -> Dict[str, Any]:
    err = state.get("_error", {})
    node = err.get("node", "unknown")
    where = err.get("where", "unknown")
    msgs = state.get("messages", [])
    msgs.append({"role": "assistant", "content": f"Sorry, something went wrong in {node} ({where}). A human will assist you shortly."})
    state["messages"] = msgs
    state.pop("_error", None)
    return state

def build_graph():
    g = StateGraph(ChatStateTD)

    g.add_node("router", router_node)
    g.add_node("menu_lookup", menu_lookup_node)
    g.add_node("take_order", take_order_node)
    g.add_node("confirm_order", confirm_order_node)
    g.add_node("pos_submit", pos_submit_node)
    g.add_node("error", error_node)

    g.set_entry_point("router")

    def after_router(state: Dict[str, Any]) -> str:
        if "_error" in state:
            return "error"
        md = state.get("metadata", {})
        if md.get("route") != "order":
            return "__end__"
        stage = md.get("stage") or "menu"
        return {
            "menu": "menu_lookup",
            "take": "take_order",
            "confirm": "confirm_order",
            "submit": "pos_submit",
        }.get(stage, "menu_lookup")

    g.add_conditional_edges(
        "router",
        after_router,
        {
            "menu_lookup": "menu_lookup",
            "take_order": "take_order",
            "confirm_order": "confirm_order",
            "pos_submit": "pos_submit",
            "__end__": END,
            "error": "error",
        },
    )

    # Each ordering node runs ONCE per turn, then stop.
    g.add_edge("menu_lookup", END)
    g.add_edge("take_order", END)
    g.add_edge("confirm_order", END)
    g.add_edge("pos_submit", END)
    g.add_edge("error", END)

    return g.compile()