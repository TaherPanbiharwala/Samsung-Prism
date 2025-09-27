# graph.py
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from state import ChatStateTD
from nodes.router import router_node
from utils.validation import validate_node
from langsmith import traceable

@traceable(name="ErrorNode_wrapper", tags=["node"])
@validate_node(name="ErrorNode")
def error_node(state: Dict[str, Any]) -> Dict[str, Any]:
    err = state.get("_error", {})
    node = err.get("node", "unknown")
    where = err.get("where", "unknown")
    msgs = state.get("messages", [])
    msgs.append({
        "role": "assistant",
        "content": f"Sorry, something went wrong in {node} ({where}). A human will assist you shortly."
    })
    state["messages"] = msgs
    state.pop("_error", None)
    return state

def build_graph():
    g = StateGraph(ChatStateTD)

    g.add_node("router", router_node)
    g.add_node("error", error_node)

    g.set_entry_point("router")

    # Conditional branch: if any node put _error in state → go to error node, else finish
    def _route_after_router(state: Dict[str, Any]) -> str:
        return "error" if "_error" in state else "__end__"

    g.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "error": "error",
            "__end__": END,
        },
    )
    # error -> END
    g.add_edge("error", END)

    return g.compile()