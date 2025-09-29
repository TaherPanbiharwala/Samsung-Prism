# graph.py
from langgraph.graph import StateGraph, END
from state import ChatStateTD

from nodes.router import router_node
from nodes.ordering import (
    menu_lookup_node, take_order_node, confirm_order_node, chitchat_node, om_router
)
from nodes.payment import (
    bill_node, split_bill_node, payment_gateway_node, feedback_node, pm_router
)

# ---------- manager subgraphs ----------

def build_ordering_manager():
    g = StateGraph(state_schema=ChatStateTD)

    # workers
    g.add_node("menu_lookup", menu_lookup_node)
    g.add_node("take_order", take_order_node)
    g.add_node("confirm_order", confirm_order_node)
    g.add_node("chitchat", chitchat_node)

    # entry = mini router (function om_router is imported)
    g.add_node("om_router", lambda s: s)
    g.set_entry_point("om_router")

    # route to workers or END
    g.add_conditional_edges(
        "om_router",
        om_router,
        {
            "menu_lookup": "menu_lookup",
            "take_order": "take_order",
            "confirm_order": "confirm_order",
            "chitchat": "chitchat",
            END: END,
        },
    )

    # ⛔ IMPORTANT: workers should END the subgraph (no loop-back)
    g.add_edge("menu_lookup", END)
    g.add_edge("take_order", END)
    g.add_edge("confirm_order", END)
    g.add_edge("chitchat", END)

    return g.compile()


def build_payment_manager():
    g = StateGraph(state_schema=ChatStateTD)

    # workers
    g.add_node("bill", bill_node)
    g.add_node("split_bill", split_bill_node)
    g.add_node("payment_gateway", payment_gateway_node)
    g.add_node("feedback", feedback_node)

    # entry = mini router
    g.add_node("pm_router", lambda s: s)
    g.set_entry_point("pm_router")

    g.add_conditional_edges(
        "pm_router",
        pm_router,
        {
            "bill": "bill",
            "split_bill": "split_bill",
            "payment_gateway": "payment_gateway",
            "feedback": "feedback",
            END: END,
        },
    )

    # ⛔ workers END the subgraph (no loop-back)
    g.add_edge("bill", END)
    g.add_edge("split_bill", END)
    g.add_edge("payment_gateway", END)
    g.add_edge("feedback", END)

    return g.compile()

# ---------- top-level graph ----------

def _route_to_manager(state):
    md = state.get("metadata", {})
    last_intent = (md.get("last_intent") or "").lower()
    route = (md.get("route") or "").lower()

    if last_intent == "app.exit":
        return END
    if last_intent == "chitchat":
        return "ordering_manager"
    if route == "order":
        return "ordering_manager"
    if route == "payment":
        return "payment_manager"
    return "error"

def error_node(state):
    msgs = state.get("messages", [])
    md = state.setdefault("metadata", {})
    msgs.append({"role": "assistant", "content": "Sorry, I hit an error. Let’s start over."})
    md["_awaiting_worker"] = False
    state["messages"] = msgs
    return state

def build_graph():
    ordering_manager = build_ordering_manager()
    payment_manager  = build_payment_manager()

    g = StateGraph(state_schema=ChatStateTD)

    g.add_node("router", router_node)
    g.add_node("ordering_manager", ordering_manager)
    g.add_node("payment_manager", payment_manager)
    g.add_node("error", error_node)

    g.set_entry_point("router")

    g.add_conditional_edges(
        "router",
        _route_to_manager,
        {
            "ordering_manager": "ordering_manager",
            "payment_manager": "payment_manager",
            "error": "error",
            END: END,
        },
    )

    # ⛔ REMOVE these to avoid same-turn loops:
    # g.add_edge("ordering_manager", "router")
    # g.add_edge("payment_manager", "router")

    # error ends the turn
    g.add_edge("error", END)

    return g.compile()