# managers/ordering.py
from langgraph.graph import StateGraph, END
from state import ChatStateTD
from nodes.ordering import (
    menu_lookup_node,
    take_order_node,
    confirm_order_node,
    chitchat_node,
)

def build_ordering_manager():
    g = StateGraph(state_schema=ChatStateTD)

    # Internal “switch” node
    def _om_router(state: dict) -> dict:
        return state

    g.add_node("om_router", _om_router)
    g.add_node("menu_lookup", menu_lookup_node)
    g.add_node("take_order", take_order_node)
    g.add_node("confirm_order", confirm_order_node)
    g.add_node("chitchat", chitchat_node)

    g.set_entry_point("om_router")

    def _order_branch(state: dict) -> str:
        md = state.get("metadata", {})
        intent = (md.get("last_intent") or "").lower()
        stage = (md.get("stage") or "").lower()

        # Prefer explicit intents first
        if intent == "ordering.lookup" or stage == "menu":
            return "menu_lookup"
        if intent == "ordering.take" or stage == "take":
            return "take_order"
        if intent.startswith("confirm.") or stage == "confirm":
            return "confirm_order"
        # default small talk under ordering manager
        return "chitchat"

    g.add_conditional_edges(
        "om_router",
        _order_branch,
        {
            "menu_lookup": "menu_lookup",
            "take_order": "take_order",
            "confirm_order": "confirm_order",
            "chitchat": "chitchat",
        },
    )

    # Every worker in a manager ends the subgraph (returns to parent graph)
    for n in ["menu_lookup", "take_order", "confirm_order", "chitchat"]:
        g.add_edge(n, END)

    return g.compile()