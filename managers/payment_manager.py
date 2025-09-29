# managers/payment.py
from langgraph.graph import StateGraph, END
from state import ChatStateTD
from nodes.payment import bill_node, split_bill_node, payment_gateway_node, feedback_node

def build_payment_manager():
    g = StateGraph(state_schema=ChatStateTD)

    def _pm_router(state: dict) -> dict:
        return state

    g.add_node("pm_router", _pm_router)
    g.add_node("bill", bill_node)
    g.add_node("split_bill", split_bill_node)
    g.add_node("payment_gateway", payment_gateway_node)
    g.add_node("feedback", feedback_node)

    g.set_entry_point("pm_router")

    def _payment_branch(state: dict) -> str:
        md = state.get("metadata", {})
        intent = (md.get("last_intent") or "").lower()
        # map your router intents to workers
        if intent == "payment.bill":
            return "bill"
        if intent == "payment.split":
            return "split_bill"
        if intent == "payment.pay":
            return "payment_gateway"
        if intent == "payment.feedback":
            return "feedback"
        # fallback: show bill
        return "bill"

    g.add_conditional_edges(
        "pm_router",
        _payment_branch,
        {
            "bill": "bill",
            "split_bill": "split_bill",
            "payment_gateway": "payment_gateway",
            "feedback": "feedback",
        },
    )

    for n in ["bill", "split_bill", "payment_gateway", "feedback"]:
        g.add_edge(n, END)

    return g.compile()