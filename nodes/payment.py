# Bill gen, split, payment

# nodes/payment.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os, time, json, math,re
from langgraph.graph import END

from pydantic import BaseModel, Field
from utils.validation import validate_node
from utils.llm import generate_json

# ---- config for feedback form ----
FEEDBACK_FORM_URL = os.getenv("FEEDBACK_FORM_URL", "").strip() or \
    "https://docs.google.com/forms/d/e/1FAIpQLSc-r4JECGsJiq7YV1mEb8e7PmViOewxo1r7e73n8LDir1TlPg/viewform?usp=header"
# Optional: prefill Google Form entry IDs (change to yours if you have them)
# Example: {"rating": "entry.12345", "comment": "entry.67890", "session": "entry.11111", "total": "entry.22222"}
FEEDBACK_PREFILL = json.loads(os.getenv("FEEDBACK_FORM_PREFILL", "{}") or "{}")

def _cart_total(cart: List[Dict[str, Any]]) -> float:
    return float(sum((float(it.get("price", 0)) * int(it.get("qty", 1))) for it in (cart or [])))

def _summarize_cart(cart: List[Dict[str, Any]]) -> str:
    if not cart:
        return "Your cart is empty."
    parts = [f"{it.get('qty',1)} x {it.get('name','?')} (₹{it.get('price',0)})" for it in cart]
    return "; ".join(parts) + f" — Total: ₹{_cart_total(cart)}"

# --------- Schemas ---------
class PayInput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PayOutput(PayInput):
    pass
# --------- Helpers ---------
def pm_router(state):
    md = state.get("metadata", {}) or {}

    # ✅ stop the subgraph if a worker already replied this turn
    if md.get("_awaiting_worker") is False:
        return END

    intent = (md.get("last_intent") or "").lower()

    if intent == "payment.bill":
        return "bill"
    if intent == "payment.split":
        return "split_bill"
    if intent == "payment.pay":
        return "payment_gateway"
    if intent == "payment.feedback":
        return "feedback"

    # not a payment intent → leave the subgraph
    return END

def _pick_order_source(md: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find items to bill: prefer last placed order, fallback to current cart."""
    if md.get("last_order", {}).get("items"):
        return md["last_order"]["items"]
    return md.get("cart", [])


def _calc_total(items: List[Dict[str, Any]], default_price: int = 100) -> int:
    total = 0
    for it in items or []:
        price = int(it.get("price", default_price))
        qty   = int(it.get("qty", 1))
        total += price * qty
    return total


@validate_node(name="BillGeneration", tags=["payment","bill"], input_model=PayInput, output_model=PayOutput)
def bill_node(state: Dict[str, Any]) -> Dict[str, Any]:
    md = state.setdefault("metadata", {})
    msgs = state.setdefault("messages", [])

    # Prefer the last placed order if available, else current cart
    order = md.get("last_order") or {}
    cart  = (order.get("items") if order else None) or md.get("cart", [])
    total = _cart_total(cart)

    if not cart:
        msgs.append({"role":"assistant","content":"I don’t see any items yet. Add some dishes and I’ll prepare your bill."})
        md["_awaiting_worker"] = False
        return state

    lines = ["🧾 **Bill Summary**", _summarize_cart(cart)]
    if order:
        lines.append(f"Order ID: …{str(order.get('order_id',''))[-4:]}")
        lines.append(f"Status: {order.get('status','PLACED')}")
    lines.append("\nWould you like to **split the bill** or **pay now**?")
    msgs.append({"role":"assistant","content":"\n".join(lines)})

    md["route"] = "payment"
    md["stage"] = "bill"
    md["_awaiting_worker"] = False
    return state


# --------- Split Bill ---------
@validate_node(name="SplitBill", tags=["payment","split"], input_model=PayInput, output_model=PayOutput)
def split_bill_node(state: Dict[str, Any]) -> Dict[str, Any]:
    md = state.setdefault("metadata", {})
    msgs = state.setdefault("messages", [])
    user = (msgs[-1]["content"] if msgs else "").strip()

    order = md.get("last_order") or {}
    cart  = (order.get("items") if order else None) or md.get("cart", [])
    if not cart:
        msgs.append({"role":"assistant","content":"There’s no bill to split yet. Add items first."})
        md["_awaiting_worker"] = False
        return state

    m = re.search(r"\b(\d+)\b", user)
    if not m:
        msgs.append({"role":"assistant","content":"How many people should I split the bill between?"})
        md["route"] = "payment"
        md["stage"] = "split_pending"
        md["_awaiting_worker"] = False
        return state

    n = max(1, int(m.group(1)))
    total = _cart_total(cart)
    share = round(total / n, 2)
    msgs.append({"role":"assistant","content":f"Split between **{n}** people → each pays **₹{share}** (total ₹{total}).\nReady to **pay now**?"})

    md["route"] = "payment"
    md["stage"] = "split"
    md["_awaiting_worker"] = False
    return state


# --------- Payment Gateway (stub) ---------
@validate_node(name="PaymentGateway", tags=["payment","pay"], input_model=PayInput, output_model=PayOutput)
def payment_gateway_node(state: Dict[str, Any]) -> Dict[str, Any]:
    md = state.setdefault("metadata", {})
    msgs = state.setdefault("messages", [])

    order = md.get("last_order") or {}
    cart  = (order.get("items") if order else None) or md.get("cart", [])
    if not cart:
        msgs.append({"role":"assistant","content":"There’s nothing to pay for yet. Your cart is empty."})
        md["_awaiting_worker"] = False
        return state

    total = _cart_total(cart)
    payment = {
        "ts": int(time.time()),
        "amount": total,
        "status": "SUCCESS",
        "method": "DUMMY_GATEWAY"
    }
    md.setdefault("payments", []).append(payment)

    # ✅ Success message + inline link
    msg = f"✅ **Payment successful** for **₹{total}**.\n" \
          f"Would you like to leave feedback? You can fill the form here:\n{FEEDBACK_FORM_URL}"
    msgs.append({"role":"assistant","content": msg})

    # 🔧 “Tool call” hint for UI clients (optional)
    md.setdefault("tool_calls", []).append({
        "tool": "open_url",
        "url": FEEDBACK_FORM_URL,
        "label": "Open feedback form"
    })

    # Keep last_order; clear cart if you want
    md["cart"] = []
    md["route"] = "payment"
    md["stage"] = "paid"
    # If you want to auto-route to Feedback node on the next tick, also set:
    # md["last_intent"] = "payment.feedback"

    md["_awaiting_worker"] = False
    return state


# --------- Feedback (Google Forms) ---------
@validate_node(name="Feedback", tags=["payment","feedback"], input_model=PayInput, output_model=PayOutput)
def feedback_node(state: Dict[str, Any]) -> Dict[str, Any]:
    md = state.setdefault("metadata", {})
    msgs = state.setdefault("messages", [])

    form_url = os.getenv("FEEDBACK_FORM_URL", "").strip()
    if form_url:
        msg = (
            "Thanks for dining with us! 🙏\n"
            f"Please share your feedback here:\n{form_url}\n\n"
            "Reply “done” after submitting, or tell me if you need anything else."
        )
    else:
        msg = (
            "Thanks for dining with us! 🙏\n"
            "Please rate your experience (1–5) and optionally add a short comment."
        )

    # Keep the convo alive, mark that we are waiting for feedback
    md["route"] = None                # <- return to top-level router naturally
    md["stage"] = "feedback.wait"
    md["awaiting_feedback"] = True
    md["_awaiting_worker"] = False
    msgs.append({"role":"assistant","content": msg})
    return state