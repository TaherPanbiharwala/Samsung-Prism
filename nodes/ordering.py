# Menu lookup, recommendations, confirm

# nodes/ordering.py
from typing import List, Dict, Any
import re
from pydantic import BaseModel, Field
from utils.validation import validate_node
from utils import rag
from utils.context import format_context_menu
from state import MessageModel

# ---------- Schemas ----------
class OrderInput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]] = Field(default_factory=list, min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class OrderOutput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]] = Field(default_factory=list, min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ---------- Helpers ----------
def _ensure_lists(state: Dict[str, Any]) -> None:
    md = state.setdefault("metadata", {})
    md.setdefault("cart", [])
    md.setdefault("candidates", [])
    md.setdefault("confirmed", False)

def _fmt_item(it: Dict) -> str:
    return f'{it["name"]} (₹{it["price"]})'

def _cart_total(cart: List[Dict]) -> int:
    return sum((it["price"] * it.get("qty", 1)) for it in cart)

def _summarize_cart(cart: List[Dict]) -> str:
    if not cart:
        return "Your cart is empty."
    parts = [f'{c["qty"]} x {c["name"]} (₹{c["price"]})' for c in cart]
    return "; ".join(parts) + f" — Total: ₹{_cart_total(cart)}"

# ---------- Nodes ----------
# nodes/ordering.py  (replace menu_lookup_node)
from utils.context import format_context_menu
from state import MessageModel

@validate_node(name="MenuLookup", tags=["ordering","menu"], input_model=OrderInput, output_model=OrderOutput)
def menu_lookup_node(state: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_lists(state)
    msgs = state["messages"]
    last = msgs[-1]["content"].lower()

    # infer simple faceted filters from user text
    cat = None
    if "starter" in last: cat = "starter"
    elif "main" in last or "main course" in last: cat = "main"
    elif "dessert" in last or "sweet" in last: cat = "dessert"
    elif "drink" in last or "beverage" in last: cat = "beverage"

    want_veg = "veg" in last and "non veg" not in last and "non-veg" not in last
    want_nonveg = "non veg" in last or "non-veg" in last or "chicken" in last or "mutton" in last or "fish" in last or "egg" in last

    # query chroma then post-filter by metadata
    results = rag.search_menu(last, top_k=12)

    def _is_cat_ok(x: Dict[str, Any]) -> bool:
        if not cat: return True
        return cat.lower() in str(x.get("category","")).lower()

    def _is_veg_ok(x: Dict[str, Any]) -> bool:
        tags = str(x.get("tags","")).lower()
        if want_veg: return "veg" in tags or "vegetarian" in tags
        if want_nonveg: return "non" in tags or "chicken" in tags or "mutton" in tags or "egg" in tags or "fish" in tags
        return True

    filtered = [r for r in results if _is_cat_ok(r) and _is_veg_ok(r)]
    final = filtered[:5] if filtered else results[:5]

    state["metadata"]["candidates"] = final

    # 🔹 NEW: Inject dynamic CONTEXT_MENU system message
    context_msg = MessageModel(role="system", content=format_context_menu(final))
    if not any(m.get("content","").startswith("CONTEXT_MENU") for m in msgs if m.get("role")=="system"):
        msgs.insert(1, context_msg.model_dump())

    if final:
        listing = "\n".join([f"{i+1}. {_fmt_item(r)}" for i, r in enumerate(final)])
        reply = (
            f"Here are some {('veg ' if want_veg else '')}{(cat or 'popular')} options:\n{listing}\n\n"
            "Tell me the numbers (e.g., '1 and 3') or item names with quantities (e.g., '2 garlic bread')."
        )
    else:
        reply = "I couldn't find a good match. Try 'veg starters', 'non-veg mains', or name a dish."

    msgs.append({"role": "assistant", "content": reply})
    state["messages"] = msgs
    state["metadata"]["stage"] = "take"
    return state

@validate_node(name="Recommend", tags=["ordering","recommend"], input_model=OrderInput, output_model=OrderOutput)
def recommend_node(state: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_lists(state)
    cand = state["metadata"].get("candidates", [])
    tips: List[str] = []
    names = [c["name"].lower() for c in cand]
    if any("pizza" in n for n in names):
        tips.append("Garlic Bread (₹149)")
        tips.append("Lemon Iced Tea (₹99)")
    if any("soup" in n for n in names):
        tips.append("Garlic Bread (₹149)")
    reply = "Recommendation: " + ", ".join(tips) if tips else "No special pairings to recommend yet."
    state["messages"].append({"role":"assistant","content": reply})
    return state

# nodes/ordering.py  (replace take_order_node)
ORDER_VERBS = (
    "add", "get me", "give me", "i'll have", "i will have",
    "i want", "we'll have", "we will have", "order", "take"
)

@validate_node(name="TakeOrder", tags=["ordering","take"], input_model=OrderInput, output_model=OrderOutput)
def take_order_node(state: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_lists(state)
    user_raw = state["messages"][-1]["content"]
    user = user_raw.lower().strip()

    # if it's informational, don't add
    if any(kw in user for kw in ["tell me", "show me", "what are", "what's", "suggest", "recommend"]):
        state["messages"].append({"role": "assistant", "content":
            "Happy to help—pick the numbers you like (e.g., '1 and 3') or say 'add 2 garlic bread'."})
        state["metadata"]["stage"] = "take"
        return state

    cands: List[Dict[str, Any]] = state["metadata"].get("candidates", [])
    added: List[str] = []

    # Split into chunks by “and” or commas so we can mix numbers and names
    chunks = re.split(r",| and ", user_raw, flags=re.IGNORECASE)

    for ch in chunks:
        ch_clean = ch.strip()
        if not ch_clean:
            continue

        # 1) number reference to candidate list
        if ch_clean.isdigit() and cands:
            idx = int(ch_clean) - 1
            if 0 <= idx < len(cands):
                it = cands[idx]
                item = {"id": it["id"], "name": it["name"], "price": it["price"], "qty": 1}
                state["metadata"]["cart"].append(item)
                added.append(it["name"])
            continue

        # 2) explicit qty + name OR order verbs
        m = re.match(r".*?(\d+)\s*x?\s*([A-Za-z].+)", ch_clean, flags=re.IGNORECASE)
        if m:
            qty = int(m.group(1))
            name = m.group(2).strip()
        else:
            qty = 1
            # strip leading verbs like "add " or "i'll have "
            cleaned = re.sub(
                r"^(add|get me|give me|i'?ll have|i will have|i want|we'?ll have|we will have|order|take)\s+",
                "",
                ch_clean,
                flags=re.IGNORECASE,
            )
            name = cleaned

        it = rag.find_by_name(name)
        if it:
            item = {"id": it["id"], "name": it["name"], "price": it["price"], "qty": qty}
            state["metadata"]["cart"].append(item)
            added.append(it["name"])
            continue

        # 3) bare fallback
        if not m and not any(v in ch_clean.lower() for v in ORDER_VERBS):
            it = rag.find_by_name(ch_clean)
            if it:
                item = {"id": it["id"], "name": it["name"], "price": it["price"], "qty": 1}
                state["metadata"]["cart"].append(item)
                added.append(it["name"])

    # Reply
    if added:
        reply = (
            f"Added: {', '.join(added)}.\n"
            f"{_summarize_cart(state['metadata']['cart'])}\n"
            "Shall I place the order? (yes/no)"
        )
        state["metadata"]["stage"] = "confirm"
    else:
        reply = (
            "I didn't catch any items to add. Pick by number (e.g., '1 and 3') "
            "or say 'add 2 garlic bread'."
        )
        state["metadata"]["stage"] = "take"

    state["messages"].append({"role": "assistant", "content": reply})
    return state

@validate_node(name="ConfirmOrder", tags=["ordering","confirm"], input_model=OrderInput, output_model=OrderOutput)
def confirm_order_node(state: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_lists(state)
    user = state["messages"][-1]["content"].strip().lower()
    cart = state["metadata"].get("cart", [])

    if user in {"yes","y","confirm","place","ok"}:
        if not cart:  # 🔹 guard against empty cart
            state["metadata"]["stage"] = "confirm"
            state["messages"].append({
                "role": "assistant",
                "content": "Your cart is empty — please add some items before confirming."
            })
            return state
        state["metadata"]["confirmed"] = True
        state["metadata"]["stage"] = "submit"
        state["messages"].append({"role":"assistant","content": "Great! Placing your order now..."})
    elif user in {"no","n","change","edit"}:
        state["metadata"]["confirmed"] = False
        state["metadata"]["stage"] = "take"
        state["messages"].append({"role":"assistant","content": "No worries. Tell me what to add/remove or new quantities."})
    else:
        state["metadata"]["stage"] = "confirm"
        state["messages"].append({"role":"assistant","content": "Please reply with 'yes' to place the order, or 'no' to modify."})
    return state

@validate_node(name="POS_Submit", tags=["ordering","pos","leaf"], input_model=OrderInput, output_model=OrderOutput)
def pos_submit_node(state: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_lists(state)
    if not state["metadata"].get("confirmed"):
        return state

    import json, time, os
    order = {
        "session_id": state["session_id"],
        "ts": int(time.time()),
        "items": state["metadata"]["cart"],
        "total": sum(it["price"] * it.get("qty", 1) for it in state["metadata"]["cart"]),
        "status": "PLACED",
    }
    path = "orders.json"
    existing: List[Dict] = []
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                existing = json.load(f)
            except Exception:
                existing = []
    existing.append(order)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)

    # Let the user know
    state["messages"].append({
        "role": "assistant",
        "content": f"✅ Order placed! Total: ₹{order['total']}. Your order id ends with ...{order['ts'] % 10000}."
    })

    # reset the flow for a fresh next order
    state["metadata"]["route"] = None
    state["metadata"]["stage"] = None
    state["metadata"]["cart"] = []
    state["metadata"]["confirmed"] = False
    return state