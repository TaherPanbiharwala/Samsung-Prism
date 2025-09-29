# ordering.py
from typing import List, Dict, Any, Optional
import os, re, json, time, traceback

from pydantic import BaseModel, Field

from utils.validation import validate_node
from utils import rag
from utils.context import format_context_menu
from state import MessageModel
from utils.nlu import classify  # (kept for traces if you want)
from utils.llm import generate_waiter , generate_json
from utils.config import SYSTEM_PROMPT, USE_WAITER_LLM
from utils.pos import send_order_to_kitchen


# ---------- Schemas ----------
class OrderInput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class OrderOutput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class OrderItem(BaseModel):
    name: str
    qty: int

class ConfirmSchema(BaseModel):
    decision: str  # "yes", "no", or "unclear"

# ---------- Helpers ----------
def _ensure_lists(state: Dict[str, Any]):
    md = state.setdefault("metadata", {})
    md.setdefault("cart", [])
    md.setdefault("candidates", [])
    md.setdefault("confirmed", False)

def _facet_from_text(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    cat = None
    if "starter" in t: cat = "starter"
    elif "main course" in t or "main" in t: cat = "main"
    elif "dessert" in t or "sweet" in t: cat = "dessert"
    elif "drink" in t or "beverage" in t: cat = "beverage"
    return {
        "veg": ("veg" in t and "non veg" not in t and "non-veg" not in t),
        "nonveg": ("non veg" in t or "non-veg" in t),
        "category": cat,
    }

def _cart_total(cart: List[Dict]) -> int:
    return sum((it["price"] * it.get("qty", 1)) for it in cart)

def _summarize_cart(cart: List[Dict]) -> str:
    if not cart:
        return "Your cart is empty."
    return "; ".join([f"{c['qty']} x {c['name']} (₹{c['price']})" for c in cart]) + f" — Total: ₹{_cart_total(cart)}"

def _best_match_from_candidates(name: str, cands: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Stricter candidate matcher: must share a meaningful non-numeric token."""
    q_tokens = [t for t in re.findall(r"[a-z0-9]+", (name or "").lower()) if t]
    if not q_tokens:
        return None
    q_set = set(q_tokens)
    q_non_num = {t for t in q_set if not t.isdigit()}

    best = None
    best_score = 0.0
    for c in cands or []:
        cand_name = str(c.get("name", "")).lower()
        n_tokens = [t for t in re.findall(r"[a-z0-9]+", cand_name) if t]
        if not n_tokens:
            continue
        n_set = set(n_tokens)
        inter = q_set & n_set
        inter_non_num = q_non_num & n_set
        if not inter_non_num:
            continue
        score = len(inter) / max(1, len(q_set))
        if name.lower() in cand_name or cand_name in name.lower():
            score += 0.5
        if score > best_score:
            best_score = score
            best = c
    return best if best_score >= 0.5 else None

def llm_decide_yes_no(user_text: str) -> str:
    """Use local LLM to classify yes/no/unclear for confirmation."""
    if os.getenv("WAITER_LLM_BACKEND", "disabled") == "disabled":
        return "unclear"
    system = (
        "You are a confirmation detector for a restaurant ordering agent.\n"
        "Reply with exactly one word: yes, no, or unclear."
    )
    out = generate_waiter(system=system, context_menu="", user=user_text, max_tokens=3)
    s = (out or "").strip().lower()
    if s.startswith("yes"): return "yes"
    if s.startswith("no"):  return "no"
    return "unclear"


# ---------- Nodes ----------
@validate_node(name="MenuLookup", tags=["ordering","menu"], input_model=OrderInput, output_model=OrderOutput)
def menu_lookup_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("[NODE] menu_lookup")
    _ensure_lists(state)
    msgs = state["messages"]; md = state["metadata"]
    last = msgs[-1]["content"]

    facets = md.get("facets", {}) or {}
    cat = (facets.get("category") or "").lower()
    want_veg = bool(facets.get("veg"))
    want_nonveg = bool(facets.get("nonveg"))
    page = int(md.get("page", 0))

    facet_terms = []
    if cat: facet_terms.append(cat)
    if want_veg: facet_terms.append("veg")
    if want_nonveg: facet_terms.append("non-veg")
    boosted_query = " ".join([last] + facet_terms).strip() or last

    try:
        results = rag.search_menu(boosted_query, top_k=50)
    except Exception as e:
        print("MenuLookup EXC:", repr(e)); traceback.print_exc()
        msgs.append({"role":"assistant","content":"Menu lookup error. I’ll notify a human."})
        state["messages"] = msgs
        state["metadata"]["_awaiting_worker"] = False
        return state

    def _as_str(x): return ", ".join(map(str, x)) if isinstance(x, list) else str(x or "")
    def _is_cat_ok(x): return True if not cat else (cat in _as_str(x.get("category")).lower())
    def _is_veg_ok(x):
        tags = _as_str(x.get("tags")).lower()
        if want_veg:    return ("veg" in tags) and ("non" not in tags)
        if want_nonveg: return ("non" in tags) or any(t in tags for t in ["chicken","mutton","fish","egg"])
        return True

    filtered = [r for r in results if _is_cat_ok(r) and _is_veg_ok(r)]
    pool = filtered if filtered else results

    PAGE_SIZE = 5
    start, end = page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE
    final = pool[start:end]
    has_more = end < len(pool)

    md["candidates"] = final
    md["route"] = "order"
    md["stage"] = "take"
    md["facets"] = {"category": cat or None, "veg": want_veg, "nonveg": want_nonveg}
    if md.get("last_intent") in {"ordering.lookup", "ordering.lookup_refine"}:
        md["page"] = 0

    if final:
        listing = "\n".join([f"- {r['name']} (₹{r['price']})" for r in final])
        base_reply = (
            "Here are some options:\n"
            f"{listing}\n"
            "Tell me the numbers (e.g., '1 and 3'), or say 'add 2 Garlic Bread'."
        )
        llm_line = None
        if USE_WAITER_LLM:
            try:
                llm_line = generate_waiter(
                    system=(
                        "You are a friendly restaurant waiter.\n"
                        "Keep replies under 2 short sentences.\n"
                        "Do NOT repeat the user's message.\n"
                        "Summarize items and end with a call-to-action to pick by number."
                    ),
                    context_menu="\n".join([f"[{i+1}] {r['name']} — ₹{r['price']}" for i, r in enumerate(final)]),
                    user=f"User asked: {last}",
                    max_tokens=90,
                )
                if not llm_line or llm_line.strip().lower() == last.strip().lower() or len(llm_line.strip()) < 6:
                    llm_line = None
            except Exception:
                llm_line = None
        reply = f"{llm_line}\n{base_reply}" if llm_line else base_reply
    else:
        reply = "I couldn’t find a good match. Try 'veg starters', 'non-veg mains', or name a dish."

    msgs.append({"role":"assistant","content": reply})
    state["messages"] = msgs
    state["metadata"]["_awaiting_worker"] = False
    return state


ORDER_VERBS = (
    "add", "get me", "give me", "i'll have", "i will have",
    "i want", "we'll have", "we will have", "order", "take"
)

@validate_node(name="TakeOrder", tags=["ordering","take"])
def take_order_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("[NODE] take_order")
    _ensure_lists(state)
    md = state["metadata"]; msgs = state["messages"]
    user_raw = msgs[-1]["content"]

    cands = md.get("candidates", [])

    # --- Step 1: Ask LLM to parse structured order ---
    try:
        parsed = generate_json(
            system=f"""
            You are a strict parser. Extract structured order items (dish name + qty) from user text.
            Consider these valid menu candidates: {[c['name'] for c in cands]}.
            Only output JSON matching schema.
            Schema: {{"items":[{{"name": "DishName", "qty": 2}}]}}
            """,
            user=user_raw,
            schema_hint='{"items":[{"name":"Chicken Tikka","qty":2}]}',
            max_tokens=80,
        )

        # Ensure we only take JSON block
        if not isinstance(parsed, dict):
            print("[Order Parse Warning] LLM did not return dict:", parsed)
            parsed = {"items": []}
        items = parsed.get("items", [])
        if not isinstance(items, list):
            items = []
    except Exception as e:
        print("[Order Parse Error]", e)
        items = []

    # --- Step 2: Match items to menu ---
    added = []
    for it in items:
        try:
            name = it.get("name")
            qty = int(it.get("qty", 1))
            if not name:
                continue

            # Prefer candidates first
            menu_item = _best_match_from_candidates(name, cands)
            if not menu_item:
                menu_item = rag.find_by_name(name)

            if menu_item:
                # merge or add new
                existing = next((c for c in md["cart"] if c["id"] == menu_item["id"]), None)
                if existing:
                    existing["qty"] += qty
                else:
                    md["cart"].append({
                        "id": menu_item["id"],
                        "name": menu_item["name"],
                        "price": menu_item["price"],
                        "qty": qty,
                    })
                added.append(f"{qty} x {menu_item['name']}")
        except Exception as e:
            print("[Add Item Error]", e, it)

    # --- Step 3: Build response ---
    if added:
        md["stage"] = "confirm"
        msgs.append({
            "role": "assistant",
            "content": f"Added: {', '.join(added)}.\n{_summarize_cart(md['cart'])}\nShall I place the order? (yes/no)"
        })
        state["metadata"]["_awaiting_worker"] = False
        return state

    # fallback
    msgs.append({
        "role": "assistant",
        "content": "I didn’t catch any items. You can say '2 Paneer 65 and 1 Dal Tadka'."
    })
    state["metadata"]["_awaiting_worker"] = False
    return state




@validate_node(name="ConfirmOrder", tags=["ordering","confirm"], input_model=OrderInput, output_model=OrderOutput)
def confirm_order_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("[NODE] confirm_order")
    _ensure_lists(state)
    md = state["metadata"]; msgs = state["messages"]
    cart = md.get("cart") or []
    user_raw = msgs[-1]["content"].strip()
    signal = md.get("confirm_signal")

    if not cart:
        md["stage"] = "take"
        msgs.append({"role":"assistant","content":"Your cart is empty — tell me what to add."})
        state["messages"] = msgs
        state["metadata"]["_awaiting_worker"] = False
        return state

    # ✅ If router already set a confirm signal, honor it
    if signal in {"confirm.yes", "confirm.no"}:
        decision = "yes" if signal == "confirm.yes" else "no"
    else:
        try:
            parsed = generate_json(
                system="""
                You are a confirmation detector for a restaurant ordering agent.
                Output JSON only.
                Schema: {"decision": "yes"|"no"|"unclear"}
                """,
                user=user_raw,
                schema_hint='{"decision":"yes"}',
                max_tokens=5,
            )
            decision = parsed.get("decision","unclear").lower()
        except Exception as e:
            print("[Confirm Parse Error]", e)
            decision = "unclear"

    if decision == "yes":
        order = {
            "session_id": state["session_id"],
            "ts": int(time.time()),
            "items": cart,
            "total": sum(it["price"] * it.get("qty", 1) for it in cart),
            "status": "PLACED",
        }
        order_id = send_order_to_kitchen(order)
        msgs.append({"role":"assistant",
                     "content": f"✅ Order placed! Total: ₹{order['total']}. Your order id ends with ...{order_id}."})
        md.update({"confirmed": False, "stage": None, "cart": [], "route": None})
        state["messages"] = msgs
        state["metadata"]["_awaiting_worker"] = False
        return state

    if decision == "no":
        md["confirmed"] = False
        md["stage"] = "take"
        msgs.append({"role":"assistant","content":"No worries. Tell me what to add/remove or change quantities."})
        state["messages"] = msgs
        state["metadata"]["_awaiting_worker"] = False
        return state

    # unclear
    md["stage"] = "confirm"
    msgs.append({"role":"assistant","content":"Please reply with 'yes' to place the order, or 'no' to modify."})
    state["messages"] = msgs
    state["metadata"]["_awaiting_worker"] = False
    return state