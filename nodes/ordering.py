# Menu lookup, recommendations, confirm
from typing import List, Dict, Any
import re, traceback
from pydantic import BaseModel, Field
from utils.validation import validate_node
from utils import rag
from utils.context import format_context_menu
from state import MessageModel      # ✅ local state module (not "Flow.state")
from utils.nlu import classify
from utils.llm import generate_waiter   # or: from utils.llm import chat as generate_waiter
from utils.config import SYSTEM_PROMPT, USE_WAITER_LLM
from utils.pos import send_order_to_kitchen
import time

# ---------- Schemas ----------

class OrderInput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]] = Field(default_factory=list, min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class OrderOutput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]] = Field(default_factory=list, min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# at top of file (near other helpers)
def _safe_llm_reply(candidate: str, user: str, fallback: str) -> str:
    if not candidate:
        return fallback
    a = candidate.strip().lower()
    b = (user or "").strip().lower()
    if a == b or len(a) < 4:
        return fallback
    return candidate

def _facet_from_text(text: str) -> Dict[str, Any]:
    t = text.lower()
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

def _best_match_from_candidates(name: str, cands: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Stricter candidate matcher: must share a meaningful word, not just numbers."""
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

        # must overlap at least one non-numeric token
        if not inter_non_num:
            continue

        score = len(inter) / max(1, len(q_set))
        # boost full substring matches
        if name.lower() in cand_name or cand_name in name.lower():
            score += 0.5

        if score > best_score:
            best_score = score
            best = c

    return best if best_score >= 0.5 else None
# ---------- Nodes ----------
# (you already import rag, validate_node, format_context_menu, MessageModel, etc.)

@validate_node(name="MenuLookup", tags=["ordering","menu"], input_model=OrderInput, output_model=OrderOutput)
def menu_lookup_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("[NODE] chitchat")
    print(f"[NODE ENTER] menu_lookup: stage={state.get('metadata',{}).get('stage')}, "
          f"intent={state.get('metadata',{}).get('last_intent')}, "
          f"_awaiting={state.get('metadata',{}).get('_awaiting_worker')}")

    _ensure_lists(state)
    msgs = state["messages"]; md = state["metadata"]
    last = msgs[-1]["content"]

    # facet hints (from router or inferred earlier)
    facets = md.get("facets", {}) or {}
    cat = (facets.get("category") or "").lower()
    want_veg = bool(facets.get("veg"))
    want_nonveg = bool(facets.get("nonveg"))
    page = int(md.get("page", 0))  # 0-based

    # ---- 1) Build a facet-boosted query for RAG
    facet_terms = []
    if cat:       facet_terms.append(cat)
    if want_veg:  facet_terms.append("veg")
    if want_nonveg: facet_terms.append("non-veg")
    boosted_query = " ".join([last] + facet_terms).strip() or last

    try:
        results = rag.search_menu(boosted_query, top_k=50)
    except Exception as e:
        print("MenuLookup EXC:", repr(e)); traceback.print_exc()
        msgs.append({"role":"assistant","content": f"Menu lookup error: {e}. I’ll notify a human."})
        state["messages"] = msgs
        state["_error"] = {"node":"MenuLookup","where":"runtime","errors":[{"msg":str(e)}]}
        state["metadata"]["_awaiting_worker"] = False
        # hard guard
        if not state.get("messages") or state["messages"][-1]["role"] != "assistant":
            raise RuntimeError("Worker returned without replying")
        if state["metadata"].get("_awaiting_worker") is not False:
            raise RuntimeError("Worker did not set _awaiting_worker=False")
        return state

    # ---- 2) Filter by facets
    def _as_str(x): return ", ".join(map(str, x)) if isinstance(x, list) else str(x or "")
    def _is_cat_ok(x):
        if not cat: return True
        return cat in _as_str(x.get("category")).lower()
    def _is_veg_ok(x):
        tags = _as_str(x.get("tags")).lower()
        if want_veg:    return ("veg" in tags) and ("non" not in tags)
        if want_nonveg: return ("non" in tags) or any(t in tags for t in ["chicken","mutton","fish","egg"])
        return True

    filtered = [r for r in results if _is_cat_ok(r) and _is_veg_ok(r)]
    pool = filtered if filtered else results   # <-- use filtered if we have any

    # ---- 3) Paging (5 per page)
    PAGE_SIZE = 5
    start = page * PAGE_SIZE
    end = start + PAGE_SIZE
    final = pool[start:end]
    has_more = end < len(pool)

    # Empty? broad fallback by category only
    if not final and not pool:
        broad_q = (cat or "popular")
        try:
            pool = rag.search_menu(broad_q, top_k=20)
        except Exception as e:
            print("MenuLookup fallback EXC:", repr(e)); traceback.print_exc()
            pool = []
        final = pool[:PAGE_SIZE]
        has_more = len(pool) > PAGE_SIZE

    # ---- 4) Persist state for next turn (ALWAYS do this)
    md["candidates"] = final
    md["route"] = "order"
    md["stage"] = "take"  # next user can say "1 and 3" / "add 2 ..."
    md["facets"] = {"category": cat or None, "veg": want_veg, "nonveg": want_nonveg}

    # Reset page unless user explicitly asked "more"
    if md.get("last_intent") in {"ordering.lookup", "ordering.lookup_refine"}:
        md["page"] = 0

    # ---- 5) Build reply
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
                # sanitize: ignore empty/very short or pure echo lines
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

    print(f"[NODE EXIT]  menu_lookup: stage={state.get('metadata',{}).get('stage')}, "
          f"cands={len(state.get('metadata',{}).get('candidates', []))}, "
          f"_awaiting={state.get('metadata',{}).get('_awaiting_worker')}")

    # hard guards
    if not state.get("messages") or state["messages"][-1]["role"] != "assistant":
        raise RuntimeError("Worker returned without replying")
    if state["metadata"].get("_awaiting_worker") is not False:
        raise RuntimeError("Worker did not set _awaiting_worker=False")

    return state

@validate_node(name="Recommend", tags=["ordering","recommend"], input_model=OrderInput, output_model=OrderOutput)
def recommend_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("[NODE] chitchat")
    print(f"[NODE ENTER] menu_lookup: stage={state.get('metadata',{}).get('stage')}, "
      f"intent={state.get('metadata',{}).get('last_intent')}, "
      f"_awaiting={state.get('metadata',{}).get('_awaiting_worker')}")
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
    state["metadata"]["_awaiting_worker"] = False
    print(f"[NODE EXIT]  menu_lookup: stage={state.get('metadata',{}).get('stage')}, "
      f"cands={len(state.get('metadata',{}).get('candidates', []))}, "
      f"_awaiting={state.get('metadata',{}).get('_awaiting_worker')}")
    if not state.get("messages") or state["messages"][-1]["role"] != "assistant":
        raise RuntimeError("Worker returned without replying")
    if state["metadata"].get("_awaiting_worker") is not False:
        raise RuntimeError("Worker did not set _awaiting_worker=False")
    return state

# nodes/ordering.py  (replace take_order_node)
ORDER_VERBS = (
    "add", "get me", "give me", "i'll have", "i will have",
    "i want", "we'll have", "we will have", "order", "take"
)

@validate_node(name="TakeOrder", tags=["ordering","take"], input_model=OrderInput, output_model=OrderOutput)
def take_order_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("[NODE] take_order")
    print(f"[NODE ENTER] take_order: stage={state.get('metadata',{}).get('stage')}, "
          f"intent={state.get('metadata',{}).get('last_intent')}, "
          f"_awaiting={state.get('metadata',{}).get('_awaiting_worker')}")

    _ensure_lists(state)
    md = state["metadata"]
    msgs = state["messages"]

    user_raw = msgs[-1]["content"]
    user = user_raw.lower().strip()

    # --- slot hints from router ---
    slots = md.get("last_slots") or {}
    has_numbers = str(slots.get("has_numbers", "False")).lower() == "true"
    has_addverb = str(slots.get("has_add_verb", "False")).lower() == "true"

    # -- helper: add (or bump qty) in cart by menu item dict --
    def _add_cart_item(it: Dict[str, Any], qty: int = 1):
        for c in md["cart"]:
            if c.get("id") == it["id"]:
                c["qty"] = c.get("qty", 1) + qty
                return
        md["cart"].append({
            "id": it["id"], "name": it["name"], "price": it["price"], "qty": qty
        })

    cands: List[Dict[str, Any]] = md.get("candidates", []) or []
    added: List[str] = []

    # 1) numbers referencing candidates directly: "1 and 3"
    if cands and (has_numbers or re.search(r"\b\d+(?:\s*(?:,|and)\s*\d+)*\b", user)):
        nums = re.findall(r"\b(\d+)\b", user)
        for n in nums:
            idx = int(n) - 1
            if 0 <= idx < len(cands):
                it = cands[idx]
                _add_cart_item(it, 1)
                added.append(it["name"])

    # 2) explicit qty + name pairs (can be multiple!)
    qty_name_gate = has_addverb or bool(re.search(r"\b\d+\s*x?\s+[A-Za-z]", user))

    if not added and qty_name_gate:
        # 🔹 Find all pairs in the utterance, e.g. "3 chicken 65", "2 gobi 65"
        pairs = re.findall(r"(\d+)\s*x?\s*([a-z][a-z0-9\s\-\&']+)", user_raw, flags=re.I)

        for qty_str, name in pairs:
            qty = int(qty_str)
            name = name.strip()

            it = _best_match_from_candidates(name, cands) or rag.find_by_name(name)
            if it:
                _add_cart_item(it, qty)
                added.append(f"{qty} x {it['name']}")

        # If no (qty, name) pairs matched, but an add-verb exists → fallback "add paneer 65"
        if not added and has_addverb:
            cleaned = re.sub(
                r"^(add|get me|give me|i'?ll have|i will have|i want|"
                r"we'?ll have|we will have|order|take)\s+",
                "",
                user_raw,
                flags=re.I,
            ).strip()
            if cleaned:
                it = _best_match_from_candidates(cleaned, cands) or rag.find_by_name(cleaned)
                if it:
                    _add_cart_item(it, 1)
                    added.append(it["name"])

    # ✅ If items were added
    if added:
        md["stage"] = "confirm"
        msgs.append({"role": "assistant", "content":
            f"Added: {', '.join(added)}.\n{_summarize_cart(md['cart'])}\nShall I place the order? (yes/no)"})
        state["messages"] = msgs
        state["metadata"]["_awaiting_worker"] = False
        print(f"[NODE EXIT]  take_order: stage={md.get('stage')}, "
              f"cands={len(md.get('candidates', []))}, _awaiting={md.get('_awaiting_worker')}")
        return state

    # --- Nothing added → routing logic ---
    if any(k in user for k in ["more","more options","show more","next","another","others"]):
        md["last_intent"] = "ordering.more"
        md["stage"] = "menu"
        md["facets"] = md.get("facets") or _facet_from_text(user_raw)
        print(f"[NODE EXIT]  take_order -> route: stage={md.get('stage')}, intent=ordering.more")
        return state

    looks_like_menu_refine = any(w in user for w in [
        "starter","starters","main","mains","main course",
        "dessert","desserts","sweet","beverage","beverages","drink","drinks",
        "veg","non veg","non-veg"
    ])
    if looks_like_menu_refine and not has_addverb and not has_numbers:
        md["last_intent"] = "ordering.lookup_refine"
        md["stage"] = "menu"
        md["facets"] = _facet_from_text(user_raw)
        print(f"[NODE EXIT]  take_order -> route: stage={md.get('stage')}, intent=ordering.lookup_refine")
        return state

    # --- Still nothing: helpful fallback ---
    md["stage"] = "take"
    msgs.append({"role": "assistant", "content":
        "I didn't catch any items to add. Pick by number (e.g., '1 and 3') or say 'add 2 Garlic Naan'. "
        "You can also say a category like 'veg starters' and I’ll show options."})
    state["messages"] = msgs
    state["metadata"]["_awaiting_worker"] = False
    print(f"[NODE EXIT]  take_order: stage={md.get('stage')}, "
          f"cands={len(md.get('candidates', []))}, _awaiting={md.get('_awaiting_worker')}")
    return state


@validate_node(name="ConfirmOrder", tags=["ordering","confirm"], input_model=OrderInput, output_model=OrderOutput)
def confirm_order_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("[NODE] confirm_order")
    _ensure_lists(state)
    md = state["metadata"]; msgs = state["messages"]
    cart = md.get("cart") or []
    user = msgs[-1]["content"].strip().lower()
    signal = md.get("confirm_signal")

    # If no items, back to take
    if not cart:
        md["stage"] = "take"
        msgs.append({"role":"assistant","content":"Your cart is empty — tell me what to add."})
        state["messages"] = msgs
        state["metadata"]["_awaiting_worker"] = False
        return state

    # ✅ YES: send to POS immediately
    if signal == "confirm.yes" or user in {"yes","y","yeah","yep","yup","sure","ok","okay","confirm","place","definitely","absolutely","go ahead"}:
        order = {
            "session_id": state["session_id"],
            "ts": int(time.time()),
            "items": cart,
            "total": sum(it["price"] * it.get("qty", 1) for it in cart),
            "status": "PLACED",
        }
        order_id = send_order_to_kitchen(order)

        msgs.append({"role":"assistant","content": f"✅ Order placed! Total: ₹{order['total']}. Your order id ends with ...{order_id}."})

        # reset flow
        md["confirmed"] = False
        md["stage"] = None
        md["cart"] = []
        md["route"] = None
        state["messages"] = msgs
        state["metadata"]["_awaiting_worker"] = False
        return state

    # ❌ NO: cancel
    if signal == "confirm.no" or user in {"no","n","nope","nah","change","edit","later","cancel","stop"}:
        md["confirmed"] = False
        md["stage"] = "take"
        msgs.append({"role":"assistant","content":"No worries. Tell me what to add/remove or new quantities."})
        state["messages"] = msgs
        state["metadata"]["_awaiting_worker"] = False
        return state

    # else: ask again
    md["stage"] = "confirm"
    msgs.append({"role":"assistant","content":"Please reply with 'yes' to place the order, or 'no' to modify."})
    state["messages"] = msgs
    state["metadata"]["_awaiting_worker"] = False
    return state

