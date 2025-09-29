# ordering.py
from typing import List, Dict, Any, Optional
import os, re, json, time, traceback,string

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
# --- add these helpers near your other helpers ---
def _fallback_extract_items(user_text: str, menu: List[Dict[str, Any]]) -> list[dict]:
    """
    Very simple matcher:
    - Finds quantities like '2 garlic naan' or standalone dish names
    - Matches against your menu names (case-insensitive, token overlap)
    - Defaults action='add'
    """
    text = user_text.lower()
    names = [m["name"] for m in menu]
    items = []

    # 1) qty + name (e.g., '2 garlic naan', '1 butter chicken')
    qty_name = re.findall(r"(\d+)\s+([a-z][a-z0-9\s\-']+)", text)
    for q, raw in qty_name:
        q = int(q)
        raw = raw.strip()
        best = None; best_overlap = 0
        raw_tokens = set(re.findall(r"[a-z0-9]+", raw))
        for nm in names:
            nm_l = nm.lower()
            nm_tokens = set(re.findall(r"[a-z0-9]+", nm_l))
            ov = len(raw_tokens & nm_tokens)
            if ov > best_overlap:
                best_overlap = ov; best = nm
        if best and best_overlap > 0:
            items.append({"name": best, "qty": q, "action": "add"})

    # 2) bare dish names (no qty) → qty=1
    for nm in names:
        if nm.lower() in text and not any(nm == i["name"] for i in items):
            items.append({"name": nm, "qty": 1, "action": "add"})

    return items


def _get_menu_pool(md: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Use current candidates if present; otherwise fall back to the full menu.
    Works even if you don't have a dedicated 'load_full_menu' helper.
    """
    cands = md.get("candidates") or []
    if cands:
        return cands
    try:
        # try an explicit full-menu call if you have it
        if hasattr(rag, "load_full_menu"):
            return rag.load_full_menu()
    except Exception:
        pass
    try:
        # broad search fallback (empty query / popular)
        return rag.search_menu("popular", top_k=500)
    except Exception:
        return []

_WORD = r"[a-zA-Z][a-zA-Z\-\(\)\s']*"  # item name-ish tokens

def _regex_parse_updates(text: str) -> List[Dict[str, Any]]:
    """
    Very forgiving parser: extracts a list of {name, qty, action} from free text.
    Recognizes verbs: add / remove / delete and bare "1 butter chicken".
    """
    t = (text or "").lower()

    # normalize separators
    parts = re.split(r"\s*(?:,| and | & | plus | with )\s*", t)
    items: List[Dict[str, Any]] = []

    for p in parts:
        p = p.strip()
        if not p:
            continue

        action = "add"
        if re.search(r"\b(remove|delete|rm)\b", p):
            action = "remove"

        # e.g., "add 2 garlic naan", "2 garlic naan", "garlic naan", "remove butter chicken"
        m = re.search(r"(?:add|give me|get me|order|take|i'?ll have|i will have|i want|remove|delete)?\s*(\d+)?\s*(" + _WORD + ")", p)
        if not m:
            continue

        qty_s, name = m.groups()
        qty = int(qty_s) if qty_s and qty_s.isdigit() else 1
        name = name.strip()
        # filter out ultra-generic fragments
        if len(name) < 3:
            continue

        items.append({"name": name, "qty": qty, "action": action})

    # merge duplicates within this utterance
    merged: Dict[tuple, Dict[str, Any]] = {}
    for it in items:
        key = (it["name"], it["action"])
        if key in merged:
            merged[key]["qty"] += it["qty"]
        else:
            merged[key] = it
    return list(merged.values())

def _norm_tokens(s: str) -> List[str]:
    if not s:
        return []
    s = s.lower()
    table = str.maketrans("", "", string.punctuation)
    s = s.translate(table)
    toks = [t for t in s.split() if t not in {"and","with","of","the","a","an"}]
    # naive singularize
    out = []
    for t in toks:
        if len(t) > 3 and t.endswith("es"):
            out.append(t[:-2])
        elif len(t) > 2 and t.endswith("s"):
            out.append(t[:-1])
        else:
            out.append(t)
    return out

def _score_match(qname: str, cand_name: str) -> float:
    qt = set(_norm_tokens(qname))
    ct = set(_norm_tokens(cand_name))
    if not qt or not ct:
        return 0.0
    inter = qt & ct
    score = len(inter) / max(1, len(qt))
    # small bonus for substring containment
    if qname.lower() in cand_name.lower() or cand_name.lower() in qname.lower():
        score += 0.25
    return score

def _best_menu_match(name: str, pool: List[Dict[str,Any]]) -> Optional[Dict[str,Any]]:
    best = None
    best_score = 0.0
    for c in pool or []:
        s = _score_match(name, c.get("name",""))
        if s > best_score:
            best_score = s
            best = c
    return best if best_score >= 0.45 else None  # threshold tuned a bit lower

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
    md = state["metadata"]
    msgs = state["messages"]
    user_raw = msgs[-1]["content"]

    # ---- Candidate pool ----
    cands = md.get("candidates", [])
    if not cands:
        try:
            cands = rag.search_menu("popular", top_k=145)
        except Exception:
            cands = []
    cand_names = [c["name"] for c in (cands or [])]

    # ---- Ask LLM for structured cart updates ----
    schema_hint = '{"items":[{"name":"Butter Chicken","qty":1,"action":"add"}]}'
    system = f"""
You are a cart update extractor for a restaurant ordering agent.
Return ONLY JSON (no code fences, no prose).
Schema: {schema_hint}
Rules:
- Each item: name (string), qty (int), action ("add"|"remove").
- If qty missing → 1, if action missing → "add".
- Prefer dish names close to these candidates: {cand_names}.
- If nothing found, return {{"items":[]}}.
"""

    parsed = {}
    try:
        parsed = generate_json(system=system, user=user_raw, schema_hint='{"items":[]}', max_tokens=128)
    except Exception as e:
        print("[Order Parse Error]", e)
    print("[DEBUG parsed LLM JSON]", parsed)

    items = parsed.get("items") or []

    # ---- Quick fallback parser for 'remove ...' utterances ----
    # Handles: "remove butter chicken", "delete 1 naan", "rm 2 garlic naans"
    if not items:
        import re
        rm_items = []
        for m in re.finditer(r"\b(remove|delete|rm)\s+(?:(\d+)\s+)?([a-zA-Z][\w\s\-&]+)", user_raw, re.I):
            qty_str = m.group(2)
            name_str = m.group(3).strip()
            # If no qty given for remove → remove ALL of that item
            rm_items.append({"name": name_str, "qty": int(qty_str) if qty_str else None, "action": "remove"})
        if rm_items:
            items = rm_items

    # —— Fallback when still nothing —— #
    if not items:
        menu_pool = (cands or []) or rag.search_menu("popular", top_k=145)
        items = _fallback_extract_items(user_raw, menu_pool)

    added, removed, changed = [], [], []

    # ---- Fuzzy matching helpers ----
    import string
    def _norm_tokens(s: str):
        if not s: return []
        s = s.lower()
        table = str.maketrans("", "", string.punctuation)
        s = s.translate(table)
        toks = [t for t in s.split() if t not in {"and","with","of","the","a","an"}]
        out = []
        for t in toks:
            if len(t) > 3 and t.endswith("es"):
                out.append(t[:-2])
            elif len(t) > 2 and t.endswith("s"):
                out.append(t[:-1])
            else:
                out.append(t)
        return out

    def _score_match(qname: str, cand_name: str) -> float:
        qt, ct = set(_norm_tokens(qname)), set(_norm_tokens(cand_name))
        if not qt or not ct:
            return 0.0
        inter = qt & ct
        score = len(inter) / max(1, len(qt))
        if qname.lower() in cand_name.lower() or cand_name.lower() in qname.lower():
            score += 0.25
        return score

    def _best_menu_match(name: str, pool: List[Dict[str,Any]]):
        best, best_score = None, 0.0
        for c in pool or []:
            s = _score_match(name, c.get("name",""))
            if s > best_score:
                best_score, best = s, c
        return best if best_score >= 0.45 else None

    # ---- Apply items ----
    for it in items:
        try:
            name = str(it.get("name","")).strip()
            if not name:
                continue
            action = (it.get("action","add") or "add").lower()
            qty_val = it.get("qty", 1)
            qty = int(qty_val) if isinstance(qty_val, (int, str)) and str(qty_val).isdigit() else None

            # Try candidates → global → targeted search
            menu_item = _best_menu_match(name, cands) or rag.find_by_name(name)
            if not menu_item:
                try:
                    hits = rag.search_menu(name, top_k=10)
                except Exception:
                    hits = []
                menu_item = _best_menu_match(name, hits)

            if not menu_item:
                print("[WARN] no menu match for", name)
                continue

            existing = next((x for x in md["cart"] if x["name"].lower() == menu_item["name"].lower()), None)

            if action == "add":
                eff_qty = qty if qty is not None else 1
                if existing:
                    existing["qty"] += eff_qty
                    changed.append(f"+{eff_qty} {menu_item['name']}")
                else:
                    menu_copy = dict(menu_item)
                    menu_copy["qty"] = eff_qty
                    md["cart"].append(menu_copy)
                    added.append(f"{eff_qty} x {menu_item['name']}")

            elif action == "remove":
                if not existing:
                    # nothing to remove
                    continue
                if qty is None:
                    # no qty provided → remove the whole line
                    md["cart"].remove(existing)
                    removed.append(menu_item["name"])
                else:
                    existing["qty"] -= qty
                    if existing["qty"] <= 0:
                        md["cart"].remove(existing)
                        removed.append(menu_item["name"])
                    else:
                        changed.append(f"-{qty} {menu_item['name']}")
        except Exception as e:
            print("[Order Apply Error]", e)
            continue

    # ---- Build reply ----
    if added or removed or changed:
        md["stage"] = "confirm" if md["cart"] else "take"
        summary = _summarize_cart(md["cart"]) if md["cart"] else "Your cart is now empty."
        parts = []
        if added:   parts.append("Added: " + ", ".join(added))
        if changed: parts.append("Updated: " + ", ".join(changed))
        if removed: parts.append("Removed: " + ", ".join(removed))
        msg = ";\n".join(parts) + f".\n{summary}"
        if md["cart"]:
            msg += "\nShall I place the order? (yes/no)"
        msgs.append({"role": "assistant", "content": msg})
    else:
        msgs.append({
            "role": "assistant",
            "content": "I didn’t catch any updates. Try 'add 2 Garlic Naan and 1 Butter Chicken' or 'remove Butter Chicken'."
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