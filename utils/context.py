# utils/context.py
from typing import List, Dict

def format_context_menu(items: List[Dict]) -> str:
    if not items:
        return "CONTEXT_MENU: (no items currently available)"
    lines = [f"- {it.get('name')} (₹{it.get('price')}) [{it.get('id','?')}]" for it in items[:5]]
    return "CONTEXT_MENU (top items):\n" + "\n".join(lines)