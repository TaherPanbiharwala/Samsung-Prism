# utils/pos.py
from typing import Dict, Any, List
import time, os, json

def send_order_to_kitchen(order: Dict[str, Any]) -> str:
    """
    Stub for POS tool. Replace this with actual API call to kitchen system.
    Saves to orders.json locally for now.
    """
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

    # simulate order id
    return f"{order['ts'] % 10000}"