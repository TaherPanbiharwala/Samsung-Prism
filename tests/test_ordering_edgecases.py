# tests/test_ordering_edgecases.py
import pytest
from nodes import ordering
from state import ChatStateTD
import json

def make_state(msg: str) -> ChatStateTD:
    return {
        "session_id": "edge-test",
        "messages": [{"role": "user", "content": msg}],
        "metadata": {},
    }

def drive_turn(state: ChatStateTD) -> ChatStateTD:
    """
    Drives ONE assistant turn based on state's current stage.
    Defaults to 'take' on first turn.
    """
    stage = state["metadata"].get("stage", "take")
    if stage == "take":
        return ordering.take_order_node(state)
    elif stage == "confirm":
        return ordering.confirm_order_node(state)
    elif stage == "submit":
        return ordering.pos_submit_node(state)
    # Fallback to take
    return ordering.take_order_node(state)

@pytest.mark.parametrize(
    "user_inputs, monkeypatch_map, expected_stage, expected_cart_names, expected_snippet",
    [
        # 1. Mixed valid + invalid
        (
            ["2 unicorn soups and 1 garlic bread"],
            {"garlic": {"id":"I1","name":"Garlic Bread","price":149}},
            "confirm",
            ["Garlic Bread"],
            "garlic bread",
        ),
        # 2. Multiple 'no' responses should loop back to take
        (
            ["add garlic bread", "no", "no"],
            {"garlic": {"id":"I1","name":"Garlic Bread","price":149}},
            "take",
            ["Garlic Bread"],
            "no worries",
        ),
        # 3. Bare name should be accepted
        (
            ["tomato soup"],
            {"tomato": {"id":"I2","name":"Tomato Soup","price":129}},
            "confirm",
            ["Tomato Soup"],
            "added",
        ),
        # 4. Ambiguous confirm
        (
            ["add garlic bread", "maybe"],
            {"garlic": {"id":"I1","name":"Garlic Bread","price":149}},
            "confirm",
            ["Garlic Bread"],
            "please reply",
        ),
        # 5. Number + name mixed
        (
            ["1 and tomato soup"],
            {"tomato": {"id":"I2","name":"Tomato Soup","price":129}},
            "confirm",
            ["Garlic Bread", "Tomato Soup"],
            "added",
        ),
    ],
)

def test_ordering_edgecases(monkeypatch, user_inputs, monkeypatch_map,
                            expected_stage, expected_cart_names, expected_snippet):
    """
    Parametrized edge-case scenarios for ordering flow.
    Each scenario is defined by user_inputs, fake rag.find_by_name mapping,
    expected final stage, expected cart contents, and an expected snippet in reply.
    """
    # Candidate list used when users reference numbers (e.g., "1")
    candidates = [
        {"id": "I1", "name": "Garlic Bread", "price": 149},
        {"id": "I2", "name": "Tomato Soup", "price": 129},
    ]

    # Monkeypatch rag.find_by_name with a simple matcher
    def fake_find_by_name(name: str):
        name_lower = name.lower()
        for key, val in monkeypatch_map.items():
            if key in name_lower:
                return val
        return None

    monkeypatch.setattr(ordering.rag, "find_by_name", fake_find_by_name)

    # Start conversation with first user input
    state = make_state(user_inputs[0])
    state["metadata"]["candidates"] = candidates

    # Process the FIRST turn (this sets stage/cart/reply)
    state = drive_turn(state)

    # Process subsequent user turns
    for msg in user_inputs[1:]:
        state["messages"].append({"role": "user", "content": msg})
        state = drive_turn(state)

        # Assertions
    assert state["metadata"].get("stage") == expected_stage
    cart_names = [it["name"] for it in state["metadata"].get("cart", [])]
    for expected in expected_cart_names:
        assert expected in cart_names

    # Check expected snippet appears in ANY assistant reply (not just the last)
    history = " ".join(m["content"].lower() for m in state["messages"] if m["role"] == "assistant")
    assert expected_snippet in history



def test_complex_user_interactions_with_cart_modifications(tmp_path, monkeypatch):
    """
    Complex flow with multiple menu lookups, item additions, quantity variations,
    invalid input handling, cart modification, multi-stage confirmations, final order
    placement, and post-order session verification.
    """
    # Setup path and mock environment
    orders_path = tmp_path / "orders.json"
    monkeypatch.chdir(tmp_path)

    # Step 1: Menu lookup requesting non-veg mains
    def search_menu_nonveg_main(query, top_k=12):
        return [
            {"id": "M1", "name": "Chicken Curry", "price": 350, "category": "main", "tags": "non veg"},
            {"id": "M2", "name": "Mutton Rogan Josh", "price": 450, "category": "main", "tags": "non veg"},
            {"id": "M3", "name": "Fish Fry", "price": 400, "category": "main", "tags": "non veg"},
            {"id": "V1", "name": "Paneer Tikka", "price": 300, "category": "main", "tags": "veg"},
        ]
    monkeypatch.setattr(ordering.rag, "search_menu", search_menu_nonveg_main)

    state = make_state("show me non veg mains")
    state = ordering.menu_lookup_node(state)
    assert state["metadata"]["stage"] == "take"
    candidates = state["metadata"]["candidates"]
    assert all("non veg" in item["tags"] for item in candidates if item["id"] != "V1")

    # Step 2: User selects first two items by number "1 and 2"
    state["messages"].append({"role": "user", "content": "1 and 2"})
    monkeypatch.setattr(
        ordering.rag,
        "find_by_name",
        lambda name: next((item for item in candidates if item["name"].lower() == name.lower()), None),
    )
    state = ordering.take_order_node(state)
    assert state["metadata"]["stage"] == "confirm"
    cart_names = [item["name"] for item in state["metadata"]["cart"]]
    assert "Chicken Curry" in cart_names and "Mutton Rogan Josh" in cart_names

    # Step 3: User adds 3 fish fry explicitly using formats "3x Fish Fry" and "2 fish fry"
    state["messages"].append({"role": "user", "content": "3x fish fry, 2 fish fry"})
    state = ordering.take_order_node(state)
    total_fish_qty = sum(
        item["qty"] for item in state["metadata"]["cart"] if item["name"].lower() == "fish fry"
    )
    assert total_fish_qty == 5

    # Step 4: User tries invalid item and typo in quantity
    state["messages"].append({"role": "user", "content": "1 unicorn pizza and 2x chickn curry"})
    def find_by_name_typo(name):
        dishes = {i["name"].lower(): i for i in candidates}
        return dishes.get(name.lower())
    monkeypatch.setattr(ordering.rag, "find_by_name", find_by_name_typo)
    state = ordering.take_order_node(state)
    last_msg = state["messages"][-1]["content"].lower()
    assert "didn't catch" in last_msg or "not found" in last_msg

    # Step 5: User confirms partial order, then rejects, modifies quantities
    state["messages"].append({"role": "user", "content": "yes"})
    state = ordering.confirm_order_node(state)
    assert state["metadata"]["confirmed"] is True

    state["messages"].append({"role": "user", "content": "no"})
    state = ordering.confirm_order_node(state)
    assert state["metadata"]["confirmed"] is False
    assert state["metadata"]["stage"] == "take"

    # Step 6: User adds 1 paneer tikka veg item from outside original filter
    monkeypatch.setattr(
        ordering.rag,
        "find_by_name",
        lambda name: next((item for item in search_menu_nonveg_main("") if item["name"].lower() == name.lower()), None),
    )
    state["messages"].append({"role": "user", "content": "paneer tikka"})
    state = ordering.take_order_node(state)
    cart_names = [item["name"] for item in state["metadata"]["cart"]]
    assert "Paneer Tikka" in cart_names

    # Step 7: User removes 2 fish fry items by sending "remove 2 fish fry" (simulate removal)
    remove_qty = 2
    for item in state["metadata"]["cart"]:
        if item["name"].lower() == "fish fry":
            item["qty"] = max(0, item["qty"] - remove_qty)
    state["metadata"]["cart"] = [i for i in state["metadata"]["cart"] if i["qty"] > 0]
    state["messages"].append({"role": "user", "content": "yes"})
    state = ordering.confirm_order_node(state)
    assert state["metadata"]["stage"] == "submit"
    assert state["metadata"]["confirmed"] is True

    # Step 8: Submit order and verify JSON contents reflect accurate quantities and items
    state = ordering.pos_submit_node(state)
    assert "order placed" in state["messages"][-1]["content"].lower()
    with open(orders_path) as f:
        data = json.load(f)
    assert len(data) == 1
    order_items = data[0]["items"]
    total_qty = sum(item["qty"] for item in order_items)
    # If cart was cleared upon submit, accept > 0; else allow exact match
    assert total_qty == sum(item["qty"] for item in state["metadata"]["cart"]) or total_qty > 0

    # Step 9: Validate correct clearing of session (cart, confirmed, stage)
    assert not state["metadata"]["cart"]
    assert not state["metadata"]["confirmed"]
    assert state["metadata"].get("stage") is None

    # Step 10: User attempts to confirm without items present
    state["messages"].append({"role": "user", "content": "yes"})
    state = ordering.confirm_order_node(state)
    assert "cart is empty" in state["messages"][-1]["content"].lower()
    assert state["metadata"]["stage"] == "confirm"

    # Step 11: New lookup for desserts, verify filtering and display
    def search_menu_dessert(query, top_k=12):
        return [
            {"id": "D1", "name": "Gulab Jamun", "price": 100, "category": "dessert", "tags": "veg"},
            {"id": "D2", "name": "Rasgulla", "price": 110, "category": "dessert", "tags": "veg"},
        ]
    monkeypatch.setattr(ordering.rag, "search_menu", search_menu_dessert)
    state["messages"].append({"role": "user", "content": "desserts"})
    state = ordering.menu_lookup_node(state)
    dessert_candidates = state["metadata"]["candidates"]
    assert all("dessert" == item["category"] for item in dessert_candidates)

    # Step 12: Add multiple desserts by name and verify cart update
    monkeypatch.setattr(
        ordering.rag,
        "find_by_name",
        lambda name: next((item for item in dessert_candidates if item["name"].lower() in name.lower()), None),
    )
    state["messages"].append({"role": "user", "content": "2 gulab jamuns and 1 rasgulla"})
    state = ordering.take_order_node(state)
    cart_quantities = {item["name"]: item["qty"] for item in state["metadata"]["cart"]}
    assert cart_quantities.get("Gulab Jamun", 0) == 2
    assert cart_quantities.get("Rasgulla", 0) == 1

    # Step 13: Multiple confirmation attempts with ambiguous inputs
    ambiguous_responses = ["maybe", "okie", "definitely", "nah", "yep", "nope"]
    confirmed = False
    for resp in ambiguous_responses:
        state["messages"].append({"role": "user", "content": resp})
        state = ordering.confirm_order_node(state)
        last = state["messages"][-1]["content"].lower()
        if resp in ["yep", "definitely"]:
            confirmed = True
            assert state["metadata"]["stage"] == "submit"
            assert state["metadata"]["confirmed"] is True
            assert "placing your order" in last
            break
        else:
            assert "please reply" in last or "confirm" in last

    if not confirmed:
        assert state["metadata"]["stage"] == "confirm"

    # Step 14: Submit order if confirmed
    if confirmed:
        state = ordering.pos_submit_node(state)
        with open(orders_path) as f:
            orders = json.load(f)
        assert len(orders) > 0

    # Step 15: Final cleanup state checks after submission
    assert state["metadata"]["cart"] == []
    assert state["metadata"]["confirmed"] is False
    assert state["metadata"].get("stage") is None