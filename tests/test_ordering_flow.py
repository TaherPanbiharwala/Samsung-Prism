# tests/test_ordering_flow.py
import os, json, pytest
from nodes import ordering
from state import ChatStateTD
from langgraph.graph import StateGraph, END

def make_state(msg: str) -> ChatStateTD:
    return {
        "session_id": "flow-test",
        "messages": [{"role": "user", "content": msg}],
        "metadata": {},
    }

def test_full_flow_success(tmp_path, monkeypatch):
    """Happy path: lookup -> take -> confirm -> submit"""
    path = tmp_path / "orders.json"
    monkeypatch.chdir(tmp_path)

    # Step 1: menu lookup returns some dishes
    monkeypatch.setattr(ordering.rag, "search_menu",
                        lambda q, top_k=12: [
                            {"id": "I1", "name": "Garlic Bread", "price": 149, "category": "starter"},
                            {"id": "I2", "name": "Tomato Soup", "price": 129, "category": "starter"},
                        ])
    state = make_state("show me starters")
    state = ordering.menu_lookup_node(state)
    assert state["metadata"]["stage"] == "take"
    assert len(state["metadata"]["candidates"]) == 2

    # Step 2: user picks item by number
    state["messages"].append({"role": "user", "content": "1"})
    state = ordering.take_order_node(state)
    assert state["metadata"]["stage"] == "confirm"
    assert state["metadata"]["cart"][0]["name"] == "Garlic Bread"

    # Step 3: user confirms
    state["messages"].append({"role": "user", "content": "yes"})
    state = ordering.confirm_order_node(state)
    assert state["metadata"]["stage"] == "submit"
    assert state["metadata"]["confirmed"] is True

    # Step 4: POS submit writes order file
    state = ordering.pos_submit_node(state)
    assert not state["metadata"]["route"]
    data = json.loads(path.read_text())
    assert data[0]["items"][0]["name"] == "Garlic Bread"


def test_flow_user_says_no_changes(monkeypatch):
    """User rejects order at confirm stage and goes back to take stage"""
    # candidates prepared manually
    state = make_state("add garlic bread")
    state["metadata"]["candidates"] = [{"id":"I1","name":"Garlic Bread","price":149}]
    monkeypatch.setattr(ordering.rag, "find_by_name",
                        lambda name: {"id":"I1","name":"Garlic Bread","price":149})
    state = ordering.take_order_node(state)
    assert state["metadata"]["stage"] == "confirm"

    # User says no
    state["messages"].append({"role": "user", "content": "no"})
    state = ordering.confirm_order_node(state)
    assert state["metadata"]["stage"] == "take"
    assert state["metadata"]["confirmed"] is False


def test_flow_invalid_item_then_valid(monkeypatch):
    """User tries invalid item first, then a valid one"""
    monkeypatch.setattr(ordering.rag, "search_menu",
                        lambda q, top_k=12: [
                            {"id":"I1","name":"Garlic Bread","price":149},
                            {"id":"I2","name":"Tomato Soup","price":129},
                        ])
    state = make_state("starters")
    state = ordering.menu_lookup_node(state)

    # Step 2: invalid item
    monkeypatch.setattr(ordering.rag, "find_by_name", lambda name: None)
    state["messages"].append({"role": "user", "content": "unicorn soup"})
    state = ordering.take_order_node(state)
    assert "didn't catch" in state["messages"][-1]["content"].lower()
    assert state["metadata"]["stage"] == "take"

    # Step 3: valid item
    monkeypatch.setattr(ordering.rag, "find_by_name",
                        lambda name: {"id":"I2","name":"Tomato Soup","price":129})
    state["messages"].append({"role": "user", "content": "tomato soup"})
    state = ordering.take_order_node(state)
    assert state["metadata"]["cart"], "Cart should not be empty"
    assert state["metadata"]["cart"][-1]["name"] == "Tomato Soup"
    assert state["metadata"]["stage"] in {"take", "confirm"}


def test_flow_confirm_without_cart():
    """Edge: confirm is asked but cart is empty"""
    state = make_state("yes")
    state["metadata"]["cart"] = []  # nothing added
    state = ordering.confirm_order_node(state)
    assert state["metadata"]["stage"] == "confirm"
    assert "cart is empty" in state["messages"][-1]["content"].lower()

def test_graph_recursion_guard():
    """Simulate a misconfigured graph with a cycle to ensure recursion guard triggers."""
    # Build a fake tiny graph with a cycle
    g = StateGraph(ChatStateTD)

    def loop_node(state: ChatStateTD) -> ChatStateTD:
        # Just echoes state back, no stage change
        return state

    g.add_node("loop", loop_node)
    g.set_entry_point("loop")
    g.add_edge("loop", "loop")  # cycle

    graph_app = g.compile()

    # Prepare a minimal state
    state: ChatStateTD = {
        "session_id": "recursion-test",
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {},
    }

    # Expect LangGraph to stop with GraphRecursionError
    from langgraph.errors import GraphRecursionError
    with pytest.raises(GraphRecursionError):
        graph_app.invoke(state, config={"recursion_limit": 10})

def test_flow_mixed_valid_and_invalid(monkeypatch):
    state = make_state("2 unicorn soups and 1 garlic bread")
    monkeypatch.setattr(ordering.rag, "find_by_name",
                        lambda name: {"id":"I1","name":"Garlic Bread","price":149} if "garlic" in name.lower() else None)
    state = ordering.take_order_node(state)
    assert "garlic bread" in state["messages"][-1]["content"].lower()
    assert state["metadata"]["stage"] == "confirm"
    assert len(state["metadata"]["cart"]) == 1


def test_flow_multiple_no_responses(monkeypatch):
    state = make_state("add garlic bread")
    monkeypatch.setattr(ordering.rag, "find_by_name",
                        lambda name: {"id":"I1","name":"Garlic Bread","price":149})
    state = ordering.take_order_node(state)
    # user says no twice
    for _ in range(2):
        state["messages"].append({"role": "user", "content": "no"})
        state = ordering.confirm_order_node(state)
        assert state["metadata"]["stage"] == "take"


def test_flow_double_yes(monkeypatch, tmp_path):
    path = tmp_path / "orders.json"
    monkeypatch.chdir(tmp_path)
    # add garlic bread
    state = make_state("garlic bread")
    monkeypatch.setattr(ordering.rag, "find_by_name",
                        lambda name: {"id":"I1","name":"Garlic Bread","price":149})
    state = ordering.take_order_node(state)

    # confirm once → submit
    state["messages"].append({"role":"user","content":"yes"})
    state = ordering.confirm_order_node(state)
    state = ordering.pos_submit_node(state)
    first_reply = state["messages"][-1]["content"]
    assert "order placed" in first_reply.lower()

    # confirm again → should NOT create new entry
    before = json.loads(path.read_text())
    state["messages"].append({"role":"user","content":"yes"})
    state = ordering.confirm_order_node(state)
    state = ordering.pos_submit_node(state)
    after = json.loads(path.read_text())

    # The file content should be unchanged
    assert before == after
    # And the assistant should politely say cart is empty
    assert "cart is empty" in state["messages"][-1]["content"].lower()


def test_flow_empty_confirm():
    state = make_state("yes")
    state["metadata"]["cart"] = [{"id":"I1","name":"Garlic Bread","price":149,"qty":1}]
    state = ordering.confirm_order_node(state)
    # send ambiguous response
    state["messages"].append({"role":"user","content":"maybe"})
    state = ordering.confirm_order_node(state)
    assert state["metadata"]["stage"] == "confirm"
    assert "please reply" in state["messages"][-1]["content"].lower()


def test_flow_number_and_name(monkeypatch):
    state = make_state("1 and tomato soup")
    state["metadata"]["candidates"] = [
        {"id":"I1","name":"Garlic Bread","price":149},
        {"id":"I2","name":"Tomato Soup","price":129},
    ]
    monkeypatch.setattr(ordering.rag, "find_by_name",
                        lambda name: {"id":"I2","name":"Tomato Soup","price":129} if "tomato" in name.lower() else None)
    state = ordering.take_order_node(state)
    assert len(state["metadata"]["cart"]) == 2
    assert state["metadata"]["stage"] == "confirm"