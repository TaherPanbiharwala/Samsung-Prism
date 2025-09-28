# tests/test_nlu.py
from utils.nlu import classify

def test_confirm_yes_variants():
    for t in ["yes", "yep", "definitely", "go ahead", "ok"]:
        out = classify(t, stage="confirm")
        assert out["label"] == "confirm.yes"
        assert out["confidence"] >= 0.65

def test_confirm_no_variants():
    for t in ["no", "nope", "nah", "cancel"]:
        out = classify(t, stage="confirm")
        assert out["label"] == "confirm.no"
        assert out["confidence"] >= 0.65

def test_take_more_synonyms():
    for t in ["more options", "next", "another"]:
        out = classify(t, stage="take")
        assert out["label"] == "ordering.more"
        assert out["confidence"] >= 0.6

def test_take_numbers_and_add():
    out = classify("1 and 3", stage="take")
    assert out["label"] == "ordering.take"
    assert out["confidence"] >= 0.6
    out2 = classify("add 2 garlic bread", stage="take")
    assert out2["label"] == "ordering.take"
    assert out2["confidence"] >= 0.6

def test_none_top_level():
    out = classify("i want to order", stage=None)
    assert out["label"] in {"ordering.lookup","chitchat","payment"}