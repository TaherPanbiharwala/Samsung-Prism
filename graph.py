# graph.py
import os
from langgraph.graph import StateGraph, END
from state import ChatStateTD

from nodes.router import router_node
from nodes.ordering import (
    menu_lookup_node, take_order_node, confirm_order_node, chitchat_node, om_router
)
from nodes.payment import (
    bill_node, split_bill_node, payment_gateway_node, feedback_node, pm_router
)
from nodes.voice_hooks import stt_listen, tts_speak

# ---------- ordering manager as a subgraph ----------

def asr_node(state):
    """Take audio_in_path from metadata, transcribe, append as user message."""
    md = state.setdefault("metadata", {})
    audio_path = md.pop("audio_in_path", None)
    if not audio_path:
        return state

    try:
        text = stt_listen(audio_path)
        if text:
            state["messages"].append({"role": "user", "content": text})
    except Exception as e:
        state["messages"].append({"role": "assistant", "content": f"(ASR error: {e})"})

    md["_awaiting_worker"] = False
    return state


def tts_node(state):
    """Take last assistant message, synthesize to audio_out_path."""
    md = state.setdefault("metadata", {})
    msgs = state.get("messages", [])

    if not msgs or msgs[-1]["role"] != "assistant":
        return state

    text = msgs[-1]["content"]
    out_path = "tts_output.wav"
    try:
        tts_speak(text, out_path=out_path)
        md["audio_out_path"] = out_path
    except Exception as e:
        md["audio_out_path"] = None
        msgs.append({"role": "assistant", "content": f"(TTS error: {e})"})

    md["_awaiting_worker"] = False
    return state

def build_ordering_manager():
    g = StateGraph(state_schema=ChatStateTD)

    # workers
    g.add_node("menu_lookup", menu_lookup_node)
    g.add_node("take_order", take_order_node)
    g.add_node("confirm_order", confirm_order_node)
    g.add_node("chitchat", chitchat_node)

    # entry = mini router (function om_router is imported)
    g.add_node("om_router", lambda s: s)
    g.set_entry_point("om_router")

    # route to workers or END
    g.add_conditional_edges(
        "om_router",
        om_router,
        {
            "menu_lookup": "menu_lookup",
            "take_order": "take_order",
            "confirm_order": "confirm_order",
            "chitchat": "chitchat",
            END: END,
        },
    )

    # Workers END the subgraph (no loop-back inside the manager)
    g.add_edge("menu_lookup", END)
    g.add_edge("take_order", END)
    g.add_edge("confirm_order", END)
    g.add_edge("chitchat", END)

    return g.compile()


# ---------- payment manager as a subgraph ----------

def build_payment_manager():
    g = StateGraph(state_schema=ChatStateTD)

    # workers
    g.add_node("bill", bill_node)
    g.add_node("split_bill", split_bill_node)
    g.add_node("payment_gateway", payment_gateway_node)
    g.add_node("feedback", feedback_node)

    # entry = mini router
    g.add_node("pm_router", lambda s: s)
    g.set_entry_point("pm_router")

    g.add_conditional_edges(
        "pm_router",
        pm_router,
        {
            "bill": "bill",
            "split_bill": "split_bill",
            "payment_gateway": "payment_gateway",
            "feedback": "feedback",
            END: END,
        },
    )

    # Workers END the subgraph (no loop-back inside the manager)
    g.add_edge("bill", END)
    g.add_edge("split_bill", END)
    g.add_edge("payment_gateway", END)
    g.add_edge("feedback", END)

    return g.compile()


# ---------- top-level routing helpers ----------

def _route_to_manager(state):
    md = state.get("metadata", {})
    last_intent = (md.get("last_intent") or "").lower()
    route = (md.get("route") or "").lower()

    if last_intent == "app.exit":
        return END
    if last_intent == "chitchat":
        return "ordering_manager"
    if route == "order":
        return "ordering_manager"
    if route == "payment":
        return "payment_manager"
    return "error"


def _entry_switch(state):
    """Kick off with ASR if an audio file is waiting; else go straight to router."""
    md = state.get("metadata", {})
    return "asr" if md.get("audio_in_path") else "router"


def _maybe_tts(state):
    """If VOICE_TTS_ALWAYS=1 or metadata.speak is truthy, go to TTS once; else end the turn."""
    md = state.get("metadata", {})
    if os.getenv("VOICE_TTS_ALWAYS", "1") == "1":
        return "tts"
    return "tts" if md.get("speak") else END


def error_node(state):
    msgs = state.get("messages", [])
    md = state.setdefault("metadata", {})
    msgs.append({"role": "assistant", "content": "Sorry, I hit an error. Let’s start over."})
    md["_awaiting_worker"] = False
    state["messages"] = msgs
    return state


# ---------- build top-level graph ----------

def build_graph():
    ordering_manager = build_ordering_manager()
    payment_manager = build_payment_manager()

    g = StateGraph(state_schema=ChatStateTD)

    # multimodal leaves
    g.add_node("asr", asr_node)
    g.add_node("tts", tts_node)

    # core nodes
    g.add_node("entry", lambda s: s)  # tiny switch: ASR or router
    g.add_node("router", router_node)
    g.add_node("ordering_manager", ordering_manager)
    g.add_node("payment_manager", payment_manager)
    g.add_node("error", error_node)

    # entry
    g.set_entry_point("entry")
    g.add_conditional_edges(
        "entry",
        _entry_switch,
        {
            "asr": "asr",
            "router": "router",
            END: END,
        },
    )
    g.add_edge("asr", "router")  # after ASR → route as normal

    # router → managers (or error/end)
    g.add_conditional_edges(
        "router",
        _route_to_manager,
        {
            "ordering_manager": "ordering_manager",
            "payment_manager": "payment_manager",
            "error": "error",
            END: END,
        },
    )

    # after any manager finishes a turn, optionally synthesize TTS once
    g.add_conditional_edges(
        "ordering_manager",
        _maybe_tts,
        {"tts": "tts", END: END},
    )
    g.add_conditional_edges(
        "payment_manager",
        _maybe_tts,
        {"tts": "tts", END: END},
    )

    # error ends the turn
    g.add_edge("error", END)

    return g.compile()
