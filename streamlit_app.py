# streamlit_app.py (WS everywhere, auto-reconnect, single mic button, safe UI)
import io, json, time, base64, threading
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

try:
    import websocket  # pip install websocket-client
except Exception:
    websocket = None

WS_OPCODE_BINARY = 2
WS_CHAT  = "ws://localhost:8000/ws/chat"
WS_VOICE = "ws://localhost:8000/ws/voice"
SESSION_ID = "streamlit"

st.set_page_config(page_title="AI Waiter", layout="wide")

# ---------- chat look & feel ----------
st.markdown("""
<style>
.chatwrap {max-width: 880px; margin: 0 auto;}
.bubble {border-radius: 14px; padding: 10px 14px; margin: 6px 0; display:inline-block; max-width: 85%;}
.me     {background:#2b313e; color:#fff; align-self:flex-end;}
.ai     {background:#1f232b; color:#eaeaea;}
.row    {display:flex; margin:6px 0;}
.row.me {justify-content:flex-end;}
.composer {position: sticky; bottom: 0; background: #0e1117; padding: 10px; border-top:1px solid #333;}
.inputbar {display:flex; gap:8px;}
.inputbar input {flex:1}

/* Mic button like ChatGPT */
.mic-wrap { display:flex; align-items:center; gap:.75rem; justify-content:flex-end; }
.mic-btn {
  width:56px; height:56px; border-radius:50%;
  background:#ff4b4b; color:white; border:none;
  display:flex; align-items:center; justify-content:center;
  font-size:24px; cursor:pointer; user-select:none;
  box-shadow:0 6px 16px rgba(255,75,75,0.35);
}
.mic-btn[aria-pressed="true"] { background:#ff6d6d; }
.mic-hint { opacity:0.7; font-size:0.9rem; }
.badge {padding: 2px 8px; border-radius: 999px; font-size: 12px; border: 1px solid #333;}
.badge.ok { color:#5bdc5b; border-color:#234a23; }
.badge.err{ color:#ff7777; border-color:#4a2323; }
</style>
""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("chat", [])
ss.setdefault("cart", [])
ss.setdefault("total", 0)
ss.setdefault("menu", [])
ss.setdefault("talking", False)         # mic state
ss.setdefault("was_talking", False)     # edge detection
ss.setdefault("voice_audio_b64", None)  # server TTS audio mailbox
ss.setdefault("partial", "")            # partial ASR for display

# ---------- Robust WebSocket client with auto-reconnect ----------
class WSClient:
    def __init__(self, url, on_message=None, on_error=None):
        self.url = url
        self.on_message = on_message
        self.on_error = on_error
        self.ws = None
        self._open = False
        self._thread = None
        self._stop = False

    def _safe_cb(self, cb, *a):
        try:
            if cb: cb(*a)
        except Exception:
            pass

    def connect(self):
        if not websocket:
            return
        def _runner():
            backoff = 0.5
            while not self._stop:
                try:
                    self.ws = websocket.WebSocketApp(
                        self.url,
                        on_open=self._on_open,
                    
                        on_close=self._on_close,
                        on_error=lambda ws, err: self._safe_cb(self.on_error, ws, err),
                        on_message=lambda ws, msg: self._safe_cb(self.on_message, ws, msg),
                    )
                    self.ws.run_forever()
                except Exception as e:
                    self._safe_cb(self.on_error, None, e)
                finally:
                    self._open = False
                    sleep_t = backoff
                    backoff = min(8.0, backoff * 2)
                    t0 = time.time()
                    while not self._stop and time.time() - t0 < sleep_t:
                        time.sleep(0.1)
        if not self._thread or not self._thread.is_alive():
            self._thread = threading.Thread(target=_runner, daemon=True)
            self._thread.start()

    def close(self):
        self._stop = True
        try:
            if self.ws: self.ws.close()
        except Exception:
            pass

    def _on_open(self, ws):
        self._open = True
        try:
            ws.send(json.dumps({"session_id": SESSION_ID}))
        except Exception:
            pass

    def _on_close(self, ws, code, reason):
        self._open = False

    def is_open(self):
        w = self.ws
        return self._open and w and getattr(w, "sock", None) and w.sock.connected

    def send_json(self, payload):
        if not self.is_open(): return False
        try:
            self.ws.send(json.dumps(payload))
            return True
        except Exception:
            self._open = False
            return False

    def send_bytes(self, data, opcode=WS_OPCODE_BINARY):
        if not self.is_open(): return False
        try:
            self.ws.send(data, opcode=opcode)
            return True
        except Exception:
            self._open = False
            return False

# ---------- Chat WS ----------
def chat_on_message(ws, message):
    try:
        obj = json.loads(message)
    except Exception:
        return
    if obj.get("type") == "hello":
        ss["cart"]  = obj.get("state", {}).get("cart", [])
        ss["total"] = obj.get("state", {}).get("total", 0)
    elif obj.get("type") == "reply":
        ss["chat"].append(("ai", obj.get("text","")))
        ss["cart"]  = obj.get("state", {}).get("cart", [])
        ss["total"] = obj.get("state", {}).get("total", 0)

def chat_on_error(ws, err):
    ss["chat"].append(("ai", f"(chat ws error: {err})"))

if "chat_client" not in ss:
    ss["chat_client"] = WSClient(WS_CHAT, chat_on_message, chat_on_error)
    ss["chat_client"].connect()

# ---------- Voice WS ----------
def voice_on_message(ws, message):
    try:
        obj = json.loads(message)
    except Exception:
        return
    if obj.get("partial"):
        ss["partial"] = obj.get("text", "")
    elif obj.get("final"):
        reply = obj.get("reply", "")
        b64   = obj.get("audio_wav_b64")
        ss["chat"].append(("ai", reply))
        ss["cart"]  = obj.get("state", {}).get("cart", [])
        ss["total"] = obj.get("state", {}).get("total", 0)
        ss["voice_audio_b64"] = b64 or None
    elif "text" in obj:
        ss["chat"].append(("user", obj["text"]))

def voice_on_error(ws, err):
    ss["chat"].append(("ai", f"(voice ws error: {err})"))

if "voice_client" not in ss:
    ss["voice_client"] = WSClient(WS_VOICE, voice_on_message, voice_on_error)
    ss["voice_client"].connect()

# ---------- Layout ----------
top_left, top_right = st.columns([2, 1])
with top_right:
    st.markdown(
        f"""
        <span class="badge {'ok' if ss['chat_client'].is_open() else 'err'}">
          chat: {'connected' if ss['chat_client'].is_open() else 'offline'}
        </span>
        &nbsp;
        <span class="badge {'ok' if ss['voice_client'].is_open() else 'err'}">
          voice: {'connected' if ss['voice_client'].is_open() else 'offline'}
        </span>
        """,
        unsafe_allow_html=True
    )

left, right = st.columns([2, 1], gap="large")

with left:
    st.markdown('<div class="chatwrap">', unsafe_allow_html=True)

    for role, msg in ss["chat"]:
        who = "me" if role == "user" else "ai"
        st.markdown(f'<div class="row {who}"><div class="bubble {who}">{msg}</div></div>', unsafe_allow_html=True)

    # render any server TTS audio on the main thread
    if ss.get("voice_audio_b64"):
        try:
            raw = base64.b64decode(ss["voice_audio_b64"])
            st.audio(io.BytesIO(raw), format="audio/wav")
        finally:
            ss["voice_audio_b64"] = None

    # show partial ASR if any
    if ss.get("partial"):
        st.caption(f"🎤 {ss['partial']}")

    # Composer (text + mic)
    st.markdown('<div class="composer"><div class="inputbar">', unsafe_allow_html=True)
    text = st.text_input("Message", key="compose",
                         placeholder="Message the waiter…",
                         label_visibility="collapsed")
    send_col, mic_col = st.columns([1, .4])

    with send_col:
        if st.button("Send", use_container_width=True):
            if text.strip():
                ss["chat"].append(("user", text.strip()))
                ok = ss["chat_client"].send_json({"text": text.strip()})
                if not ok:
                    st.warning("Chat connection not ready")

    with mic_col:
        # Single mic button (toggle)
        if st.button("🎙️", help="Click to start speaking. Click again to stop.",
                     use_container_width=True, key="mic_btn"):
            ss["talking"] = not ss["talking"]
        st.markdown(
            f'<div class="mic-wrap"><button class="mic-btn" aria-pressed="{str(ss["talking"]).lower()}">🎙️</button>'
            f'<span class="mic-hint">{"Recording…" if ss["talking"] else "Click to speak"}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("</div></div>", unsafe_allow_html=True)

    # WebRTC microphone (streams frames only when ss["talking"] = True)
    def _audio_cb(frame: av.AudioFrame):
        # convert to 16k mono pcm16
        samples = frame.to_ndarray()
        if samples.ndim == 2:
            samples = samples.mean(axis=0)
        samples = samples.astype(np.float32)
        sr_in = frame.sample_rate
        if sr_in != 16000:
            dur = samples.shape[0] / sr_in
            t_in  = np.linspace(0, dur, samples.shape[0], endpoint=False)
            t_out = np.linspace(0, dur, int(dur * 16000), endpoint=False)
            samples = np.interp(t_out, t_in, samples).astype(np.float32)
        samples = np.clip(samples, -1.0, 1.0)
        pcm16 = (samples * 32767.0).astype(np.int16).tobytes()

        # stream while talking
        if ss["talking"] and ss["voice_client"].is_open():
            ss["voice_client"].send_bytes(pcm16)
        return frame

    webrtc_streamer(
        key="inline-mic",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"audio": True, "video": False},
        audio_receiver_size=256,
        video_receiver_size=0,
        audio_frame_callback=_audio_cb,
    )

    # Edge detection for UX hint
    if (not ss["talking"]) and ss["was_talking"]:
        st.toast("Sending voice…", icon="🎤")
    ss["was_talking"] = ss["talking"]

with right:
    st.subheader("🧾 Cart")
    if ss["cart"]:
        for it in ss["cart"]:
            n = it.get("name","?"); q = int(it.get("qty",1)); p = it.get("price",0)
            cols = st.columns([1.6, .5, .8, .8])
            cols[0].markdown(f"**{n}**")
            cols[1].markdown(f"x{q}")
            cols[2].markdown(f"₹{p}")
            if cols[3].button("🗑️", key=f"rm-{n}"):
                ss["chat"].append(("user", f"remove {n}"))
                ss["chat_client"].send_json({"text": f"remove {n}"})
    else:
        st.info("Cart is empty.")
    st.markdown(f"**Total: ₹{ss['total']}**")

    st.subheader("📜 Menu")
    if not ss["menu"]:
        try:
            from pathlib import Path
            p = Path(__file__).parent / "menu.json"
            if p.exists():
                ss["menu"] = json.loads(p.read_text())
        except Exception:
            ss["menu"] = []

    if ss["menu"]:
        cats = {}
        for item in ss["menu"]:
            cats.setdefault((item.get("category") or "Other").title(), []).append(item)
        sel = st.selectbox("Category", sorted(cats.keys()))
        for it in cats.get(sel, []):
            row = st.columns([1.7, .8, .8])
            row[0].markdown(f"**{it.get('name','?')}**")
            row[1].markdown(f"₹{it.get('price','-')}")
            with row[2]:
                c1, c2 = st.columns(2)
                if c1.button("+1", key=f"add-{it['name']}"):
                    ss["chat"].append(("user", f"add 1 {it['name']}"))
                    ss["chat_client"].send_json({"text": f"add 1 {it['name']}"})
                if c2.button("−1", key=f"sub-{it['name']}"):
                    ss["chat"].append(("user", f"remove 1 {it['name']}"))
                    ss["chat_client"].send_json({"text": f"remove 1 {it['name']}"})