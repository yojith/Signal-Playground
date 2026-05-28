import io
import uuid

import numpy as np
import streamlit as st

from signal_module import Signal
from graph import SignalGraph
from llm import process_prompt
from state_manager import save_state, load_state, init_db, cleanup_old_sessions


@st.cache_data(show_spinner=False)
def cached_audio_bytes(data_bytes: bytes, sr: int) -> bytes:
    signal = Signal(np.frombuffer(data_bytes, dtype=np.float64), sr)
    return signal.to_audio_bytes().getvalue()

# Set page name and favicon
st.set_page_config(page_title="Signal Playground", page_icon="assets/favicon.png")  # Podcast icons created by Risman Muhammad - Flaticon

# Initialize database and clean up old sessions on app start
init_db()
cleanup_old_sessions()

# Save session ID in query params for persistence across reloads
session_id = st.query_params.get("session_id")

if session_id is None:
    session_id = str(uuid.uuid4())
    st.query_params["session_id"] = session_id

st.session_state.session_id = session_id

# Load session data if exists, otherwise initialize
if "signal" not in st.session_state or "messages" not in st.session_state:
    sig, msgs = load_state(st.session_state.session_id)  # Load previous state if exists
    st.session_state.signal = sig
    st.session_state.messages = msgs

st.set_page_config(layout="wide")
st.title("LLM Signal Playground")
left, right = st.columns([1, 1])  # Layout

# --------------------------------------------------
# Left panel (chat)
# --------------------------------------------------
with left:
    uploaded = st.file_uploader("Upload WAV", type=["wav"], max_upload_size=100)
    if uploaded:
        st.session_state.signal = Signal.from_wav(uploaded)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Modify the signal...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        if st.session_state.signal is not None:
            response = process_prompt(prompt, st.session_state.signal, st.session_state.messages)
            if response.analysis is not None:
                # Analysis tools preserve the current signal and return a text result
                st.session_state.messages.append({"role": "assistant", "content": response.message})
            else:
                st.session_state.signal = response.signal
                st.session_state.messages.append({"role": "assistant", "content": response.message})
            st.rerun()

# --------------------------------------------------
# RIGHT PANEL (visualization)
# --------------------------------------------------
with right:
    if st.session_state.signal is not None:
        sig = st.session_state.signal
        # Prepare audio bytes once and reuse for playback and download
        audio_bytes = cached_audio_bytes(sig.data.tobytes(), sig.sr)

        st.audio(audio_bytes, format="audio/wav")
        # Download current signal as WAV
        st.download_button(
            label="Download WAV",
            data=audio_bytes,
            file_name=f"signal_{st.session_state.session_id}.wav",
            mime="audio/wav",
        )

        graph = SignalGraph(sig)
        graph.plot_time_domain(st)
        graph.plot_spectrogram(st)
        graph.plot_frequency_domain(st)
        st.info("Upload a WAV file to begin")

save_state(st.session_state.session_id, st.session_state.signal, st.session_state.messages)
