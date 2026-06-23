"""
Streamlit chat frontend for BlueBot. Calls the FastAPI backend's /chat endpoint.
Run: streamlit run streamlit_app.py
(Keep the FastAPI server running in a separate terminal: uvicorn src.api:app --reload)
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/chat")

st.set_page_config(page_title="BlueBot — Airblue Assistant", page_icon="✈️", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

    :root {
        --ink: #0B2545;
        --accent: #2D6FE0;
        --accent-soft: #E8F0FE;
        --paper: #F6F8FC;
        --surface: #FFFFFF;
        --line: #DCE3F0;
        --muted: #5B6B82;
        --teal: #137A6F;
        --teal-soft: #E3F3F0;
    }

    .stApp { background-color: var(--paper); }

    [data-testid="stAppViewContainer"] * { font-family: 'Inter', sans-serif; }

    /* Header block */
    .bb-header { padding: 0.25rem 0 0.75rem 0; border-bottom: 1px solid var(--line); margin-bottom: 1.25rem; }
    .bb-title {
        font-family: 'Sora', sans-serif;
        font-weight: 800;
        font-size: 2.0rem;
        color: var(--ink);
        letter-spacing: -0.02em;
        margin: 0;
    }
    .bb-subtitle { color: var(--muted); font-size: 0.95rem; margin-top: 2px; }
    .bb-status {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        color: var(--teal);
        background: var(--teal-soft);
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 100px;
        margin-top: 10px;
    }
    .bb-status .dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: var(--teal);
        box-shadow: 0 0 0 0 rgba(19,122,111,0.5);
        animation: bb-pulse 2s infinite;
    }
    @keyframes bb-pulse {
        0% { box-shadow: 0 0 0 0 rgba(19,122,111,0.35); }
        70% { box-shadow: 0 0 0 5px rgba(19,122,111,0); }
        100% { box-shadow: 0 0 0 0 rgba(19,122,111,0); }
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background-color: var(--surface);
        border: 1px solid var(--line);
        border-left: 3px solid var(--accent);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(11,37,69,0.04);
    }
    [data-testid="stChatMessage"] p {
        color: var(--ink);
        font-size: 0.96rem;
        line-height: 1.5;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        border: 1.5px solid var(--line) !important;
        border-radius: 10px !important;
        color: var(--ink) !important;
        background: var(--surface) !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }

    /* Buttons (starter questions) */
    button[kind="secondary"] {
        background-color: var(--surface) !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        font-size: 0.85rem !important;
    }
    button[kind="secondary"]:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }

    /* Grounding badge */
    .bb-badge {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.03em;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .bb-badge.grounded { background: var(--teal-soft); color: var(--teal); }
    .bb-badge.ungrounded { background: #F1F2F6; color: var(--muted); }

    /* Source stubs - boarding-pass style */
    .bb-sources { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; }
    .bb-stub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        color: var(--ink);
        background: var(--accent-soft);
        border: 1px dashed #A9C2F2;
        border-radius: 4px;
        padding: 3px 8px 3px 10px;
        white-space: nowrap;
    }

    .bb-meta { color: var(--muted); font-size: 0.72rem; margin-top: 6px; font-family: 'IBM Plex Mono', monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----- Header -----
st.markdown(
    """
    <div class="bb-header">
        <p class="bb-title">✈️ BlueBot</p>
        <p class="bb-subtitle">Airblue's policy assistant — fares, baggage, check-in, refunds, and passenger rights.</p>
        <span class="bb-status"><span class="dot"></span>ONLINE · LOCAL RAG PIPELINE</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def _send(query: str) -> None:
    st.session_state.messages.append({"role": "user", "content": query})
    try:
        response = requests.post(API_URL, json={"query": query}, timeout=30)
        response.raise_for_status()
        data = response.json()
        answer = data["answer"]
        sources = data.get("sources", [])
        is_relevant = data.get("is_relevant", False)
        response_time = data.get("response_time")
    except requests.RequestException as exc:
        answer = f"Couldn't reach the BlueBot backend: {exc}"
        sources, is_relevant, response_time = [], False, None

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "is_relevant": is_relevant,
            "response_time": response_time,
        }
    )


# ----- Empty state: starter questions -----
if not st.session_state.messages:
    st.markdown("<p style='color: var(--muted); font-size: 0.85rem; margin-bottom: 6px;'>Try asking:</p>", unsafe_allow_html=True)
    starters = [
        "What's the baggage allowance for Value fare?",
        "How do I get a refund?",
        "What happens if my flight is cancelled?",
        "How does the BlueMiles program work?",
    ]
    cols = st.columns(2)
    for i, starter in enumerate(starters):
        if cols[i % 2].button(starter, key=f"starter_{i}", use_container_width=True):
            with st.spinner("Thinking..."):
                _send(starter)
            st.rerun()

# ----- Render conversation -----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧳" if msg["role"] == "user" else "✈️"):
        if msg["role"] == "assistant":
            if msg.get("is_relevant"):
                st.markdown('<span class="bb-badge grounded">GROUNDED IN POLICY DOCS</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="bb-badge ungrounded">OUTSIDE CURRENT SCOPE</span>', unsafe_allow_html=True)

        st.write(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            stubs = "".join(f'<span class="bb-stub">{s}</span>' for s in msg["sources"])
            st.markdown(f'<div class="bb-sources">{stubs}</div>', unsafe_allow_html=True)

        if msg["role"] == "assistant" and msg.get("response_time"):
            st.markdown(f'<p class="bb-meta">{msg["response_time"]:.1f}s</p>', unsafe_allow_html=True)

# ----- Chat input -----
query = st.chat_input("Ask a question about Airblue policies...")
if query:
    with st.spinner("Thinking..."):
        _send(query)
    st.rerun()