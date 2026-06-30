"""
Streamlit chat frontend for BlueBot. Calls the FastAPI backend's /chat endpoint.
Run: streamlit run streamlit_app.py
(Keep the FastAPI server running in a separate terminal: uvicorn src.api:app --reload)
"""

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/chat")

st.set_page_config(
    page_title="BlueBot — Airblue Policy Assistant",
    page_icon="https://www.airblue.com/Content/Layouts/Clean/images/airblue-site-ID.svg",
    layout="centered",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --ab-navy:      #0B2040;
        --ab-blue:      #1A4B8C;
        --ab-sky:       #2E7DD1;
        --ab-sky-light: #E8F1FB;
        --ab-white:     #FFFFFF;
        --ab-off:       #F4F7FC;
        --ab-border:    #D0DAEA;
        --ab-muted:     #6B7A90;
        --ab-success:   #0F7A55;
        --ab-success-bg:#E6F4EF;
        --ab-warn-bg:   #F1F3F7;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--ab-off) !important;
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }

    /* Page wrapper */
    .block-container {
        max-width: 760px !important;
        padding: 0 1rem 2rem 1rem !important;
    }

    /* ── Header ── */
    .ab-header {
        background: var(--ab-navy);
        margin: 0 -1rem 0 -1rem;
        padding: 18px 28px 16px 28px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 3px solid var(--ab-sky);
    }
    .ab-header-left { display: flex; align-items: center; gap: 14px; }
    .ab-logo-box {
        background: var(--ab-sky);
        color: var(--ab-white);
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 500;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        padding: 6px 12px;
        border-radius: 4px;
    }
    .ab-header-title {
        color: var(--ab-white);
        font-weight: 700;
        font-size: 1.05rem;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .ab-header-sub {
        color: #8BAFD4;
        font-size: 0.78rem;
        margin: 2px 0 0 0;
    }
    .ab-status-pill {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.05em;
        color: #7DD4A8;
        background: rgba(125,212,168,0.1);
        border: 1px solid rgba(125,212,168,0.25);
        padding: 4px 10px;
        border-radius: 100px;
        display: flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
    }
    .ab-status-dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: #7DD4A8;
        animation: ab-pulse 2.4s ease infinite;
    }
    @keyframes ab-pulse {
        0%   { opacity: 1; }
        50%  { opacity: 0.35; }
        100% { opacity: 1; }
    }

    /* ── Chat area ── */
    .ab-chat-area {
        padding: 20px 0 8px 0;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }

    /* ── Message rows ── */
    .ab-row {
        display: flex;
        align-items: flex-end;
        gap: 10px;
    }
    .ab-row.user  { flex-direction: row-reverse; }
    .ab-row.bot   { flex-direction: row; }

    /* Avatar */
    .ab-avatar {
        width: 32px; height: 32px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.04em;
        flex-shrink: 0;
        font-family: 'IBM Plex Mono', monospace;
    }
    .ab-avatar.user {
        background: var(--ab-blue);
        color: var(--ab-white);
    }
    .ab-avatar.bot {
        background: var(--ab-navy);
        color: var(--ab-sky);
        border: 1.5px solid var(--ab-sky);
    }

    /* Bubble */
    .ab-bubble {
        max-width: 78%;
        padding: 12px 16px;
        border-radius: 12px;
        font-size: 0.92rem;
        line-height: 1.55;
        color: var(--ab-navy);
    }
    .ab-bubble.user {
        background: var(--ab-blue);
        color: var(--ab-white);
        border-bottom-right-radius: 3px;
    }
    .ab-bubble.bot {
        background: var(--ab-white);
        border: 1px solid var(--ab-border);
        border-bottom-left-radius: 3px;
        box-shadow: 0 1px 3px rgba(11,32,64,0.06);
    }

    /* Badge */
    .ab-badge {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.63rem;
        letter-spacing: 0.06em;
        padding: 2px 7px;
        border-radius: 3px;
        margin-bottom: 7px;
        font-weight: 500;
    }
    .ab-badge.grounded {
        background: var(--ab-success-bg);
        color: var(--ab-success);
    }
    .ab-badge.ungrounded {
        background: var(--ab-warn-bg);
        color: var(--ab-muted);
    }

    /* Sources */
    .ab-sources {
        margin-top: 10px;
        padding-top: 9px;
        border-top: 1px solid var(--ab-border);
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
        align-items: center;
    }
    .ab-sources-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.63rem;
        color: var(--ab-muted);
        letter-spacing: 0.04em;
        margin-right: 2px;
    }
    .ab-source-tag {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.63rem;
        color: var(--ab-blue);
        background: var(--ab-sky-light);
        border: 1px solid #C2D8F2;
        border-radius: 3px;
        padding: 2px 7px;
        white-space: nowrap;
    }

    .ab-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.63rem;
        color: var(--ab-muted);
        margin-top: 6px;
    }

    /* ── Empty state ── */
    .ab-empty {
        background: var(--ab-white);
        border: 1px solid var(--ab-border);
        border-radius: 10px;
        padding: 20px 20px 14px 20px;
        margin-bottom: 8px;
    }
    .ab-empty-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--ab-muted);
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    /* Starter buttons */
    div[data-testid="stButton"] button {
        background: var(--ab-white) !important;
        color: var(--ab-navy) !important;
        border: 1px solid var(--ab-border) !important;
        border-radius: 6px !important;
        font-size: 0.83rem !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 400 !important;
        text-align: left !important;
        padding: 8px 12px !important;
        transition: border-color 0.15s, color 0.15s !important;
    }
    div[data-testid="stButton"] button:hover {
        border-color: var(--ab-sky) !important;
        color: var(--ab-sky) !important;
        background: var(--ab-sky-light) !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        border-top: 1px solid var(--ab-border) !important;
        padding-top: 12px !important;
    }
    [data-testid="stChatInput"] textarea {
        border: 1.5px solid var(--ab-border) !important;
        border-radius: 8px !important;
        color: var(--ab-navy) !important;
        background: var(--ab-white) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.92rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--ab-sky) !important;
        box-shadow: 0 0 0 2px rgba(46,125,209,0.15) !important;
    }

    /* Hide default st.chat_message chrome — we render our own bubbles */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    [data-testid="stChatMessageContent"] { padding: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="ab-header">
        <div class="ab-header-left">
            <div class="ab-logo-box">AB</div>
            <div>
                <p class="ab-header-title">BlueBot</p>
                <p class="ab-header-sub">Airblue Policy Assistant</p>
            </div>
        </div>
        <div class="ab-status-pill">
            <span class="ab-status-dot"></span>ONLINE
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


def _source_label(filename: str) -> str:
    """Convert raw filename to readable label."""
    return filename.replace("_", " ").replace(".txt", "").title()


def _render_message(msg: dict) -> None:
    role = msg["role"]
    content = msg["content"]

    if role == "user":
        st.markdown(
            f"""
            <div class="ab-row user">
                <div class="ab-avatar user">YOU</div>
                <div class="ab-bubble user">{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        is_relevant = msg.get("is_relevant", False)
        badge_class = "grounded" if is_relevant else "ungrounded"
        badge_text = "POLICY GROUNDED" if is_relevant else "OUTSIDE SCOPE"

        sources_html = ""
        if msg.get("sources"):
            tags = "".join(
                f'<span class="ab-source-tag">{_source_label(s)}</span>'
                for s in msg["sources"]
            )
            sources_html = f'<div class="ab-sources"><span class="ab-sources-label">SOURCES</span>{tags}</div>'

        meta_html = ""
        if msg.get("response_time"):
            meta_html = f'<div class="ab-meta">{msg["response_time"]:.1f}s</div>'

        bubble_inner = (
            f'<div class="ab-badge {badge_class}">{badge_text}</div>'
            f'<div>{content}</div>'
            f'{sources_html}'
            f'{meta_html}'
        )

        st.markdown(
            f'<div class="ab-row bot">'
            f'<div class="ab-avatar bot">AB</div>'
            f'<div class="ab-bubble bot">{bubble_inner}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _send(query: str) -> None:
    history = []
    msgs = st.session_state.messages
    if len(msgs) >= 2 and msgs[-2]["role"] == "user" and msgs[-1]["role"] == "assistant":
        history = [{
            "query": msgs[-2]["content"],
            "answer": msgs[-1]["content"],
            "is_relevant": msgs[-1].get("is_relevant", False),
        }]

    st.session_state.messages.append({"role": "user", "content": query})

    try:
        response = requests.post(
            API_URL,
            json={"query": query, "history": history},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        answer = data["answer"]
        sources = data.get("sources", [])
        is_relevant = data.get("is_relevant", False)
        response_time = data.get("response_time")
    except requests.RequestException as exc:
        answer = f"Unable to reach the BlueBot backend: {exc}"
        sources, is_relevant, response_time = [], False, None

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "is_relevant": is_relevant,
        "response_time": response_time,
    })


# ── Empty state ──────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ab-empty"><div class="ab-empty-label">Suggested questions</div>', unsafe_allow_html=True)
    starters = [
        "What is the baggage allowance for Value fare?",
        "How do I request a refund?",
        "What happens if my flight is cancelled?",
        "How does the BlueMiles program work?",
    ]
    cols = st.columns(2)
    for i, starter in enumerate(starters):
        if cols[i % 2].button(starter, key=f"starter_{i}", use_container_width=True):
            with st.spinner(""):
                _send(starter)
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ── Conversation ─────────────────────────────────────────────────────────────
if st.session_state.messages:
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        _render_message(msg)
        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

# ── Input ────────────────────────────────────────────────────────────────────
st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
query = st.chat_input("Ask about Airblue fares, baggage, check-in, or refunds...")
if query:
    # Render user message immediately before waiting for response
    _render_message({"role": "user", "content": query})
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner(""):
        msgs = st.session_state.messages
        history = []
        if len(msgs) >= 3 and msgs[-3]["role"] == "user" and msgs[-2]["role"] == "assistant":
            history = [{
                "query": msgs[-3]["content"],
                "answer": msgs[-2]["content"],
                "is_relevant": msgs[-2].get("is_relevant", False),
            }]
        try:
            response = requests.post(
                API_URL,
                json={"query": query, "history": history},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            answer = data["answer"]
            sources = data.get("sources", [])
            is_relevant = data.get("is_relevant", False)
            response_time = data.get("response_time")
        except requests.RequestException as exc:
            answer = f"Unable to reach the BlueBot backend: {exc}"
            sources, is_relevant, response_time = [], False, None

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "is_relevant": is_relevant,
            "response_time": response_time,
        })
    st.rerun()