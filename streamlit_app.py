"""
Streamlit frontend for BlueBot.
Run:
    streamlit run streamlit_app.py

Make sure the FastAPI backend is running:
    uvicorn src.api:app --reload
"""

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/chat")

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="BlueBot - Airblue Assistant",
    page_icon="✈️",
    layout="centered",
)

# ----------------------------------------------------
# Custom CSS
# ----------------------------------------------------

st.markdown(
    """
<style>
.stApp {
    background-color: #f5f7fb;
}

.main > div {
    padding-top: 1rem;
}

.hero {
    text-align: center;
    margin-bottom: 1.5rem;
}

.hero h1 {
    color: #003580;
    font-size: 3rem;
    margin-bottom: 0rem;
}

.hero h3 {
    color: #555555;
    font-weight: 500;
    margin-top: 0.25rem;
}

.hero p {
    color: #777777;
    font-size: 1rem;
}

[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 0.8rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    margin-bottom: 0.75rem;
}

[data-testid="stChatInput"] textarea {
    border: 2px solid #003580 !important;
    border-radius: 12px !important;
}

button[kind="primary"] {
    border-radius: 10px !important;
}

.small-note {
    color: #888888;
    text-align: center;
    font-size: 0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

with st.sidebar:
    st.title("✈️ BlueBot")

    st.markdown("### I can help with")

    st.markdown("""
- 🧳 Baggage allowance
- 🛫 Check-in information
- 💺 Seat selection
- 💳 Refund policies
- 🎫 Fare rules
- 📋 Travel policies
""")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------------------
# Hero
# ----------------------------------------------------

st.markdown(
    """
<div class="hero">
    <h1>✈️ BlueBot</h1>
    <h3>Your Airblue Travel Assistant</h3>
    <p>
        Get instant answers about baggage, fares,
        check-in, refunds, and travel policies.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# Session State
# ----------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------------------------
# Welcome Card
# ----------------------------------------------------

if len(st.session_state.messages) == 0:
    st.info(
        """
👋 **Welcome to BlueBot!**

You can ask questions like:

- What is the baggage allowance?
- When does online check-in open?
- Can I change my ticket?
- What is Airblue's refund policy?
"""
    )

# ----------------------------------------------------
# Display Chat History
# ----------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):
            with st.expander("Sources"):
                for src in message["sources"]:
                    st.markdown(f"- {src}")

# ----------------------------------------------------
# Chat Input
# ----------------------------------------------------

prompt = st.chat_input(
    "Ask about Airblue baggage, fares, check-in, or travel policies..."
)

if prompt:

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):

        with st.spinner("Looking up Airblue policies..."):

            try:
                response = requests.post(
                    API_URL,
                    json={"query": prompt},
                    timeout=30,
                )

                response.raise_for_status()

                data = response.json()

                answer = data.get(
                    "answer",
                    "No response returned.",
                )

                sources = data.get(
                    "sources",
                    [],
                )

            except requests.RequestException as exc:

                answer = (
                    "⚠️ Unable to reach the BlueBot backend.\n\n"
                    f"Error: {exc}"
                )

                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.markdown(f"- {src}")

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )