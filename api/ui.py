"""
ui.py — Streamlit frontend for the Recipe RAG Multi-Agent System.

Two modes (toggled in sidebar):
  • Classic   — POST /recommend,        blocking
  • Streaming — POST /recommend_stream, SSE, shows pipeline progress + live tokens

Stream event protocol from server:
  {"status": "<node display name>"}  — agent started
  {"token":  "<chunk>"}              — LLM token (responder only)
  {"answer": "<full answer>"}        — final answer
  {"error":  "<message>"}            — error
"""

import json
import time

import uuid
import requests
import streamlit as st

import os
API_URL = os.getenv("API_URL", "http://backend:8010")

st.set_page_config(
    page_title="🍳 Recipe RAG Agent",
    page_icon="🍳",
    layout="centered",
)

st.title("🍳 Recipe Recommendation Agent")
st.markdown(
    "Tell me what ingredients you have, your dietary preferences, "
    "or what kind of meal you're in the mood for."
)


# ── Session state ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    use_streaming = st.toggle(
        "⚡ Streaming mode",
        value=st.session_state.get("use_streaming", True),
        help="Show live pipeline progress and tokens as they arrive.",
    )
    st.session_state.use_streaming = use_streaming

    st.divider()
    st.markdown("**💡 Try asking:**")
    example_queries = [
        "I have chicken, garlic and soy sauce. Something Asian.",
        "High protein vegetarian dinner, no gluten.",
        "Quick breakfast with eggs and cheese.",
    ]
    for q in example_queries:
        if st.button(q, use_container_width=True, key=q):
            st.session_state.prefill = q
            st.rerun()

    st.divider()

    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "**Pipeline:**\n"
        "🥕 Ingredient Agent → 🍳 Recipe Agent → 💊 Nutrition Agent "
        "→ 🎬 YouTube Agent → 💬 Responder"
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "⚠️ AI-generated recommendations. Not medical or nutritional advice. "
        "Always consult a professional for dietary decisions. "
        "By using this system you consent to anonymous usage logging."
    )


# ── Chat history ───────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("duration"):
            st.caption(f"✅ {msg['duration']:.1f}s")


# ── API calls ──────────────────────────────────────────────────────────────────

def send_feedback(rating: int):
    """Send thumbs up/down rating to backend."""
    try:
        requests.post(
            f"{API_URL}/feedback",
            data={
                "query":     st.session_state.last_query[:300],
                "answer":    st.session_state.last_answer[:300],
                "rating":    rating,
                "thread_id": st.session_state.thread_id,
            },
            timeout=5,
        )
    except Exception:
        pass


def call_sync(query: str) -> str:
    resp = requests.post(
        f"{API_URL}/recommend",
        data={"query": query},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("answer", "No recommendations found.")


def call_stream(
    query: str,
    status_placeholder,
    answer_placeholder,
) -> str:
    answer_chunks: list[str] = []
    full_answer = ""

    with requests.post(
        f"{API_URL}/recommend_stream",
        data={"query": query},
        stream=True,
        timeout=120,
        headers={"Accept": "text/event-stream"},
    ) as resp:
        resp.raise_for_status()

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if not line.startswith("data:"):
                continue

            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue

            if "status" in event:
                status_placeholder.markdown(f"*⏳ {event['status']}…*")

            elif "token" in event:
                answer_chunks.append(event["token"])
                answer_placeholder.markdown("".join(answer_chunks) + "▌")

            elif "answer" in event:
                full_answer = event["answer"]
                status_placeholder.empty()
                answer_placeholder.markdown(full_answer)

            elif "error" in event:
                status_placeholder.empty()
                st.error(f"❌ Error: {event['error']}")
                break

    return full_answer if full_answer else "".join(answer_chunks)


# ── Chat input ────────────────────────────────────────────────────────────────

prefill = st.session_state.pop("prefill", None)

prompt = st.chat_input(
    "e.g. I have chicken, broccoli and soy sauce. High protein please.",
)

# Sidebar example button was clicked
if prefill and not prompt:
    prompt = prefill

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = ""
        start  = time.time()
        try:
            if use_streaming:
                status_placeholder = st.empty()
                answer_placeholder = st.empty()
                answer = call_stream(prompt, status_placeholder, answer_placeholder)
            else:
                with st.spinner("⏳ Finding the best recipes for you..."):
                    answer = call_sync(prompt)
                st.markdown(answer)

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the server. Is it running?")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

        duration = time.time() - start
        st.caption(f"✅ {duration:.1f}s")

    st.session_state.last_query  = prompt
    st.session_state.last_answer = answer
    st.session_state.messages.append({
        "role":     "assistant",
        "content":  answer,
        "duration": duration,
    })

# ── Feedback buttons (shown after last response) ───────────────────────────────
if st.session_state.last_answer:
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("👍", help="Good recommendation", key="thumbs_up"):
            send_feedback(1)
            st.toast("Thanks for the feedback!", icon="👍")
    with col2:
        if st.button("👎", help="Poor recommendation", key="thumbs_down"):
            send_feedback(-1)
            st.toast("Thanks! We'll improve.", icon="👎")