"""
dashboard.py — Performance metrics dashboard for the Recipe RAG Agent.

Reads from GET /metrics endpoint which aggregates:
  - recipe_rag_audit.jsonl  → request stats
  - node_timings.jsonl      → per-node duration
  - llm_calls.jsonl         → token usage + TTFT
  - feedback.jsonl          → user ratings

Run:
    streamlit run dashboard.py --server.port 8504 --server.address 0.0.0.0
"""

import os
import time
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8010")

st.set_page_config(
    page_title="Recipe RAG — Metrics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Recipe RAG Agent — Performance Dashboard")

AUTO_REFRESH = st.sidebar.toggle("🔄 Auto-refresh (10s)", value=False)
if st.sidebar.button("Refresh now"):
    st.rerun()

st.sidebar.divider()
st.sidebar.caption("Reads live from agent log files via /metrics endpoint.")


# ── Fetch metrics ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def fetch_metrics() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Cannot reach API at {API_URL}/metrics — is the backend running? ({e})")
        return None


m = fetch_metrics()
if not m:
    st.stop()

req  = m["requests"]
llm  = m["llm"]
fb   = m["feedback"]
nodes = m["nodes"]


# ── Row 1: Request overview ────────────────────────────────────────────────────

st.subheader("Requests")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total requests",   req["total"])
c2.metric("Successful",       req["success"])
c3.metric("Errors",           req["error"],
          delta=f"-{req['error']}" if req["error"] else None,
          delta_color="inverse")
c4.metric("Avg duration",     f"{req['avg_duration']}s")
c5.metric("Success rate",
          f"{round(req['success'] / max(req['total'], 1) * 100, 1)}%")


# ── Row 2: LLM stats ───────────────────────────────────────────────────────────

st.divider()
st.subheader("LLM Usage")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("LLM calls",         llm["total_calls"])
c2.metric("Total tokens",      f"{llm['total_tokens']:,}")
c3.metric("Prompt tokens",     f"{llm['prompt_tokens']:,}")
c4.metric("Completion tokens", f"{llm['completion_tokens']:,}")
c5.metric("Avg TTFT",          f"{llm['avg_ttft_sec']}s")


# ── Row 3: User feedback ───────────────────────────────────────────────────────

st.divider()
st.subheader("User Feedback")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total ratings",  fb["total"])
c2.metric("👍 Thumbs up",   fb["thumbs_up"])
c3.metric("👎 Thumbs down", fb["thumbs_down"])
c4.metric("Satisfaction",   f"{fb['satisfaction']}%")

if m.get("recent_feedback"):
    st.markdown("**Recent feedback:**")
    for f in m["recent_feedback"]:
        icon  = "👍" if f["rating"] == 1 else "👎"
        query = f.get("query", "")[:80]
        ts    = f.get("timestamp", "")[:19].replace("T", " ")
        st.caption(f"{icon} `{ts}` — {query}")


# ── Row 4: Node timing breakdown ───────────────────────────────────────────────

st.divider()
st.subheader("Agent Node Timings (avg seconds)")

if nodes:
    NODE_LABELS = {
        "ingredient": "🥕 Ingredient Agent",
        "recipe":     "🍳 Recipe Agent",
        "nutrition":  "💊 Nutrition Agent",
        "youtube":    "🎬 YouTube Agent",
        "responder":  "💬 Responder",
    }
    cols = st.columns(len(nodes))
    for col, (node, avg) in zip(cols, sorted(nodes.items(), key=lambda x: x[1], reverse=True)):
        label = NODE_LABELS.get(node, node)
        col.metric(label, f"{avg}s")

    # Bar chart
    import pandas as pd
    df = pd.DataFrame([
        {"Node": NODE_LABELS.get(n, n), "Avg (s)": v}
        for n, v in sorted(nodes.items(), key=lambda x: x[1], reverse=True)
    ])
    st.bar_chart(df.set_index("Node"))
else:
    st.info("No node timing data yet — run some queries first.")




# ── Row 5: Cache + YouTube quota ──────────────────────────────────────────────

st.divider()
col_cache, col_quota = st.columns(2)

with col_cache:
    st.subheader("Query Cache")
    cache = m.get("cache", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Cached entries",  cache.get("size", 0))
    c2.metric("Hit rate",        f"{cache.get('hit_rate', 0)}%")
    c3.metric("Hits / Misses",   f"{cache.get('hits', 0)} / {cache.get('misses', 0)}")

with col_quota:
    st.subheader("YouTube API Quota")
    quota = m.get("youtube_quota", {})
    used  = quota.get("calls_used", 0)
    limit = quota.get("daily_limit", 100)
    pct   = quota.get("usage_pct", 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Used today",    used)
    c2.metric("Remaining",     quota.get("calls_remaining", limit))
    c3.metric("Usage",         f"{pct}%",
              delta=f"{'⚠️ High' if pct > 80 else 'OK'}",
              delta_color="inverse" if pct > 80 else "normal")
    st.progress(min(pct / 100, 1.0))

# ── Auto-refresh ───────────────────────────────────────────────────────────────

if AUTO_REFRESH:
    time.sleep(10)
    st.rerun()