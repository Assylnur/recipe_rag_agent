"""
server.py — FastAPI server for the Recipe RAG multi-agent system.

Endpoints:
  POST /recommend        — blocking, returns full recommendation
  POST /recommend_stream — SSE streaming with node progress + tokens
  GET  /health           — liveness check

Stream event protocol (same as Legal RAG for UI compatibility):
  {"status": "<node display name>"}   — node started
  {"token":  "<text chunk>"}          — LLM token (responder only)
  {"answer": "<full answer>"}         — final answer
  {"error":  "<message>"}             — error
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from graph import get_app
from security import RateLimiter, filter_request, mask_api_key
from cache import query_cache, quota_tracker
from logger import log_transaction

NODE_DISPLAY_NAMES = {
    "ingredient_agent": "🥕 Parsing ingredients and preferences...",
    "recipe_agent":     "🍳 Searching recipe database...",
    "nutrition_agent":  "💊 Evaluating nutritional fit...",
    "youtube_agent":    "🎬 Fetching YouTube tutorials...",
    "responder":        "💬 Generating recommendations...",
}
TRACKED_NODES = set(NODE_DISPLAY_NAMES.keys())

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

rate_limiter = RateLimiter(max_requests=20, window=60)  # 20 req/min per IP


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = await get_app()
    import os
    yt_key = os.getenv("YOUTUBE_API_KEY", "")
    logger.info("Recipe RAG agent ready | YouTube API key: %s", mask_api_key(yt_key))
    yield
    logger.info("Shutting down")


app = FastAPI(title="Recipe RAG Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class RecommendResponse(BaseModel):
    answer: str


def _initial_state(query: str, streaming: bool) -> dict:
    return {
        "user_question":    query,
        "messages":         [HumanMessage(content=query)],
        "streaming":        streaming,
        "candidate_recipes": [],
        "final_recipes":    [],
        "video_links":      [],
    }


# ── POST /recommend ────────────────────────────────────────────────────────────

@app.post("/recommend", response_model=RecommendResponse)
async def recommend_endpoint(request: Request, query: str = Form(...)):
    # Rate limiting
    client_ip = request.client.host or "unknown"
    allowed, remaining = rate_limiter.is_allowed(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in 60 seconds.")

    # Input validation + content filtering
    query, blocked, reason = filter_request(query)
    if blocked:
        raise HTTPException(status_code=400, detail=f"Request blocked: {reason}")

    try:
        start = time.time()
        result = await app.state.graph.ainvoke(_initial_state(query, streaming=False))
        log_transaction("sync", result, time.time() - start)
        return RecommendResponse(
            answer=result.get("final_answer", "Рекомендации не найдены.")
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /recommend_stream ─────────────────────────────────────────────────────

@app.post("/recommend_stream")
async def recommend_stream_endpoint(request: Request, query: str = Form(...)):
    # Rate limiting
    client_ip = request.client.host or "unknown"
    allowed, remaining = rate_limiter.is_allowed(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in 60 seconds.")

    # Input validation + content filtering
    query, blocked, reason = filter_request(query)
    if blocked:
        raise HTTPException(status_code=400, detail=f"Request blocked: {reason}")

    async def event_generator():
        start = time.time()
        final_state: dict = {}
        answer_sent = False

        async for event in app.state.graph.astream_events(
            _initial_state(query, streaming=True), version="v2"
        ):
            kind      = event["event"]
            node_name = event["metadata"].get("langgraph_node", "")

            if kind == "on_chain_start" and node_name in TRACKED_NODES:
                display = NODE_DISPLAY_NAMES[node_name]
                yield f"data: {json.dumps({'status': display})}\n\n"

            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                token = chunk.additional_kwargs.get("reasoning_content") or chunk.content
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"

            elif kind == "on_chain_end" and node_name in TRACKED_NODES:
                node_output = event["data"].get("output") or {}
                if isinstance(node_output, dict):
                    final_state.update(node_output)
                if not answer_sent and final_state.get("final_answer"):
                    answer_sent = True
                    yield f"data: {json.dumps({'answer': final_state['final_answer']})}\n\n"

        log_transaction("stream", final_state, time.time() - start)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── GET /health ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}




@app.post("/feedback")
async def feedback_endpoint(
    query:     str = Form(...),
    answer:    str = Form(...),
    rating:    int = Form(...),   # 1 = thumbs up, -1 = thumbs down
    thread_id: str = Form("anonymous"),
):
    """Store user rating for a response. Appends to feedback.jsonl."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "rating":    rating,
        "query":     query[:300],
        "answer":    answer[:300],
    }
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / "feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Feedback received: rating=%d thread=%s", rating, thread_id)
    return {"status": "ok"}


def _read_quota_file() -> dict:
    """Read quota file directly from disk — works regardless of singleton state."""
    import os
    paths_to_try = [
        "logs/youtube_quota.json",
        "/workspace/logs/youtube_quota.json",
        os.path.join(os.path.dirname(__file__), "logs", "youtube_quota.json"),
    ]
    for p in paths_to_try:
        try:
            from pathlib import Path
            f = Path(p)
            if f.exists():
                data = json.loads(f.read_text())
                calls = data.get("calls", 0)
                limit = 100
                return {
                    "date":            data.get("date"),
                    "calls_used":      calls,
                    "calls_remaining": max(limit - calls, 0),
                    "daily_limit":     limit,
                    "usage_pct":       round(calls / limit * 100, 1),
                    "exhausted":       calls >= limit,
                    "source_file":     p,
                }
        except Exception:
            continue
    return {"calls_used": 0, "calls_remaining": 100, "daily_limit": 100,
            "usage_pct": 0, "exhausted": False, "date": None, "source_file": None}


@app.get("/metrics")
async def metrics_endpoint():
    """Return aggregated metrics from log files for the dashboard."""
    import json
    from pathlib import Path
    from collections import defaultdict

    def read_jsonl(path: Path) -> list:
        if not path.exists():
            return []
        lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                lines.append(json.loads(line))
            except Exception:
                pass
        return lines

    logs_dir = Path("logs")
    audits   = read_jsonl(logs_dir / "recipe_rag_audit.jsonl")
    timings  = [r for r in read_jsonl(logs_dir / "node_timings.jsonl")
                if r.get("log_type") == "node_timing"]
    llm_logs = read_jsonl(logs_dir / "llm_calls.jsonl")
    feedback = read_jsonl(logs_dir / "feedback.jsonl")

    # ── Request stats ──────────────────────────────────────────────────────────
    total_requests = len(audits)
    success        = sum(1 for a in audits if a.get("status") == "SUCCESS")
    error          = total_requests - success
    avg_duration   = round(
        sum(a.get("duration_sec", 0) for a in audits) / max(total_requests, 1), 2
    )

    # ── Node timings ───────────────────────────────────────────────────────────
    node_stats = defaultdict(list)
    for t in timings:
        node_stats[t["node"]].append(t["duration_sec"])
    node_avg = {
        node: round(sum(times) / len(times), 3)
        for node, times in node_stats.items()
    }

    # ── LLM stats ──────────────────────────────────────────────────────────────
    total_tokens  = sum(r.get("total_tokens", 0) for r in llm_logs)
    total_prompt  = sum(r.get("prompt_tokens", 0) for r in llm_logs)
    total_completion = sum(r.get("completion_tokens", 0) for r in llm_logs)
    avg_ttft = round(
        sum(r.get("ttft_sec", 0) or 0 for r in llm_logs) / max(len(llm_logs), 1), 3
    )

    # ── Feedback stats ─────────────────────────────────────────────────────────
    thumbs_up   = sum(1 for f in feedback if f.get("rating") == 1)
    thumbs_down = sum(1 for f in feedback if f.get("rating") == -1)
    satisfaction = round(
        thumbs_up / max(thumbs_up + thumbs_down, 1) * 100, 1
    )

    return {
        "requests": {
            "total":        total_requests,
            "success":      success,
            "error":        error,
            "avg_duration": avg_duration,
        },
        "nodes": node_avg,
        "llm": {
            "total_calls":       len(llm_logs),
            "total_tokens":      total_tokens,
            "prompt_tokens":     total_prompt,
            "completion_tokens": total_completion,
            "avg_ttft_sec":      avg_ttft,
        },
        "feedback": {
            "total":        len(feedback),
            "thumbs_up":    thumbs_up,
            "thumbs_down":  thumbs_down,
            "satisfaction": satisfaction,
        },
        "recent_feedback": feedback[-5:][::-1],
        "cache":           query_cache.stats,
        "youtube_quota":   _read_quota_file(),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)