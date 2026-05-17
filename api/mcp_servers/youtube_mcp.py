"""
mcp_servers/youtube_mcp.py — YouTube Data API MCP server.

Exposes one tool: search_youtube(query, max_results)
Returns top video results with title, url, channel, thumbnail.

MCP transport: SSE (Server-Sent Events) — compatible with LangChain MCP client.

Get a free YouTube Data API v3 key:
  https://console.cloud.google.com → Enable "YouTube Data API v3" → Create credentials

Usage:
    YOUTUBE_API_KEY=your_key python mcp_servers/youtube_mcp.py
    # Runs on http://localhost:8001/sse
"""

import json
import os
import asyncio
import logging
from typing import Any

import httpx
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cache import quota_tracker
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
print(f"Starting YouTube MCP Server | API key set: {bool(YOUTUBE_API_KEY)}")
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="YouTube MCP Server")


# ── Tool definition ────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "search_youtube",
        "description": (
            "Search YouTube for cooking tutorial videos. "
            "Returns top videos with title, url, channel, and thumbnail."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, e.g. 'Teriyaki Chicken recipe tutorial'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 3, max: 5)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    }
]


# ── YouTube API call ───────────────────────────────────────────────────────────

async def search_youtube(query: str, max_results: int = 3) -> list[dict]:
    """Call YouTube Data API v3 search endpoint."""
    if not YOUTUBE_API_KEY:
        log.warning("YOUTUBE_API_KEY not set — returning empty results")
        return []

    params = {
        "part":       "snippet",
        "q":          f"{query} recipe cooking tutorial",
        "type":       "video",
        "maxResults": min(max_results, 5),
        "key":        YOUTUBE_API_KEY,
        "relevanceLanguage": "en",
        "videoCategoryId":   "26",  # Howto & Style
    }

    # Check daily quota before calling YouTube API
    allowed, remaining = quota_tracker.can_call()
    if not allowed:
        log.error("YouTube daily quota exhausted — returning empty results")
        return []

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(YOUTUBE_SEARCH_URL, params=params)
        resp.raise_for_status()

    quota_tracker.record_call()
    log.info("YouTube quota: %d/%d used today", 
             quota_tracker.status["calls_used"], quota_tracker.daily_limit)

    items = resp.json().get("items", [])
    results = []
    for item in items:
        video_id = item["id"].get("videoId", "")
        snippet  = item.get("snippet", {})
        results.append({
            "title":     snippet.get("title", ""),
            "url":       f"https://www.youtube.com/watch?v={video_id}",
            "channel":   snippet.get("channelTitle", ""),
            "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
        })

    log.info("YouTube search '%s' → %d results", query, len(results))
    return results


# ── MCP SSE protocol ───────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _handle_message(message: dict) -> dict | None:
    method = message.get("method")
    msg_id = message.get("id")

    # tools/list — return tool definitions
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id":      msg_id,
            "result":  {"tools": TOOLS},
        }

    # tools/call — execute a tool
    if method == "tools/call":
        params    = message.get("params", {})
        tool_name = params.get("name")
        args      = params.get("arguments", {})

        if tool_name == "search_youtube":
            try:
                videos = await search_youtube(
                    query=args.get("query", ""),
                    max_results=args.get("max_results", 3),
                )
                return {
                    "jsonrpc": "2.0",
                    "id":      msg_id,
                    "result":  {
                        "content": [
                            {"type": "text", "text": json.dumps(videos, ensure_ascii=False)}
                        ]
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id":      msg_id,
                    "error":   {"code": -32000, "message": str(e)},
                }

    # initialize handshake
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id":      msg_id,
            "result":  {
                "protocolVersion": "2024-11-05",
                "capabilities":    {"tools": {}},
                "serverInfo":      {"name": "youtube-mcp", "version": "1.0.0"},
            },
        }

    return None


@app.get("/sse")
async def sse_endpoint(request: Request):
    """SSE endpoint — MCP client connects here."""

    async def event_stream():
        # Send server capabilities on connect
        yield _sse({
            "jsonrpc": "2.0",
            "method":  "notifications/initialized",
        })

        # Keep connection alive and process incoming messages
        # In production, messages come via POST /messages; here we wait for disconnect
        while True:
            if await request.is_disconnected():
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/messages")
async def messages_endpoint(request: Request):
    """Receive JSON-RPC messages from MCP client."""
    body    = await request.json()
    response = await _handle_message(body)

    async def event_stream():
        if response:
            yield _sse(response)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {
        "status":      "ok",
        "api_key_set": bool(YOUTUBE_API_KEY),
        "quota":       quota_tracker.status,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)