"""
nodes/youtube.py — YouTube Agent node.

Calls the YouTube MCP server to fetch top cooking tutorial videos
for each of the final_recipes recommended by Nutrition Agent.

MCP call flow:
  1. For each recipe in final_recipes → build search query "{name} recipe"
  2. Call YouTube MCP server tool: search_youtube(query, max_results=1)
  3. Attach video links to each recipe dict
  4. Also store flat video_links list in state for responder

MCP server must be running:
  python mcp_servers/youtube_mcp.py   # http://localhost:8001
"""

from __future__ import annotations

import json
import os
import asyncio
import logging

import httpx

from node_logger import profile_node
from state import RecipeAgentState

YOUTUBE_MCP_URL = os.getenv("YOUTUBE_MCP_URL", "http://localhost:8001/messages")

log = logging.getLogger(__name__)


# ── MCP client ─────────────────────────────────────────────────────────────────

async def _call_mcp_tool(tool_name: str, arguments: dict) -> list[dict]:
    """
    Call a tool on the YouTube MCP server via JSON-RPC over HTTP POST.
    Returns parsed list of video dicts, or [] on any error.
    """
    payload = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "tools/call",
        "params":  {"name": tool_name, "arguments": arguments},
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(YOUTUBE_MCP_URL, json=payload)
            resp.raise_for_status()

        # SSE response: parse "data: {...}" lines
        videos = []
        for line in resp.text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                data = json.loads(line[5:].strip())
                content = data.get("result", {}).get("content", [])
                for block in content:
                    if block.get("type") == "text":
                        videos = json.loads(block["text"])
                        break
        return videos

    except Exception as e:
        log.warning("[youtube_agent] MCP call failed: %s", e)
        return []


# ── Node ───────────────────────────────────────────────────────────────────────

@profile_node
async def youtube_node(state: RecipeAgentState) -> dict:
    """
    YouTube Agent — ONE API call per user query, returns top 3 videos.

    Builds a combined search query from all recipe names and searches once.
    This costs 1 quota unit instead of N (one per recipe).
    """
    final_recipes = state.get("final_recipes", [])
    if not final_recipes:
        return {"final_recipes": [], "video_links": []}

    # Build one combined query from top 3 recipe names
    names = [
        r.get("meta", {}).get("name", "")
        for r in final_recipes[:3]
        if r.get("meta", {}).get("name")
    ]

    # Fall back to corpus YouTube links if no names found
    if not names:
        video_links = _corpus_fallback(final_recipes)
        return {"final_recipes": final_recipes, "video_links": video_links}

    # Single API call — fetch top 3 videos for the first/best recipe
    # (most relevant — user can explore others via source links)
    query  = f"{names[0]} recipe cooking tutorial"
    videos = await _call_mcp_tool("search_youtube", {"query": query, "max_results": 3})

    # Attach first video to first recipe, distribute rest
    enriched    = []
    video_links = []
    for i, recipe in enumerate(final_recipes):
        video = videos[i] if i < len(videos) else None
        enriched.append({**recipe, "videos": [video] if video else []})
        if video:
            video_links.append({
                "recipe":  recipe.get("meta", {}).get("name", ""),
                "title":   video["title"],
                "url":     video["url"],
                "channel": video["channel"],
            })

    # For remaining recipes with no video — use TheMealDB corpus link as fallback
    from nodes.responder import _load_corpus
    corpus = _load_corpus()
    for i, recipe in enumerate(enriched):
        if not recipe.get("videos"):
            meal_id  = recipe.get("meta", {}).get("meal_id", "")
            yt_link  = corpus.get(meal_id, {}).get("youtube", "")
            name     = recipe.get("meta", {}).get("name", "")
            if yt_link and name:
                video_links.append({
                    "recipe":  name,
                    "title":   f"{name} — TheMealDB",
                    "url":     yt_link,
                    "channel": "TheMealDB",
                })

    log.info(
        "[youtube_agent] 1 API call → %d videos | %d corpus fallbacks",
        min(len(videos), len(final_recipes)),
        max(0, len(final_recipes) - len(videos)),
    )

    return {
        "final_recipes": enriched,
        "video_links":   video_links,
    }


def _corpus_fallback(final_recipes: list[dict]) -> list[dict]:
    """Use TheMealDB YouTube links from corpus when API is unavailable."""
    from nodes.responder import _load_corpus
    corpus      = _load_corpus()
    video_links = []
    for recipe in final_recipes:
        meal_id  = recipe.get("meta", {}).get("meal_id", "")
        yt_link  = corpus.get(meal_id, {}).get("youtube", "")
        name     = recipe.get("meta", {}).get("name", "")
        if yt_link and name:
            video_links.append({
                "recipe":  name,
                "title":   f"{name} — TheMealDB",
                "url":     yt_link,
                "channel": "TheMealDB",
            })
    return video_links