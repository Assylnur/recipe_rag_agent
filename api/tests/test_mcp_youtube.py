"""
tests/test_mcp_youtube.py — MCP server and YouTube Agent tests.

Tests the YouTube MCP server directly (JSON-RPC) and the agent node behavior.

Run:
    pytest tests/test_mcp_youtube.py -v
"""

import json
import pytest
import httpx

import os
_MCP_BASE = os.getenv("YOUTUBE_MCP_URL", "http://host.docker.internal:8801/messages").replace("/messages", "")
MCP_URL   = _MCP_BASE + "/messages"
HEALTH_URL = _MCP_BASE + "/health"


def mcp_available() -> bool:
    try:
        r = httpx.post(MCP_URL, json={"jsonrpc":"2.0","id":0,"method":"tools/list","params":{}}, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


requires_mcp = pytest.mark.skipif(
    not mcp_available(),
    reason="YouTube MCP server not running — start with: python mcp_servers/youtube_mcp.py",
)


def call_tool(name: str, arguments: dict) -> list:
    """Call MCP tool and return parsed video list."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    r = httpx.post(MCP_URL, json=payload, timeout=15)
    r.raise_for_status()

    for line in r.text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            data = json.loads(line[5:].strip())
            content = data.get("result", {}).get("content", [])
            for block in content:
                if block.get("type") == "text":
                    return json.loads(block["text"])
    return []


@requires_mcp
class TestMCPServer:

    def test_health_endpoint(self):
        """MCP server is reachable — tools/list returns 200."""
        payload = {"jsonrpc": "2.0", "id": 0, "method": "tools/list", "params": {}}
        r = httpx.post(MCP_URL, json=payload, timeout=5)
        assert r.status_code == 200

    def test_tools_list(self):
        """MCP server must expose search_youtube tool."""
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        r = httpx.post(MCP_URL, json=payload, timeout=5)
        r.raise_for_status()
        for line in r.text.splitlines():
            if line.startswith("data:"):
                data = json.loads(line[5:].strip())
                tools = data.get("result", {}).get("tools", [])
                names = [t["name"] for t in tools]
                assert "search_youtube" in names
                return
        pytest.fail("No tools/list response received")

    def test_search_returns_videos(self):
        """search_youtube must return at least one video."""
        videos = call_tool("search_youtube", {"query": "chicken teriyaki recipe", "max_results": 3})
        if not videos:
            pytest.skip("YOUTUBE_API_KEY not set in MCP server — no results returned")
        assert len(videos) > 0

    def test_video_has_required_fields(self):
        """Each video must have title, url, channel."""
        videos = call_tool("search_youtube", {"query": "pasta recipe tutorial", "max_results": 1})
        if not videos:
            pytest.skip("YOUTUBE_API_KEY not set in MCP server — no results returned")
        video = videos[0]
        assert "title" in video and video["title"]
        assert "url" in video and "youtube.com" in video["url"]
        assert "channel" in video

    def test_max_results_respected(self):
        """max_results parameter must be respected."""
        videos = call_tool("search_youtube", {"query": "chicken recipe", "max_results": 2})
        assert len(videos) <= 2

    def test_url_format_valid(self):
        """YouTube URLs must be in correct format."""
        videos = call_tool("search_youtube", {"query": "beef stir fry recipe", "max_results": 1})
        if videos:
            url = videos[0]["url"]
            assert url.startswith("https://www.youtube.com/watch?v=")
            assert len(url) > len("https://www.youtube.com/watch?v=") + 5

    def test_empty_query_handled(self):
        """Empty query: should return empty list or minimal results, not crash."""
        try:
            videos = call_tool("search_youtube", {"query": "", "max_results": 1})
            assert isinstance(videos, list)
        except Exception as e:
            assert "500" not in str(e)

    def test_nonsense_query_handled(self):
        """Nonsense query: should return empty list, not crash."""
        videos = call_tool("search_youtube", {"query": "zxcvbn qwerty asdfgh 99999", "max_results": 1})
        assert isinstance(videos, list)

    def test_max_results_capped_at_5(self):
        """max_results above 5 should be capped."""
        videos = call_tool("search_youtube", {"query": "chicken recipe", "max_results": 10})
        assert len(videos) <= 5