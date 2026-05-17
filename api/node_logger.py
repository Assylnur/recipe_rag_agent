"""
node_logger.py — per-node duration logger for the Legal RAG agent.

Writes two files to logs/:
  node_timings.jsonl   — one JSON line per node execution (append-only, queryable in Loki)
  node_timings.log     — human-readable, same data

Drop-in replacement for @profile_node in utils.py:
  Replace:  from utils import profile_node
  With:     from node_logger import profile_node

Or keep both — this one writes to disk, profile_node just prints.

Each JSONL line:
{
  "timestamp": "2026-03-16T10:23:01.123Z",
  "log_type":  "node_timing",
  "thread_id": "abc-123",          <- from LangGraph config (optional)
  "node":      "reranker",
  "status":    "success" | "error",
  "duration_sec": 4.21,
  "error":     null | "exception message"
}

Per-run summary (printed + appended to node_timings.jsonl at run end):
{
  "log_type":      "run_summary",
  "thread_id":     "abc-123",
  "total_sec":     18.4,
  "nodes": {
    "router":        {"duration_sec": 1.2,  "status": "success"},
    "node_x":{"duration_sec": 2.1,  "status": "success"},
    ...
  },
  "slowest_node":  "responder"
}
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler
# ---------------------------------------------------------------------------
# File setup
# ---------------------------------------------------------------------------

LOG_DIR          = os.getenv("LOG_DIR", "logs")
TIMING_JSONL     = os.path.join(LOG_DIR, "node_timings.jsonl")
TIMING_READABLE  = os.path.join(LOG_DIR, "node_timings.log")
os.makedirs(LOG_DIR, exist_ok=True)

# JSONL logger (machine-readable, one JSON per line — Loki-ready)
# MUST have a StreamHandler so lines appear in Docker stdout for Alloy to scrape.
_jl = logging.getLogger("NodeTimingJSON")
_jl.setLevel(logging.INFO)
_jl.propagate = False
# _jl_fh = logging.FileHandler(TIMING_JSONL, encoding="utf-8")
_jl_fh = RotatingFileHandler(TIMING_JSONL, maxBytes=1024*1024, backupCount=5, encoding="utf-8")  # 1MB max file size

_jl_fh.setFormatter(logging.Formatter("%(message)s"))
_jl_sh = logging.StreamHandler()          # ← stdout so Alloy can scrape it
_jl_sh.setFormatter(logging.Formatter("%(message)s"))
_jl.addHandler(_jl_fh)
_jl.addHandler(_jl_sh)

# Human-readable logger (stdout + .log file)
_hl = logging.getLogger("NodeTimingHuman")
_hl.setLevel(logging.INFO)
_hl.propagate = False
# _hl_fh = logging.FileHandler(TIMING_READABLE, encoding="utf-8")
_hl_fh = RotatingFileHandler(TIMING_READABLE, maxBytes=1024*1024, backupCount=5, encoding="utf-8")  # 1MB max file size

_hl_fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
_hl_sh = logging.StreamHandler()
_hl_sh.setFormatter(logging.Formatter("%(message)s"))
_hl.addHandler(_hl_fh)
_hl.addHandler(_hl_sh)

# ---------------------------------------------------------------------------
# Internal per-thread run accumulator
# Keyed by thread_id so concurrent requests don't mix timings.
# ---------------------------------------------------------------------------

_run_timings: Dict[str, Dict[str, Any]] = {}
_run_start:   Dict[str, float]          = {}


def _get_thread_id(args, kwargs) -> str:
    """
    Extract thread_id for logging. Priority:
      1. state["thread_id"]  — set in initial_state by server.py (most reliable)
      2. config["configurable"]["thread_id"] — only present on nodes that
         declare `config: RunnableConfig` in their signature
      3. "default" fallback
    """
    # 1. State is always the first positional arg in LangGraph nodes
    state = args[0] if args and isinstance(args[0], dict) else {}
    tid = state.get("thread_id")
    if tid:
        return str(tid)

    # 2. Config kwarg (only on nodes that declare it)
    config = kwargs.get("config")
    if not config and len(args) > 1 and isinstance(args[1], dict):
        config = args[1]
    if config and isinstance(config, dict):
        tid = config.get("configurable", {}).get("thread_id")
        if tid:
            return str(tid)

    return "default"


# ---------------------------------------------------------------------------
# Core emit
# ---------------------------------------------------------------------------

def _emit(
    node: str,
    status: str,
    duration_sec: float,
    thread_id: str = "default",
    error: Optional[str] = None,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()

    record = {
        "timestamp":    ts,
        "log_type":     "node_timing",
        "thread_id":    thread_id,
        "node":         node,
        "status":       status,
        "duration_sec": round(duration_sec, 3),
        "error":        error,
    }
    _jl.info(json.dumps(record, ensure_ascii=False))

    icon = "✅" if status == "success" else "❌"
    _hl.info(f"{icon}  [{thread_id[:8]}]  {node:25s}  {duration_sec:6.2f}s  {status}{f'  ERR: {error}' if error else ''}")

    # Accumulate per-run
    if thread_id not in _run_timings:
        _run_timings[thread_id] = {}
        _run_start[thread_id]   = time.perf_counter() - duration_sec  # back-fill start

    _run_timings[thread_id][node] = {
        "duration_sec": round(duration_sec, 3),
        "status":       status,
        "error":        error,
    }


def emit_run_summary(thread_id: str = "default") -> None:
    """
    Call this after a full graph run to write a summary line.
    server.py calls this at the end of /chat and /chat_stream.
    """
    nodes = _run_timings.pop(thread_id, {})
    if not nodes:
        return

    total = sum(v["duration_sec"] for v in nodes.values())
    slowest = max(nodes, key=lambda n: nodes[n]["duration_sec"]) if nodes else "—"

    summary = {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "log_type":    "run_summary",
        "thread_id":   thread_id,
        "total_sec":   round(total, 3),
        "slowest_node": slowest,
        "nodes":       nodes,
    }
    _jl.info(json.dumps(summary, ensure_ascii=False))

    # Pretty print to readable log
    _hl.info(f"\n{'─'*60}")
    _hl.info(f"  RUN SUMMARY  [{thread_id[:8]}]  total={total:.2f}s  slowest={slowest}")
    for name, v in sorted(nodes.items(), key=lambda x: x[1]["duration_sec"], reverse=True):
        bar = "█" * max(1, int(v["duration_sec"] / total * 30)) if total else ""
        _hl.info(f"  {name:25s}  {v['duration_sec']:6.2f}s  {bar}")
    _hl.info(f"{'─'*60}\n")

    # Clean up start tracker
    _run_start.pop(thread_id, None)


# ---------------------------------------------------------------------------
# @profile_node  — drop-in replacement for utils.profile_node
# ---------------------------------------------------------------------------

def profile_node(func):
    """
    Decorator for sync and async LangGraph nodes.
    Measures wall-clock duration, logs to JSONL + human-readable file,
    and accumulates per-run totals for emit_run_summary().
    """
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            node_name = func.__name__.replace("_node", "")
            thread_id = _get_thread_id(args, kwargs)
            _hl.info(f"⏱️   [{thread_id[:8]}]  {node_name:25s}  starting…")
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                _emit(node_name, "success", time.perf_counter() - start, thread_id)
                return result
            except Exception as e:
                _emit(node_name, "error", time.perf_counter() - start, thread_id, error=str(e))
                raise
        return async_wrapper

    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            node_name = func.__name__.replace("_node", "")
            thread_id = _get_thread_id(args, kwargs)
            _hl.info(f"⏱️   [{thread_id[:8]}]  {node_name:25s}  starting…")
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                _emit(node_name, "success", time.perf_counter() - start, thread_id)
                return result
            except Exception as e:
                _emit(node_name, "error", time.perf_counter() - start, thread_id, error=str(e))
                raise
        return sync_wrapper