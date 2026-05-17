"""
llm_logger.py — per-LLM-call token and duration logger.

Logs structured JSON to stdout + file.
Alloy scrapes stdout and forwards to Loki automatically.

Each line is a JSON object that Loki parses with | json in LogQL.

Token extraction strategy (in priority order):
  1. generations[0][0].message.usage_metadata  — always populated by LangChain
     from vLLM's stream_options.include_usage response. Works for both
     streaming and non-streaming. Keys: input_tokens / output_tokens.
  2. llm_output["token_usage"]                 — non-streaming fallback,
     older LangChain versions.
  3. Never count tokens manually (+1 per token) — inaccurate for reasoning
     models that emit reasoning tokens separately.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from logging.handlers import RotatingFileHandler


LOG_DIR      = "logs"
LLM_LOG_FILE = os.path.join(LOG_DIR, "llm_calls.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

_logger = logging.getLogger("LLMCallLogger")
_logger.setLevel(logging.INFO)
_logger.propagate = False  # prevent duplicate lines via uvicorn root logger

# File handler — pure JSON, no prefix (Alloy regex extracts {.*})
# _fh = logging.FileHandler(LLM_LOG_FILE, encoding="utf-8")
_fh = RotatingFileHandler(LLM_LOG_FILE, maxBytes=1024*1024, backupCount=5, encoding="utf-8")  # 1MB max file size

_fh.setFormatter(logging.Formatter("%(message)s"))
_logger.addHandler(_fh)

# Stdout handler — emoji prefix for human readability in docker logs
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("%(emoji)s %(message)s"))
_logger.addHandler(_sh)

_STATUS_EMOJI = {
    "success": "˚🎀༘⋆ꉂ",
    "error":   "🥀",
}


class LLMCallbackLogger(BaseCallbackHandler):
    def __init__(
        self,
        node_type: str,
        model: str = "",
    ):
        super().__init__()
        self.node_type = node_type
        self.model     = model

        self._start_time:       Optional[float] = None
        self._first_token_time: Optional[float] = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        self._start_time       = time.time()
        self._first_token_time = None

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Capture TTFT on the very first content token
        if self._first_token_time is None and token:
            self._first_token_time = time.time()

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        end_time = time.time()
        duration = round(end_time - (self._start_time or end_time), 3)

        # TTFT — only meaningful for streaming (responder).
        # Non-streaming nodes: first token = full response = duration.
        if self._first_token_time:
            ttft = round(self._first_token_time - self._start_time, 3)
        else:
            ttft = duration

        # ── Token extraction ──────────────────────────────────────────────────
        # Priority 1: generations[0][0].message.usage_metadata
        # LangChain populates this from vLLM's include_usage response for both
        # streaming and non-streaming. Keys: input_tokens / output_tokens.
        prompt_tokens     = 0
        completion_tokens = 0
        total_tokens      = 0

        try:
            msg = response.generations[0][0].message
            um  = getattr(msg, "usage_metadata", None)
            if um:
                prompt_tokens     = um.get("input_tokens",  0)
                completion_tokens = um.get("output_tokens", 0)
                total_tokens      = um.get("total_tokens",  0) or (prompt_tokens + completion_tokens)
        except (IndexError, AttributeError):
            pass

        # Priority 2: llm_output fallback (non-streaming, older LangChain)
        if prompt_tokens == 0 and response.llm_output:
            usage = (
                response.llm_output.get("token_usage") or
                response.llm_output.get("usage") or {}
            )
            prompt_tokens     = usage.get("prompt_tokens",     0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens      = usage.get("total_tokens",      0) or (prompt_tokens + completion_tokens)

        self._emit(
            status="success",
            duration_sec=duration,
            ttft_sec=ttft,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        end_time = time.time()
        duration = round(end_time - (self._start_time or end_time), 3)
        self._emit(
            status="error",
            duration_sec=duration,
            ttft_sec=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error=str(error),
        )

    def _emit(self, status: str, duration_sec: float, **extra) -> None:
        record = {
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "log_type":          "llm_call",
            "node_type":         self.node_type,
            "model":             self.model,
            "status":            status,
            "duration_sec":      duration_sec,
            "ttft_sec":          extra.get("ttft_sec"),
            "prompt_tokens":     extra.get("prompt_tokens",     0),
            "completion_tokens": extra.get("completion_tokens", 0),
            "total_tokens":      extra.get("total_tokens",      0),
            **({"error": extra["error"]} if status == "error" else {}),
        }
        emoji = _STATUS_EMOJI.get(status, "૮ ྀིᴗ͈ . ᴗ͈ ྀིა")
        line  = json.dumps(record, ensure_ascii=False)
        _logger.info(line, extra={"emoji": emoji})