"""
logger.py — audit logger for the Recipe RAG Agent.
"""
from __future__ import annotations
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict
from logging.handlers import RotatingFileHandler
LOG_DIR  = "logs"
LOG_FILE = os.path.join(LOG_DIR, "recipe_rag_audit.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

_logger = logging.getLogger("RecipeRAGAudit")   # ← was LegalRAGAudit
_logger.setLevel(logging.INFO)
_fh = RotatingFileHandler(LOG_FILE, maxBytes=1024*1024, backupCount=5, encoding="utf-8")  # 1MB max file size
_fh.setFormatter(logging.Formatter("%(message)s"))
_logger.addHandler(_fh)


def log_transaction(thread_id: str, state: Dict[str, Any], duration: float):
    try:
        log_entry = {
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "log_type":     "audit",
            "thread_id":    thread_id,
            "duration_sec": round(duration, 2),
            "status":       "ERROR" if state.get("error") else "SUCCESS",
            "candidates":   len(state.get("candidate_recipes") or []),
            "final":        len(state.get("final_recipes") or []),
            "has_videos":   len(state.get("video_links") or []),
            "final_answer": (state.get("final_answer") or "")[:200],
        }
        _logger.info(json.dumps(log_entry, ensure_ascii=False))
    except Exception as e:
        print(f"Failed to write audit log: {e}")