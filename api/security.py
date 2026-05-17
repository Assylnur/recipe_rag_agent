"""
security.py — Input validation, content filtering, and rate limiting
for the Recipe RAG Agent.

Components
----------
sanitize_input()     — strip dangerous characters, enforce length limits
is_harmful()         — keyword-based content filter for obvious abuse cases
RateLimiter          — in-memory sliding window, no Redis needed
mask_api_key()       — safe logging of API keys
"""

from __future__ import annotations

import re
import time
import logging
from collections import defaultdict
from threading import Lock

log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

MAX_INPUT_LENGTH  = 1000   # characters
MAX_REQUESTS      = 20     # per window
WINDOW_SECONDS    = 60     # sliding window size


# ── Input sanitization ─────────────────────────────────────────────────────────

# Patterns to strip entirely
_STRIP_PATTERNS = [
    re.compile(r"<[^>]*>"),                          # HTML tags
    re.compile(r"javascript\s*:", re.IGNORECASE),    # JS protocol
    re.compile(r"data\s*:", re.IGNORECASE),          # data URI
    re.compile(r"on\w+\s*=", re.IGNORECASE),         # event handlers
]

# Characters to normalize
_NORMALIZE = str.maketrans({
    "\x00": "",   # null byte
    "\r":   " ",  # carriage return
    "\t":   " ",  # tab → space
})


def sanitize_input(text: str) -> tuple[str, list[str]]:
    """
    Clean user input before passing to agents.

    Returns
    -------
    (cleaned_text, list_of_warnings)

    Warnings are informational — sanitized input is still processed.
    Callers should reject only if cleaned text is empty or too short.
    """
    if not isinstance(text, str):
        return "", ["Input is not a string"]

    warnings: list[str] = []
    original_len = len(text)

    # 1. Normalize whitespace control chars
    text = text.translate(_NORMALIZE)

    # 2. Strip HTML / script injection patterns
    for pattern in _STRIP_PATTERNS:
        if pattern.search(text):
            warnings.append(f"Stripped pattern: {pattern.pattern[:30]}")
            text = pattern.sub("", text)

    # 3. Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text).strip()

    # 4. Enforce length limit
    if len(text) > MAX_INPUT_LENGTH:
        warnings.append(f"Input truncated from {len(text)} to {MAX_INPUT_LENGTH} chars")
        text = text[:MAX_INPUT_LENGTH].strip()

    if len(text) < original_len and not warnings:
        warnings.append("Input was modified during sanitization")

    return text, warnings


# ── Content filtering ──────────────────────────────────────────────────────────

# Grouped by category — any match in a category triggers rejection
_HARMFUL_PATTERNS: dict[str, list[re.Pattern]] = {
    "weapons": [
        re.compile(r"\b(explosive|bomb|poison|toxic|weapon|grenade|ammunition)\b", re.I),
        re.compile(r"\b(make.*poison|synthesize.*drug|cook.*meth)\b", re.I),
    ],
    "self_harm": [
        re.compile(r"\b(suicide|self.harm|overdose|kill myself)\b", re.I),
    ],
    "jailbreak": [
        re.compile(r"\b(ignore\b.{0,20}\b(instructions?|rules?|guidelines?|prompt))\b", re.I),
        re.compile(r"\b(you are now|act as|pretend (to be|you are)|DAN|jailbreak)\b", re.I),
        re.compile(r"\b(forget (you are|your|that you))\b", re.I),
        re.compile(r"(system prompt|reveal your prompt|print your instructions)", re.I),
    ],
    "injection": [
        re.compile(r"(;\s*(DROP|SELECT|INSERT|DELETE|UPDATE)\s+\w+)", re.I),  # SQL
        re.compile(r"\{\{.*\}\}|\{%.*%\}"),                                   # template injection
    ],
}


def is_harmful(text: str) -> tuple[bool, str]:
    """
    Check if input contains obviously harmful or adversarial content.

    Returns
    -------
    (is_harmful, reason)

    Note: jailbreak/injection attempts are flagged but the system
    still processes them — the LLM's own safety training handles them.
    We log the attempt and sanitize, not necessarily reject.
    """
    for category, patterns in _HARMFUL_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                reason = f"{category}: matched '{pattern.pattern[:50]}'"
                log.warning("[security] Harmful pattern detected — %s", reason)
                return True, reason
    return False, ""


def filter_request(text: str) -> tuple[str, bool, str]:
    """
    Full security pipeline: sanitize → filter.

    Returns
    -------
    (sanitized_text, should_block, reason)

    should_block=True only for weapons/self_harm categories.
    Jailbreak and injection attempts are sanitized and logged but allowed through
    (the LLM's safety handles them; our job is to log + sanitize).
    """
    sanitized, warnings = sanitize_input(text)

    if warnings:
        log.info("[security] Input sanitized: %s", "; ".join(warnings))

    if not sanitized:
        return "", True, "Empty input after sanitization"

    harmful, reason = is_harmful(sanitized)
    if harmful:
        # Only hard-block weapons and self-harm
        category = reason.split(":")[0]
        if category in ("weapons", "self_harm"):
            return sanitized, True, reason
        else:
            # Log jailbreak/injection but let it through
            log.warning("[security] Adversarial input allowed through (LLM will handle): %s", reason)

    return sanitized, False, ""


# ── Rate limiter ───────────────────────────────────────────────────────────────

class RateLimiter:
    """
    In-memory sliding window rate limiter.
    Keyed by client IP. Thread-safe via Lock.

    No Redis needed — suitable for single-instance deployment.
    For multi-instance, replace with Redis-backed implementation.
    """

    def __init__(self, max_requests: int = MAX_REQUESTS, window: int = WINDOW_SECONDS):
        self.max_requests = max_requests
        self.window       = window
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """
        Check if client is within rate limit.

        Returns
        -------
        (allowed, requests_remaining)
        """
        now = time.time()
        cutoff = now - self.window

        with self._lock:
            # Remove expired timestamps
            self._requests[client_id] = [
                t for t in self._requests[client_id] if t > cutoff
            ]
            count = len(self._requests[client_id])

            if count >= self.max_requests:
                log.warning("[rate_limit] Client '%s' exceeded limit (%d/%d)",
                            client_id, count, self.max_requests)
                return False, 0

            self._requests[client_id].append(now)
            return True, self.max_requests - count - 1

    def reset(self, client_id: str) -> None:
        """Reset rate limit for a client (admin use)."""
        with self._lock:
            self._requests.pop(client_id, None)


# ── API key masking ────────────────────────────────────────────────────────────

def mask_api_key(key: str) -> str:
    """Return safely masked version for logging. e.g. AIza***...***XcQ"""
    if not key or len(key) < 8:
        return "***"
    return f"{key[:4]}***...***{key[-3:]}"