"""
rag_quality.py — RAG quality assurance for the Recipe RAG Agent.

Components
----------
check_retrieval_confidence()  — flag low-score results, log precision estimate
detect_hallucination()        — check if answer mentions recipes not in retrieved set
format_sources()              — build source attribution block for responder
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.30   # COSINE score below this → low confidence warning
HALLUCINATION_THRESHOLD = 0.5  # fraction of recipe names in answer that must be grounded


# ── Retrieval confidence ───────────────────────────────────────────────────────

def check_retrieval_confidence(hits: list[dict], label: str = "search") -> dict:
    """
    Evaluate retrieval quality for a set of Milvus hits.

    Returns
    -------
    {
        "total":          int,
        "above_threshold": int,
        "precision_est":  float,   # fraction above threshold
        "min_score":      float,
        "max_score":      float,
        "avg_score":      float,
        "low_confidence": bool,    # True if avg below threshold
        "warning":        str | None
    }
    """
    if not hits:
        return {
            "total": 0, "above_threshold": 0, "precision_est": 0.0,
            "min_score": 0.0, "max_score": 0.0, "avg_score": 0.0,
            "low_confidence": True, "warning": "No results returned",
        }

    scores         = [h.get("vector_score", 0.0) for h in hits]
    above          = [s for s in scores if s >= CONFIDENCE_THRESHOLD]
    avg_score      = round(sum(scores) / len(scores), 4)
    precision_est  = round(len(above) / len(scores), 3)
    low_confidence = avg_score < CONFIDENCE_THRESHOLD

    warning = None
    if low_confidence:
        warning = (
            f"[{label}] Low retrieval confidence: avg={avg_score:.3f} "
            f"(threshold={CONFIDENCE_THRESHOLD}). Results may be irrelevant."
        )
        log.warning(warning)
    else:
        log.info(
            "[%s] Retrieval quality: avg=%.3f precision_est=%.1f%% (%d/%d above threshold)",
            label, avg_score, precision_est * 100, len(above), len(scores),
        )

    return {
        "total":           len(hits),
        "above_threshold": len(above),
        "precision_est":   precision_est,
        "min_score":       round(min(scores), 4),
        "max_score":       round(max(scores), 4),
        "avg_score":       avg_score,
        "low_confidence":  low_confidence,
        "warning":         warning,
    }


# ── Hallucination detection ────────────────────────────────────────────────────

def detect_hallucination(answer: str, retrieved_recipes: list[dict]) -> dict:
    """
    Check if recipe names mentioned in the answer are grounded
    in the retrieved recipe set.

    Strategy: extract capitalized multi-word phrases from the answer
    (likely recipe names), check if they appear in retrieved recipe names.

    Returns
    -------
    {
        "grounded":        bool,
        "mentioned_names": list[str],   # recipe names found in answer
        "grounded_names":  list[str],   # subset that matched retrieved recipes
        "ungrounded":      list[str],   # names not in retrieved set
        "warning":         str | None,
    }
    """
    if not retrieved_recipes:
        return {
            "grounded": False, "mentioned_names": [], "grounded_names": [],
            "ungrounded": [], "warning": "No retrieved recipes to ground against",
        }

    # Build set of known recipe names (lowercase for comparison)
    known_names = {
        r.get("meta", {}).get("name", "").lower()
        for r in retrieved_recipes
        if r.get("meta", {}).get("name")
    }

    # Extract recipe-like names from answer — capitalized phrases 2-5 words
    pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b')
    candidates = pattern.findall(answer)

    # Filter to plausible recipe names (exclude common sentence starters)
    stopwords = {"Here", "The", "This", "These", "Your", "Our", "Based",
                 "Note", "Please", "Also", "However", "For", "With", "If"}
    mentioned = [c for c in candidates if c.split()[0] not in stopwords]
    mentioned = list(dict.fromkeys(mentioned))[:10]  # deduplicate, cap at 10

    grounded   = [m for m in mentioned if m.lower() in known_names]
    ungrounded = [m for m in mentioned if m.lower() not in known_names]

    # Only flag if we have clear recipe name mentions that don't match
    is_grounded = len(ungrounded) == 0 or (
        len(grounded) / max(len(mentioned), 1) >= HALLUCINATION_THRESHOLD
    )

    warning = None
    if not is_grounded and ungrounded:
        warning = (
            f"Potential hallucination: answer mentions {ungrounded} "
            f"which are not in retrieved recipes {list(known_names)[:5]}"
        )
        log.warning("[hallucination] %s", warning)
    else:
        log.info(
            "[hallucination_check] OK — %d/%d names grounded",
            len(grounded), len(mentioned),
        )

    return {
        "grounded":        is_grounded,
        "mentioned_names": mentioned,
        "grounded_names":  grounded,
        "ungrounded":      ungrounded,
        "warning":         warning,
    }


# ── Source attribution ─────────────────────────────────────────────────────────

def format_sources(final_recipes: list[dict], corpus: dict[str, dict]) -> str:
    """
    Build a source attribution block for the responder answer.

    Format:
        ---
        **Sources**
        - Teriyaki Chicken — TheMealDB #52772 | Category: Chicken | Area: Japanese
        - Beef Stroganoff  — TheMealDB #52835 | Category: Beef | Area: Russian
    """
    if not final_recipes:
        return ""

    lines = ["\n\n---\n**Sources**"]
    seen  = set()

    for recipe in final_recipes:
        meta    = recipe.get("meta", {})
        meal_id = meta.get("meal_id", "")

        if meal_id in seen:
            continue
        seen.add(meal_id)

        name     = meta.get("name", "Unknown")
        category = meta.get("category", "")
        area     = meta.get("area", "")
        full     = corpus.get(meal_id, {})
        source   = full.get("source", "")   # original recipe URL if available
        score    = recipe.get("combined_score") or recipe.get("vector_score") or 0.0

        parts = [f"TheMealDB #{meal_id}"]
        if category:
            parts.append(f"Category: {category}")
        if area:
            parts.append(f"Area: {area}")
        parts.append(f"Relevance: {score:.0%}")

        line = f"- **{name}** — {' | '.join(parts)}"
        if source:
            line += f" | [Original recipe]({source})"
        lines.append(line)

    return "\n".join(lines)