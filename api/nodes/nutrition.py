"""
nodes/nutrition.py — Nutrition Agent node.
"""
from __future__ import annotations
import asyncio
from node_logger import profile_node
from nodes._db import nutrition_db
from config import MAX_NUTRITION_RERANK
from state import RecipeAgentState

_WARNINGS: list[tuple[str, str, float, str]] = [
    ("sodium_mg",     "high", 1200.0, "⚠️ High sodium — consider if managing hypertension"),
    ("fat_g",         "high",   40.0, "⚠️ High fat content"),
    ("sugar_g",       "high",   30.0, "⚠️ High sugar — consider if managing diabetes"),
    ("calories_kcal", "high",  800.0, "⚠️ High calorie meal"),
    ("protein_g",     "low",    8.0,  "ℹ️ Low protein"),
    ("fiber_g",       "low",    2.0,  "ℹ️ Low fiber"),
]


def _build_warnings(nutrition: dict) -> list[str]:
    warnings = []
    for field, direction, threshold, msg in _WARNINGS:
        val = float(nutrition.get(field) or 0.0)
        if direction == "high" and val > threshold:
            warnings.append(msg)
        elif direction == "low" and 0 < val < threshold:
            warnings.append(msg)
    return warnings


def _nutrition_from_meta(meta: dict) -> dict:
    keys = ["calories_kcal", "protein_g", "fat_g", "carbs_g", "fiber_g", "sugar_g", "sodium_mg"]
    return {k: round(float(meta.get(k) or 0.0), 1) for k in keys}


@profile_node
async def nutrition_node(state: RecipeAgentState) -> dict:
    candidates = state.get("candidate_recipes", [])
    if not candidates:
        return {"final_recipes": []}

    if not nutrition_db:
        print("[nutrition_agent] WARNING: nutrition_db is None — skipping re-rank")
        return {"final_recipes": candidates[:MAX_NUTRITION_RERANK]}

    query         = state.get("nutrition_query") or state.get("recipe_query") or state["user_question"]
    candidate_ids = [c.get("meta", {}).get("meal_id", "") for c in candidates if c.get("meta", {}).get("meal_id")]

    recipe_score_map = {
        c["meta"]["meal_id"]: c.get("vector_score", 0.0)
        for c in candidates if c.get("meta", {}).get("meal_id")
    }

    try:
        # ← fixed: search_nutrition() not search()
        nutrition_hits = await asyncio.to_thread(
            nutrition_db.search_nutrition,
            query,
            candidate_ids,
            MAX_NUTRITION_RERANK,
        )
    except Exception as e:
        print(f"[nutrition_agent] Search error: {e} — falling back to recipe ranking")
        return {"final_recipes": candidates[:MAX_NUTRITION_RERANK]}

    if not nutrition_hits:
        return {"final_recipes": candidates[:MAX_NUTRITION_RERANK]}

    final = []
    for hit in nutrition_hits:
        meal_id         = hit.get("meta", {}).get("meal_id", "")
        recipe_score    = recipe_score_map.get(meal_id, 0.0)
        nutrition_score = hit.get("vector_score", 0.0)
        combined_score  = round(0.5 * recipe_score + 0.5 * nutrition_score, 4)
        nutrition       = _nutrition_from_meta(hit.get("meta", {}))
        warnings        = _build_warnings(nutrition)

        final.append({
            **hit,
            "recipe_score":    round(recipe_score, 4),
            "nutrition_score": round(nutrition_score, 4),
            "combined_score":  combined_score,
            "nutrition":       nutrition,
            "warnings":        warnings,
        })

    final.sort(key=lambda x: x["combined_score"], reverse=True)
    print(f"[nutrition_agent] Re-ranked → top {len(final)}: {[f['meta'].get('name','?') for f in final]}")
    return {"final_recipes": final}