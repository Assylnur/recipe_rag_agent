"""
nodes/responder.py — Responder node.

Takes final_recipes from Nutrition Agent and generates a user-friendly
recommendation with instructions, nutrition summary, and health notes.
Full meal details (instructions, thumbnail, youtube) are loaded from
the in-memory corpus loaded at startup — no PostgreSQL needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from config import CORPUS_PATH, get_llm
from rag_quality import detect_hallucination, format_sources
from node_logger import profile_node
from state import RecipeAgentState

NO_RECIPES_MSG = (
    "Unfortunately, I couldn't find any recipes matching your request. "
    "Try changing the ingredients or relaxing the constraints."
)

RESPONDER_SYSTEM = """You are a friendly recipe recommendation assistant.
The user asked for recipe suggestions. You have already found and ranked the best matching recipes.

Present the top recipes clearly:
- Recipe name, cuisine, category
- Key ingredients
- Brief cooking instructions (summarized)
- Nutrition summary (calories, protein, carbs, fat)
- Any health warnings
- YouTube link if available

Be warm, helpful, and concise. Respond in the same language as the user's question."""

# ── Corpus loader (in-memory, loaded once) ────────────────────────────────────
_corpus: Optional[dict[str, dict]] = None


def _load_corpus() -> dict[str, dict]:
    global _corpus
    if _corpus is None:
        path = Path(CORPUS_PATH)
        if path.exists():
            meals = json.loads(path.read_text(encoding="utf-8"))
            _corpus = {m["id"]: m for m in meals}
            print(f"[responder] Corpus loaded: {len(_corpus)} meals")
        else:
            print(f"[responder] WARNING: corpus not found at {CORPUS_PATH}")
            _corpus = {}
    return _corpus


def _format_nutrition(nutrition: dict) -> str:
    return (
        f"{nutrition.get('calories_kcal', 0):.0f} kcal | "
        f"{nutrition.get('protein_g', 0):.1f}g protein | "
        f"{nutrition.get('carbs_g', 0):.1f}g carbs | "
        f"{nutrition.get('fat_g', 0):.1f}g fat"
    )


def _build_context(final_recipes: list[dict], corpus: dict[str, dict]) -> str:
    """Format ranked recipes into a context block for the LLM."""
    lines = []
    for i, recipe in enumerate(final_recipes, 1):
        meta    = recipe.get("meta", {})
        meal_id = meta.get("meal_id", "")
        full    = corpus.get(meal_id, {})

        name        = meta.get("name", "Unknown")
        category    = meta.get("category", "")
        area        = meta.get("area", "")
        ingredients = meta.get("ingredients") or full.get("ingredients", [])
        instructions = (full.get("instructions") or "")[:600]  # trim long instructions
        youtube     = full.get("youtube", "")
        thumbnail   = full.get("thumbnail", "")
        nutrition   = recipe.get("nutrition", {})
        warnings    = recipe.get("warnings", [])

        ing_str = ", ".join(
            (ing["ingredient"] if isinstance(ing, dict) else ing)
            for ing in (ingredients or [])
        )

        lines.append(
            f"### Recipe {i}: {name}\n"
            f"Cuisine: {area} | Category: {category}\n"
            f"Ingredients: {ing_str}\n"
            f"Instructions: {instructions}{'...' if len(full.get('instructions', '')) > 600 else ''}\n"
            f"Nutrition: {_format_nutrition(nutrition)}\n"
            + (f"Warnings: {' | '.join(warnings)}\n" if warnings else "")
            + (f"Video: {youtube}\n" if youtube else "")
        )

    return "\n\n".join(lines)


@profile_node
async def responder_node(state: RecipeAgentState, config: RunnableConfig) -> dict:
    """Generate the final recipe recommendation response."""
    if state.get("final_answer"):
        return {}

    final_recipes = state.get("final_recipes", [])
    if not final_recipes:
        return {
            "final_answer": NO_RECIPES_MSG,
            "messages": [AIMessage(content=NO_RECIPES_MSG)],
        }

    corpus  = _load_corpus()
    context = _build_context(final_recipes, corpus)

    intent = state.get("user_intent") or {}
    restrictions = intent.get("dietary_restrictions", [])
    health_goal  = intent.get("health_goal", "balanced")

    user_prompt = (
        f"User request: {state['user_question']}\n\n"
        f"Health goal: {health_goal}\n"
        f"Dietary restrictions: {', '.join(restrictions) or 'none'}\n\n"
        f"Top matching recipes:\n\n{context}"
    )

    is_streaming = state.get("streaming", False)
    llm = get_llm("responder", streaming=is_streaming)

    messages_to_send = [
        SystemMessage(content=RESPONDER_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages_to_send, config=config)
    answer   = response.content

    # Source attribution
    sources_block = format_sources(final_recipes, corpus)
    if sources_block:
        answer += sources_block

    # Append YouTube video links from YouTube Agent
    video_links = state.get("video_links", [])
    video_section = _format_video_links(video_links)
    if video_section:
        answer += video_section

    # Hallucination detection (logged, not shown to user)
    detect_hallucination(answer, final_recipes)

    return {
        "final_answer": answer,
        "messages":     [AIMessage(content=answer)],
    }


def _format_video_links(video_links: list[dict]) -> str:
    """Format video links section for the final answer."""
    if not video_links:
        return ""
    lines = ["\n\n---\n### 🎬 Cooking Tutorial Videos"]
    for v in video_links:
        lines.append(f"- **{v['recipe']}** — [{v['title']}]({v['url']}) by {v['channel']}")
    return "\n".join(lines)