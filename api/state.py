"""
state.py — RecipeAgentState for the Recipe RAG multi-agent system.

Pipeline flow:
    ingredient_node → recipe_node → nutrition_node → responder_node

Each agent writes its output to dedicated state fields so the next agent
has a clean, typed interface to read from.
"""

from typing import Annotated, List, Optional, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class UserIntent(TypedDict):
    """Structured output of the Ingredient Agent."""
    available_ingredients: List[str]   # what the user has on hand
    dietary_restrictions:  List[str]   # vegan, gluten-free, halal, etc.
    cuisine_preferences:   List[str]   # Asian, Mediterranean, etc.
    health_goal:           str         # "high protein", "low calorie", "balanced", etc.
    meal_type:             str         # breakfast, lunch, dinner, snack, any


class RecipeAgentState(TypedDict):
    # ── Conversation ───────────────────────────────────────────────────────────
    messages:       Annotated[List[BaseMessage], add_messages]
    user_question:  str
    final_answer:   str
    error:          Optional[str]
    streaming:      bool

    # ── Ingredient Agent output ────────────────────────────────────────────────
    user_intent:    Optional[UserIntent]    # structured parse of user request
    recipe_query:   str                     # text query for recipes_main vector search
    nutrition_query: str                    # text query for nutrition_profiles search

    # ── Recipe Agent output ────────────────────────────────────────────────────
    candidate_recipes: List[dict]           # top-N hits from recipes_main
                                            # each: {meal_id, name, category, area,
                                            #        tags, ingredients, score}

    # ── Nutrition Agent output ─────────────────────────────────────────────────
    final_recipes:  List[dict]
    video_links:    List[dict]  # [{recipe, title, url, channel}] from YouTube Agent
                                            # each: {meal_id, name, score,
                                            #        nutrition_score, nutrition, warnings}


def reset_per_turn_state() -> dict:
    """Clear all scratch fields at the start of each new turn."""
    return {
        "final_answer":     None,
        "error":            None,
        "user_intent":      None,
        "recipe_query":     "",
        "nutrition_query":  "",
        "candidate_recipes": [],
        "final_recipes":    [],
        "video_links":      [],
    }