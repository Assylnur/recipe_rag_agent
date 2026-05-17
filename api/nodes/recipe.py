"""
nodes/recipe.py — Recipe Agent node.
"""
from __future__ import annotations
import asyncio
from langchain_core.messages import AIMessage
from config import MAX_RECIPE_RESULTS
from rag_quality import check_retrieval_confidence
from node_logger import profile_node
from nodes._db import recipe_db
from state import RecipeAgentState

NO_RECIPES_MSG = (
    "Sorry, I couldn't find any recipes matching your request. "
    "Try adjusting the ingredients or relaxing some restrictions."
)

_RESTRICTION_SIGNALS: dict[str, list[str]] = {
    "vegan":       ["beef", "chicken", "pork", "lamb", "fish", "shrimp", "bacon",
                    "ham", "turkey", "tuna", "salmon", "egg", "eggs", "milk",
                    "cheese", "butter", "cream", "yogurt", "honey"],
    "vegetarian":  ["beef", "chicken", "pork", "lamb", "fish", "shrimp", "bacon",
                    "ham", "turkey", "tuna", "salmon"],
    "gluten-free": ["flour", "bread", "pasta", "wheat", "barley", "rye",
                    "soy sauce", "breadcrumbs"],
    "dairy-free":  ["milk", "cheese", "butter", "cream", "yogurt", "parmesan",
                    "mozzarella", "cheddar", "whipped cream"],
    "halal":       ["pork", "bacon", "ham", "lard", "wine", "beer",
                    "alcohol", "rum", "whiskey", "vodka"],
    "kosher":      ["pork", "bacon", "ham", "shellfish", "shrimp", "lobster",
                    "crab", "clam", "oyster"],
    "nut-free":    ["almond", "walnut", "pecan", "cashew", "pistachio",
                    "hazelnut", "peanut", "ground almonds"],
}


def _passes_restrictions(recipe: dict, restrictions: list[str]) -> bool:
    if not restrictions:
        return True
    ingredient_names = {
        ing.lower() for ing in (recipe.get("meta", {}).get("ingredients") or [])
    }
    for restriction in restrictions:
        key = restriction.lower().replace(" ", "-")
        for blocked in _RESTRICTION_SIGNALS.get(key, []):
            if any(blocked in ing for ing in ingredient_names):
                print(f"[recipe_agent] '{recipe.get('meta', {}).get('name')}' blocked by '{restriction}' → '{blocked}'")
                return False
    return True


@profile_node
async def recipe_node(state: RecipeAgentState) -> dict:
    if not recipe_db:
        print("[recipe_agent] ERROR: recipe_db is None — collection missing or Milvus unreachable")
        return {
            "final_answer":     NO_RECIPES_MSG,
            "messages":         [AIMessage(content=NO_RECIPES_MSG)],
            "candidate_recipes": [],
        }

    query       = state.get("recipe_query") or state["user_question"]
    intent      = state.get("user_intent") or {}
    restrictions = intent.get("dietary_restrictions", [])

    try:
        # ← fixed: search_recipes() not search()
        raw_hits = await asyncio.to_thread(
            recipe_db.search_recipes,
            query,
            MAX_RECIPE_RESULTS * 2,
        )
    except Exception as e:
        print(f"[recipe_agent] Search error: {e}")
        return {"candidate_recipes": [], "error": str(e)}

    filtered   = [h for h in raw_hits if _passes_restrictions(h, restrictions)]
    candidates = filtered[:MAX_RECIPE_RESULTS]

    # RAG quality check
    check_retrieval_confidence(raw_hits, label="recipe_agent")
    print(f"[recipe_agent] {len(raw_hits)} hits → {len(filtered)} after filter → {len(candidates)} candidates")

    if not candidates:
        return {
            "final_answer":     NO_RECIPES_MSG,
            "messages":         [AIMessage(content=NO_RECIPES_MSG)],
            "candidate_recipes": [],
        }

    return {"candidate_recipes": candidates}