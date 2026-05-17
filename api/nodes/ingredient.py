"""
nodes/ingredient.py — Ingredient Agent node.

Responsibilities:
  - Parse user's natural language request into a structured UserIntent
  - Build two search queries:
      recipe_query    → for recipes_main  (what dish / ingredients / cuisine)
      nutrition_query → for nutrition_profiles (health goal)
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

from config import get_llm
from node_logger import profile_node
from state import RecipeAgentState, UserIntent, reset_per_turn_state

INGREDIENT_SYSTEM = """You are an Ingredient Agent in a recipe recommendation system.

Extract from the user message:
- available_ingredients: list of ingredients the user has or wants to use
- dietary_restrictions: list of restrictions (vegan, vegetarian, gluten-free, halal, kosher, dairy-free, nut-free, etc.)
- cuisine_preferences: list of cuisines the user prefers (Italian, Asian, Mexican, etc.)
- health_goal: one of: "high protein", "low calorie", "low carb", "low fat", "high fiber", "low sodium", "balanced"
- meal_type: one of: "breakfast", "lunch", "dinner", "snack", "dessert", "any"

Then build two search queries:
- recipe_query: natural language for searching recipes by ingredients and cuisine
- nutrition_query: natural language describing the nutritional profile they want

If a field is not mentioned, use an empty list or "any"/"balanced".
"""

INGREDIENT_USER = "User request: {question}"


class IngredientAgentOutput(BaseModel):
    available_ingredients: List[str] = Field(default_factory=list)
    dietary_restrictions:  List[str] = Field(default_factory=list)
    cuisine_preferences:   List[str] = Field(default_factory=list)
    health_goal:           str       = Field(default="balanced")
    meal_type:             str       = Field(default="any")
    recipe_query:          str       = Field(..., description="Query for semantic recipe search")
    nutrition_query:       str       = Field(..., description="Query for nutrition profile search")


@profile_node
async def ingredient_node(state: RecipeAgentState) -> dict:
    """
    Ingredient Agent — parses user request into structured intent and search queries.
    Feeds recipe_query to Recipe Agent and nutrition_query to Nutrition Agent.
    """
    question = state["user_question"]
    llm = get_llm("ingredient_agent")
    structured_llm = llm.with_structured_output(IngredientAgentOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", INGREDIENT_SYSTEM),
        ("human",  INGREDIENT_USER),
    ])
    chain = prompt | structured_llm

    try:
        result: IngredientAgentOutput = await chain.ainvoke({"question": question})
    except Exception as e:
        return {
            **reset_per_turn_state(),
            "error": f"Ingredient Agent failed: {e}",
        }

    print(f"[ingredient_agent] intent={result.model_dump()}")

    user_intent: UserIntent = {
        "available_ingredients": result.available_ingredients,
        "dietary_restrictions":  result.dietary_restrictions,
        "cuisine_preferences":   result.cuisine_preferences,
        "health_goal":           result.health_goal,
        "meal_type":             result.meal_type,
    }

    return {
        **reset_per_turn_state(),
        "user_intent":     user_intent,
        "recipe_query":    result.recipe_query,
        "nutrition_query": result.nutrition_query,
    }
