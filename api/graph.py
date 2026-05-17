"""
graph.py — assembles the Recipe RAG multi-agent StateGraph.

Pipeline (sequential — each agent feeds the next):

    START
      │
      ▼
  ingredient_node   ← Ingredient Agent: parse intent, build queries
      │
      ▼
  recipe_node       ← Recipe Agent: search recipes_main, filter by restrictions
      │
      ▼
  nutrition_node    ← Nutrition Agent: re-rank by nutrition_profiles, add warnings
      │
      ▼
  responder_node    ← format and return final recommendations
      │
      ▼
    END

No PostgreSQL checkpointer — no per-session history persistence needed.
get_app() is kept async for consistency with server.py lifespan pattern.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from nodes.ingredient import ingredient_node
from nodes.recipe     import recipe_node
from nodes.nutrition  import nutrition_node
from nodes.responder  import responder_node
from nodes.youtube    import youtube_node
from state import RecipeAgentState


# ── Build graph ────────────────────────────────────────────────────────────────

workflow = StateGraph(RecipeAgentState)

workflow.add_node("ingredient_agent", ingredient_node)
workflow.add_node("recipe_agent",     recipe_node)
workflow.add_node("nutrition_agent",  nutrition_node)
workflow.add_node("youtube_agent",    youtube_node)
workflow.add_node("responder",        responder_node)

workflow.add_edge(START,              "ingredient_agent")
workflow.add_edge("ingredient_agent", "recipe_agent")
workflow.add_edge("recipe_agent",     "nutrition_agent")
workflow.add_edge("nutrition_agent",  "youtube_agent")
workflow.add_edge("youtube_agent",    "responder")
workflow.add_edge("responder",        END)


# ── Factory ────────────────────────────────────────────────────────────────────

async def get_app():
    """
    Async factory — called once from server.py lifespan().
    No checkpointer: recipe sessions are stateless (no history needed).
    Returns compiled graph.
    """
    compiled = workflow.compile()
    return compiled