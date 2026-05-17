"""
nodes/_db.py — shared search connector singleton for all agent nodes.
"""
from __future__ import annotations
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME
from db_connector import RecipeSearchConnector

embedding_model = None
recipe_db       = None   # used by recipe_node  → search_recipes()
nutrition_db    = None   # used by nutrition_node → search_nutrition()

try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    _connector      = RecipeSearchConnector(embedding_model)
    recipe_db       = _connector   # both agents share the same connector
    nutrition_db    = _connector   # methods are on the same class
except Exception as e:
    print(f"WARNING: DB connector failed ({type(e).__name__}: {e}). Agents unavailable.")