"""
api/db_connector.py — Search-only Milvus connector for the Recipe RAG API.

Used exclusively by agent nodes at query time.
No insert, no delete, no schema creation — read-only.

Two collections:
    recipes_main       → Recipe Agent  (semantic search by dish/ingredients/cuisine)
    nutrition_profiles → Nutrition Agent (semantic re-rank by health goal)
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from pymilvus import Collection, connections, utility
from sentence_transformers import SentenceTransformer

from config import (
    COLLECTION_NUTRITION,
    COLLECTION_RECIPES,
)

MILVUS_HOST = "172.17.0.1"
MILVUS_PORT = "19530"

_connected = False


def _ensure_connected() -> None:
    global _connected
    if not _connected:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        _connected = True


class RecipeSearchConnector:
    """
    Search-only Milvus connector shared across Recipe and Nutrition agents.

    Usage:
        connector = RecipeSearchConnector(embedding_model)
        hits      = connector.search_recipes("spicy chicken noodles", top_n=10)
        reranked  = connector.search_nutrition("high protein low carb", candidate_ids, top_n=5)
    """

    def __init__(self, embedding_model: SentenceTransformer):
        _ensure_connected()
        self.embedding_model = embedding_model
        self._collections: dict[str, Collection] = {}

    def _get_collection(self, name: str) -> Collection:
        if name not in self._collections:
            if not utility.has_collection(name):
                raise RuntimeError(
                    f"Collection '{name}' does not exist. Run ingest_recipes.py first."
                )
            col = Collection(name)
            col.load()
            self._collections[name] = col
        return self._collections[name]

    def _embed(self, text: str) -> list[float]:
        emb = self.embedding_model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )
        return np.array(emb)[0].tolist()

    # ── Recipe Agent search ────────────────────────────────────────────────────

    def search_recipes(
        self,
        query: str,
        top_n: int = 10,
        candidate_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Semantic search over recipes_main.

        Parameters
        ----------
        query        : natural language recipe query
        top_n        : number of results
        candidate_ids: optional list of meal_ids to restrict search scope

        Returns list of dicts:
            {id, text, meta, vector_score}
            meta: {meal_id, name, category, area, tags, ingredients, thumbnail, youtube}
        """
        col  = self._get_collection(COLLECTION_RECIPES)
        vec  = self._embed(query)
        expr = self._ids_expr(candidate_ids) if candidate_ids else None

        results = col.search(
            data=[vec],
            anns_field="embedding",
            param={"metric_type": "COSINE"},
            limit=top_n,
            expr=expr,
            output_fields=["id", "text", "meta"],
        )

        hits = []
        for hit in results[0]:
            hits.append({
                "id":           hit.id,
                "text":         hit.entity.get("text"),
                "meta":         hit.entity.get("meta"),
                "vector_score": round(float(hit.score), 4),
            })
        return hits

    # ── Nutrition Agent search ─────────────────────────────────────────────────

    def search_nutrition(
        self,
        query: str,
        candidate_ids: List[str],
        top_n: int = 5,
    ) -> List[dict]:
        """
        Semantic re-rank over nutrition_profiles restricted to candidate meal_ids.

        Parameters
        ----------
        query         : health goal query, e.g. "high protein low carb"
        candidate_ids : meal_ids from Recipe Agent — restricts search scope
        top_n         : number of results after re-ranking

        Returns list of dicts:
            {id, text, meta, vector_score}
            meta: {meal_id, name, calories_kcal, protein_g, fat_g,
                   carbs_g, fiber_g, sugar_g, sodium_mg}
        """
        if not candidate_ids:
            return []

        col  = self._get_collection(COLLECTION_NUTRITION)
        vec  = self._embed(query)
        expr = self._ids_expr(candidate_ids)

        results = col.search(
            data=[vec],
            anns_field="embedding",
            param={"metric_type": "COSINE"},
            limit=top_n,
            expr=expr,
            output_fields=["id", "text", "meta"],
        )

        hits = []
        for hit in results[0]:
            hits.append({
                "id":           hit.id,
                "text":         hit.entity.get("text"),
                "meta":         hit.entity.get("meta"),
                "vector_score": round(float(hit.score), 4),
            })
        return hits

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _ids_expr(meal_ids: List[str]) -> str:
        """Build Milvus JSON filter to restrict search to given meal_ids."""
        id_list = '", "'.join(meal_ids)
        return f'meta["meal_id"] in ["{id_list}"]'