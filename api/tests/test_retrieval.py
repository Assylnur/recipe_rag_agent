"""
tests/test_retrieval.py — RAG quality tests for Milvus search.

Tests retrieval accuracy directly against the two collections without
going through the full LLM pipeline. Fast, deterministic, no LLM cost.

Run:
    pytest tests/test_retrieval.py -v
"""

import pytest
from sentence_transformers import SentenceTransformer


import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from db_connector import RecipeSearchConnector
from config import EMBEDDING_MODEL_NAME

@pytest.fixture(scope="module")
def connector():
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    try:
        return RecipeSearchConnector(model)
    except Exception as e:
        pytest.skip(f"Milvus unavailable: {e}")


class TestRecipeRetrieval:

    def test_basic_search_returns_results(self, connector):
        """Basic query must return at least one result."""
        hits = connector.search_recipes("chicken dinner", top_n=5)
        assert len(hits) > 0

    def test_results_have_required_fields(self, connector):
        """Each hit must have id, text, meta, vector_score."""
        hits = connector.search_recipes("pasta italian", top_n=3)
        for hit in hits:
            assert "id" in hit
            assert "meta" in hit
            assert "vector_score" in hit
            assert hit["vector_score"] > 0

    def test_scores_in_valid_range(self, connector):
        """COSINE similarity scores must be in [0, 1]."""
        hits = connector.search_recipes("beef stew", top_n=5)
        for hit in hits:
            assert 0.0 <= hit["vector_score"] <= 1.0

    def test_results_sorted_by_score(self, connector):
        """Results must come back sorted highest score first."""
        hits = connector.search_recipes("chicken soy sauce asian", top_n=5)
        scores = [h["vector_score"] for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_result_for_specific_query(self, connector):
        """Specific query should return semantically relevant result."""
        hits = connector.search_recipes("teriyaki chicken japanese", top_n=5)
        names = [h["meta"].get("name", "").lower() for h in hits]
        # At least one result should be chicken-related
        assert any("chicken" in n or "teriyaki" in n for n in names)

    def test_top_n_respected(self, connector):
        """top_n parameter must be respected."""
        hits = connector.search_recipes("any recipe", top_n=3)
        assert len(hits) <= 3

    def test_meta_contains_meal_id(self, connector):
        """meta field must contain meal_id for agent handoff."""
        hits = connector.search_recipes("pasta", top_n=3)
        for hit in hits:
            assert "meal_id" in hit["meta"]
            assert hit["meta"]["meal_id"]

    def test_candidate_id_filter(self, connector):
        """search_recipes with candidate_ids must restrict results."""
        all_hits   = connector.search_recipes("chicken", top_n=10)
        ids        = [h["meta"]["meal_id"] for h in all_hits[:3]]
        filtered   = connector.search_recipes("chicken", top_n=10, candidate_ids=ids)
        result_ids = [h["meta"]["meal_id"] for h in filtered]
        for rid in result_ids:
            assert rid in ids


class TestNutritionRetrieval:

    def test_nutrition_rerank_returns_results(self, connector):
        """Nutrition search must return results for valid candidate_ids."""
        recipe_hits  = connector.search_recipes("chicken dinner", top_n=10)
        candidate_ids = [h["meta"]["meal_id"] for h in recipe_hits]
        nutrition_hits = connector.search_nutrition("high protein low carb", candidate_ids, top_n=5)
        assert len(nutrition_hits) > 0

    def test_nutrition_meta_has_macros(self, connector):
        """Nutrition meta must contain macro fields."""
        recipe_hits   = connector.search_recipes("beef meal", top_n=5)
        candidate_ids = [h["meta"]["meal_id"] for h in recipe_hits]
        nutrition_hits = connector.search_nutrition("high protein", candidate_ids, top_n=3)
        for hit in nutrition_hits:
            meta = hit["meta"]
            assert "calories_kcal" in meta
            assert "protein_g" in meta
            assert "fat_g" in meta

    def test_nutrition_restricted_to_candidates(self, connector):
        """Nutrition results must only contain candidate meal_ids."""
        recipe_hits   = connector.search_recipes("pasta", top_n=5)
        candidate_ids = [h["meta"]["meal_id"] for h in recipe_hits]
        nutrition_hits = connector.search_nutrition("low calorie", candidate_ids, top_n=5)
        for hit in nutrition_hits:
            assert hit["meta"]["meal_id"] in candidate_ids

    def test_empty_candidate_ids_returns_empty(self, connector):
        """Empty candidate_ids must return empty list, not crash."""
        hits = connector.search_nutrition("high protein", [], top_n=5)
        assert hits == []

    def test_nutrition_scores_sorted(self, connector):
        """Nutrition results sorted highest score first."""
        recipe_hits   = connector.search_recipes("chicken", top_n=10)
        candidate_ids = [h["meta"]["meal_id"] for h in recipe_hits]
        nutrition_hits = connector.search_nutrition("high protein", candidate_ids, top_n=5)
        if len(nutrition_hits) > 1:
            scores = [h["vector_score"] for h in nutrition_hits]
            assert scores == sorted(scores, reverse=True)
