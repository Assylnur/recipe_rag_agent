"""
tests/test_rag_quality.py — Unit tests for RAG quality assurance module.

Pure unit tests — no API, no Milvus, no LLM needed.

Run:
    pytest tests/test_rag_quality.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_quality import check_retrieval_confidence, detect_hallucination, format_sources


def make_hit(name: str, score: float, meal_id: str = "123") -> dict:
    return {"meta": {"name": name, "meal_id": meal_id, "category": "Chicken", "area": "Asian"},
            "vector_score": score, "combined_score": score}


class TestRetrievalConfidence:

    def test_high_score_hits_not_flagged(self):
        hits = [make_hit("Chicken Teriyaki", 0.85), make_hit("Beef Stir Fry", 0.78)]
        result = check_retrieval_confidence(hits)
        assert not result["low_confidence"]
        assert result["warning"] is None

    def test_low_score_hits_flagged(self):
        hits = [make_hit("Random Dish", 0.10), make_hit("Unknown", 0.12)]
        result = check_retrieval_confidence(hits)
        assert result["low_confidence"]
        assert result["warning"] is not None

    def test_empty_hits_flagged(self):
        result = check_retrieval_confidence([])
        assert result["low_confidence"]
        assert result["total"] == 0

    def test_precision_calculated_correctly(self):
        hits = [make_hit("A", 0.8), make_hit("B", 0.5), make_hit("C", 0.1)]
        result = check_retrieval_confidence(hits)
        # 2 out of 3 above threshold (0.30)
        assert result["above_threshold"] == 2
        assert result["precision_est"] == round(2/3, 3)

    def test_scores_reported_correctly(self):
        hits = [make_hit("A", 0.9), make_hit("B", 0.6)]
        result = check_retrieval_confidence(hits)
        assert result["max_score"] == 0.9
        assert result["min_score"] == 0.6
        assert result["avg_score"] == 0.75


class TestHallucinationDetection:

    def test_grounded_answer_passes(self):
        recipes = [make_hit("Chicken Teriyaki", 0.85, "001"),
                   make_hit("Beef Stroganoff", 0.75, "002")]
        answer = "I recommend Chicken Teriyaki as your top choice."
        result = detect_hallucination(answer, recipes)
        assert result["grounded"]
        assert result["warning"] is None

    def test_empty_recipes_flags_ungrounded(self):
        result = detect_hallucination("Some answer", [])
        assert not result["grounded"]

    def test_no_recipe_names_in_answer_passes(self):
        recipes = [make_hit("Chicken Teriyaki", 0.85, "001")]
        answer = "Here are some great options for your dinner tonight."
        result = detect_hallucination(answer, recipes)
        assert result["grounded"]

    def test_returns_required_fields(self):
        recipes = [make_hit("Chicken Teriyaki", 0.85, "001")]
        result = detect_hallucination("Chicken Teriyaki is great", recipes)
        assert "grounded" in result
        assert "mentioned_names" in result
        assert "grounded_names" in result
        assert "ungrounded" in result


class TestFormatSources:

    def test_sources_block_contains_meal_id(self):
        recipes = [make_hit("Chicken Teriyaki", 0.85, "52772")]
        corpus  = {"52772": {"source": "https://example.com", "youtube": ""}}
        block   = format_sources(recipes, corpus)
        assert "52772" in block
        assert "Chicken Teriyaki" in block

    def test_sources_block_contains_category(self):
        recipes = [make_hit("Chicken Teriyaki", 0.85, "52772")]
        corpus  = {"52772": {}}
        block   = format_sources(recipes, corpus)
        assert "Chicken" in block

    def test_empty_recipes_returns_empty(self):
        block = format_sources([], {})
        assert block == ""

    def test_duplicates_not_repeated(self):
        recipe = make_hit("Chicken Teriyaki", 0.85, "52772")
        recipes = [recipe, recipe]
        block = format_sources(recipes, {})
        assert block.count("52772") == 1

    def test_relevance_score_shown(self):
        recipes = [make_hit("Chicken Teriyaki", 0.85, "52772")]
        block = format_sources(recipes, {})
        assert "%" in block

    def test_original_source_link_included(self):
        recipes = [make_hit("Chicken Teriyaki", 0.85, "52772")]
        corpus  = {"52772": {"source": "https://recipe-site.com/teriyaki"}}
        block   = format_sources(recipes, corpus)
        assert "https://recipe-site.com/teriyaki" in block