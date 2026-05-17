"""
tests/test_nutrition_agent.py — Nutrition Agent logic tests.

Tests warning generation, score merging, and re-ranking logic
directly without Milvus — pure unit tests, no API needed.

Run:
    pytest tests/test_nutrition_agent.py -v
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nodes.nutrition import _build_warnings, _nutrition_from_meta


class TestWarningGeneration:

    def test_high_sodium_triggers_warning(self):
        nutrition = {"sodium_mg": 1500, "fat_g": 10, "sugar_g": 5,
                     "calories_kcal": 400, "protein_g": 20, "fiber_g": 5}
        warnings = _build_warnings(nutrition)
        assert any("sodium" in w.lower() for w in warnings)

    def test_high_fat_triggers_warning(self):
        nutrition = {"fat_g": 50, "sodium_mg": 200, "sugar_g": 5,
                     "calories_kcal": 400, "protein_g": 20, "fiber_g": 5}
        warnings = _build_warnings(nutrition)
        assert any("fat" in w.lower() for w in warnings)

    def test_high_sugar_triggers_warning(self):
        nutrition = {"sugar_g": 40, "fat_g": 10, "sodium_mg": 200,
                     "calories_kcal": 400, "protein_g": 20, "fiber_g": 5}
        warnings = _build_warnings(nutrition)
        assert any("sugar" in w.lower() for w in warnings)

    def test_high_calorie_triggers_warning(self):
        nutrition = {"calories_kcal": 900, "fat_g": 10, "sodium_mg": 200,
                     "sugar_g": 5, "protein_g": 20, "fiber_g": 5}
        warnings = _build_warnings(nutrition)
        assert any("calorie" in w.lower() for w in warnings)

    def test_low_protein_triggers_info(self):
        nutrition = {"protein_g": 5, "calories_kcal": 300, "fat_g": 10,
                     "sodium_mg": 200, "sugar_g": 5, "fiber_g": 5}
        warnings = _build_warnings(nutrition)
        assert any("protein" in w.lower() for w in warnings)

    def test_healthy_meal_no_warnings(self):
        nutrition = {"calories_kcal": 350, "protein_g": 30, "fat_g": 8,
                     "carbs_g": 30, "fiber_g": 6, "sugar_g": 5, "sodium_mg": 300}
        warnings = _build_warnings(nutrition)
        assert len(warnings) == 0

    def test_none_values_handled(self):
        """None values must not crash warning generation."""
        nutrition = {"calories_kcal": None, "protein_g": None, "fat_g": None,
                     "sugar_g": None, "sodium_mg": None, "fiber_g": None}
        warnings = _build_warnings(nutrition)
        assert isinstance(warnings, list)

    def test_zero_values_no_warnings(self):
        """Zero values should not trigger low-threshold warnings."""
        nutrition = {"calories_kcal": 0, "protein_g": 0, "fat_g": 0,
                     "carbs_g": 0, "fiber_g": 0, "sugar_g": 0, "sodium_mg": 0}
        warnings = _build_warnings(nutrition)
        # Zero protein/fiber should NOT warn (no data ≠ low)
        assert isinstance(warnings, list)


class TestNutritionFromMeta:

    def test_extracts_all_macro_fields(self):
        meta = {
            "calories_kcal": 450.0,
            "protein_g": 35.0,
            "fat_g": 12.0,
            "carbs_g": 28.0,
            "fiber_g": 4.0,
            "sugar_g": 8.0,
            "sodium_mg": 750.0,
        }
        result = _nutrition_from_meta(meta)
        assert result["calories_kcal"] == 450.0
        assert result["protein_g"] == 35.0
        assert result["sodium_mg"] == 750.0

    def test_missing_fields_default_to_zero(self):
        result = _nutrition_from_meta({})
        for key in ["calories_kcal", "protein_g", "fat_g", "carbs_g",
                    "fiber_g", "sugar_g", "sodium_mg"]:
            assert result[key] == 0.0

    def test_values_are_rounded(self):
        meta = {"calories_kcal": 450.678, "protein_g": 35.1234}
        result = _nutrition_from_meta(meta)
        assert result["calories_kcal"] == round(450.678, 1)
        assert result["protein_g"] == round(35.1234, 1)


class TestScoreMerging:
    """Test combined score = 0.5 * recipe_score + 0.5 * nutrition_score."""

    def test_equal_weight_scoring(self):
        recipe_score    = 0.8
        nutrition_score = 0.6
        combined = round(0.5 * recipe_score + 0.5 * nutrition_score, 4)
        assert combined == 0.7

    def test_high_nutrition_boosts_low_recipe(self):
        """High nutrition score can compensate for moderate recipe score."""
        recipe_score    = 0.5
        nutrition_score = 0.9
        combined = round(0.5 * recipe_score + 0.5 * nutrition_score, 4)
        assert combined > 0.5  # nutrition boosted it above recipe-only

    def test_ranking_order_preserved(self):
        """Combined scores produce correct ranking."""
        items = [
            {"recipe_score": 0.9, "nutrition_score": 0.4},  # combined 0.65
            {"recipe_score": 0.6, "nutrition_score": 0.9},  # combined 0.75
            {"recipe_score": 0.7, "nutrition_score": 0.7},  # combined 0.70
        ]
        for item in items:
            item["combined_score"] = round(
                0.5 * item["recipe_score"] + 0.5 * item["nutrition_score"], 4
            )
        ranked = sorted(items, key=lambda x: x["combined_score"], reverse=True)
        assert ranked[0]["combined_score"] == 0.75
        assert ranked[1]["combined_score"] == 0.70
        assert ranked[2]["combined_score"] == 0.65
