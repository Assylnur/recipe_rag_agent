"""
tests/test_dietary_filter.py — Unit tests for dietary restriction filtering logic.

Pure Python tests — no API, no Milvus, no LLM needed.
Fast and deterministic.

Run:
    pytest tests/test_dietary_filter.py -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nodes.recipe import _passes_restrictions


def make_recipe(ingredients: list[str]) -> dict:
    return {"meta": {"name": "Test Recipe", "ingredients": ingredients}}


class TestDietaryFilter:

    # ── Vegan ──────────────────────────────────────────────────────────────────

    def test_vegan_blocks_chicken(self):
        recipe = make_recipe(["chicken", "garlic", "soy sauce"])
        assert not _passes_restrictions(recipe, ["vegan"])

    def test_vegan_blocks_eggs(self):
        recipe = make_recipe(["eggs", "flour", "milk"])
        assert not _passes_restrictions(recipe, ["vegan"])

    def test_vegan_allows_vegetables(self):
        recipe = make_recipe(["tofu", "broccoli", "soy sauce", "garlic"])
        assert _passes_restrictions(recipe, ["vegan"])

    # ── Vegetarian ────────────────────────────────────────────────────────────

    def test_vegetarian_blocks_beef(self):
        recipe = make_recipe(["beef", "onion", "tomato"])
        assert not _passes_restrictions(recipe, ["vegetarian"])

    def test_vegetarian_allows_eggs(self):
        recipe = make_recipe(["eggs", "cheese", "spinach"])
        assert _passes_restrictions(recipe, ["vegetarian"])

    # ── Gluten-free ───────────────────────────────────────────────────────────

    def test_gluten_free_blocks_flour(self):
        recipe = make_recipe(["flour", "butter", "sugar"])
        assert not _passes_restrictions(recipe, ["gluten-free"])

    def test_gluten_free_blocks_pasta(self):
        recipe = make_recipe(["pasta", "tomato sauce", "parmesan"])
        assert not _passes_restrictions(recipe, ["gluten-free"])

    def test_gluten_free_allows_rice(self):
        recipe = make_recipe(["rice", "chicken", "vegetables"])
        assert _passes_restrictions(recipe, ["gluten-free"])

    # ── Halal ─────────────────────────────────────────────────────────────────

    def test_halal_blocks_pork(self):
        recipe = make_recipe(["pork", "garlic", "ginger"])
        assert not _passes_restrictions(recipe, ["halal"])

    def test_halal_blocks_alcohol(self):
        recipe = make_recipe(["chicken", "white wine", "herbs"])
        assert not _passes_restrictions(recipe, ["halal"])

    def test_halal_allows_chicken(self):
        recipe = make_recipe(["chicken", "onion", "spices"])
        assert _passes_restrictions(recipe, ["halal"])

    # ── Multiple restrictions ─────────────────────────────────────────────────

    def test_multiple_restrictions_all_must_pass(self):
        """Both vegan AND gluten-free must be satisfied."""
        recipe = make_recipe(["tofu", "rice", "vegetables"])
        assert _passes_restrictions(recipe, ["vegan", "gluten-free"])

    def test_multiple_restrictions_one_fails(self):
        """If one restriction fails, whole recipe is blocked."""
        recipe = make_recipe(["tofu", "flour", "vegetables"])
        assert not _passes_restrictions(recipe, ["vegan", "gluten-free"])

    # ── No restrictions ───────────────────────────────────────────────────────

    def test_no_restrictions_always_passes(self):
        recipe = make_recipe(["pork", "eggs", "flour", "wine"])
        assert _passes_restrictions(recipe, [])

    def test_empty_ingredients_always_passes(self):
        recipe = make_recipe([])
        assert _passes_restrictions(recipe, ["vegan", "halal"])

    # ── Case insensitivity ────────────────────────────────────────────────────

    def test_restriction_case_insensitive(self):
        """Restriction names should work regardless of case."""
        recipe = make_recipe(["beef", "onion"])
        assert not _passes_restrictions(recipe, ["Vegan"])
        assert not _passes_restrictions(recipe, ["VEGETARIAN"])
