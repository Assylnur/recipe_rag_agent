"""
ingest_recipes.py — Ingest enriched recipe corpus into two Milvus collections.

    recipes_main       ← Recipe Agent
    nutrition_profiles ← Nutrition Agent

Usage:
    python ingest_recipes.py
    python ingest_recipes.py --drop
    python ingest_recipes.py --only recipes
    python ingest_recipes.py --only nutrition
    python ingest_recipes.py --model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
"""

import argparse
import json
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer

from milvus_connector import (
    COLLECTION_NUTRITION,
    COLLECTION_RECIPES,
    MilvusRecipeConnector,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_MODEL  = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_CORPUS = Path("data/raw/enriched_corpus.json")


# ── Text builders ──────────────────────────────────────────────────────────────

def build_recipe_text(meal: dict) -> str:
    """Semantic text for Recipe Agent — cuisine identity + ingredient composition."""
    ingredients = ", ".join(ing["ingredient"] for ing in meal.get("ingredients", []))
    parts = [
        meal.get("name", ""),
        meal.get("category", ""),
        meal.get("area", ""),
        f"ingredients: {ingredients}" if ingredients else "",
        " ".join(meal.get("tags", [])),
    ]
    return " | ".join(p for p in parts if p).strip()


def build_nutrition_text(meal: dict) -> str:
    """
    Natural-language macro description for Nutrition Agent.
    Qualitative labels make it match health goal queries like 'high protein low carb'.
    """
    n = meal.get("nutrition_totals_estimate", {})

    def v(key: str) -> float:
        return round(float(n.get(key) or 0.0), 1)

    calories = v("calories_kcal")
    protein  = v("protein_g")
    fat      = v("fat_g")
    carbs    = v("carbs_g")
    fiber    = v("fiber_g")
    sugar    = v("sugar_g")
    sodium   = v("sodium_mg")

    cal_label     = "low calorie"    if calories < 300 else ("high calorie"    if calories > 700  else "moderate calorie")
    protein_label = "high protein"   if protein  > 25  else ("low protein"     if protein  < 10   else "moderate protein")
    carb_label    = "low carb"       if carbs    < 20  else ("high carb"       if carbs    > 60   else "moderate carb")
    fat_label     = "low fat"        if fat      < 10  else ("high fat"        if fat      > 30   else "moderate fat")
    fiber_label   = "high fiber"     if fiber    > 8   else ("low fiber"       if fiber    < 3    else "moderate fiber")
    sodium_label  = "low sodium"     if sodium   < 400 else ("high sodium"     if sodium   > 1000 else "moderate sodium")

    return (
        f"{calories} kcal, {protein}g protein, {fat}g fat, {carbs}g carbs, "
        f"{fiber}g fiber, {sugar}g sugar, {sodium}mg sodium. "
        f"{cal_label}, {protein_label}, {carb_label}, {fat_label}, {fiber_label}, {sodium_label}."
    )


# ── Meta builders ──────────────────────────────────────────────────────────────

def build_recipe_meta(meal: dict) -> dict:
    return {
        "meal_id":     meal["id"],
        "name":        meal.get("name", ""),
        "category":    meal.get("category", ""),
        "area":        meal.get("area", ""),
        "tags":        meal.get("tags", []),
        "ingredients": [ing["ingredient"] for ing in meal.get("ingredients", [])],
        "thumbnail":   meal.get("thumbnail", ""),
        "youtube":     meal.get("youtube", ""),
    }


def build_nutrition_meta(meal: dict) -> dict:
    n = meal.get("nutrition_totals_estimate", {})
    return {
        "meal_id":       meal["id"],
        "name":          meal.get("name", ""),
        "calories_kcal": float(n.get("calories_kcal") or 0.0),
        "protein_g":     float(n.get("protein_g")     or 0.0),
        "fat_g":         float(n.get("fat_g")         or 0.0),
        "carbs_g":       float(n.get("carbs_g")       or 0.0),
        "fiber_g":       float(n.get("fiber_g")       or 0.0),
        "sugar_g":       float(n.get("sugar_g")       or 0.0),
        "sodium_mg":     float(n.get("sodium_mg")     or 0.0),
    }


# ── Ingest ─────────────────────────────────────────────────────────────────────

def ingest(
    connector: MilvusRecipeConnector,
    meals: list[dict],
    text_fn,
    meta_fn,
    batch_size: int,
    label: str,
) -> None:
    existing = connector.get_existing_meal_ids()
    pending  = [m for m in meals if m["id"] not in existing]
    log.info("[%s] Skipping %d existing | Ingesting %d new", label, len(existing), len(pending))

    if not pending:
        log.info("[%s] Nothing to do.", label)
        return

    texts = [text_fn(m) for m in pending]
    metas = [meta_fn(m) for m in pending]
    connector.add_documents(texts, metas, batch_size=batch_size)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest recipe corpus into Milvus.")
    parser.add_argument("--corpus",  type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model",   default=DEFAULT_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--batch",   type=int, default=512)
    parser.add_argument("--drop",    action="store_true", help="Drop both collections before ingesting")
    parser.add_argument("--only",    choices=["recipes", "nutrition"], help="Ingest only one collection")
    args = parser.parse_args()

    log.info("Loading corpus from %s", args.corpus)
    meals = json.loads(args.corpus.read_text(encoding="utf-8"))
    log.info("Loaded %d meals", len(meals))

    log.info("Loading embedding model: %s", args.model)
    model = SentenceTransformer(args.model)
    dim   = model.get_sentence_embedding_dimension()
    log.info("Embedding dim: %d", dim)

    if args.only != "nutrition":
        from pymilvus import utility
        if args.drop and utility.has_collection(COLLECTION_RECIPES):
            from pymilvus import Collection
            Collection(COLLECTION_RECIPES).drop()
            log.warning("Dropped collection '%s'", COLLECTION_RECIPES)

        col_recipes = MilvusRecipeConnector(COLLECTION_RECIPES, model, dim=dim)
        ingest(col_recipes, meals, build_recipe_text, build_recipe_meta, args.batch, "recipes_main")

    if args.only != "recipes":
        from pymilvus import utility
        if args.drop and utility.has_collection(COLLECTION_NUTRITION):
            from pymilvus import Collection
            Collection(COLLECTION_NUTRITION).drop()
            log.warning("Dropped collection '%s'", COLLECTION_NUTRITION)

        col_nutrition = MilvusRecipeConnector(COLLECTION_NUTRITION, model, dim=dim)
        ingest(col_nutrition, meals, build_nutrition_text, build_nutrition_meta, args.batch, "nutrition_profiles")


if __name__ == "__main__":
    main()