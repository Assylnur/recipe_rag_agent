"""
fetch_nutrition.py — Enrich meal corpus with USDA FoodData Central nutrition data.

For every unique ingredient in corpus.json, queries USDA FDC and caches the
result locally. Then writes enriched_corpus.json with per-ingredient nutrition
(per 100g) and a meal-level summary of key macros.

API key is passed via X-Api-Key header per api.data.gov spec.
Rate limit: 1000 req/hour. 429 responses are retried with exponential backoff.

Get your free API key: https://api.nal.usda.gov/

Usage:
    python fetch_nutrition.py --key YOUR_API_KEY
    python fetch_nutrition.py --key YOUR_API_KEY --corpus data/raw/corpus.json --out data/raw
"""

import argparse
import json
import logging
import time
from pathlib import Path

import requests

USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# Nutrients we care about (USDA nutrient IDs)
NUTRIENT_IDS = {
    1008: "calories_kcal",
    1003: "protein_g",
    1004: "fat_g",
    1005: "carbs_g",
    1079: "fiber_g",
    2000: "sugar_g",
    1093: "sodium_mg",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── USDA API ───────────────────────────────────────────────────────────────────

def search_usda(ingredient: str, api_key: str, retries: int = 3) -> dict | None:
    """
    Query USDA FDC for an ingredient and return nutrient values per 100g.

    - API key sent via X-Api-Key header (api.data.gov standard).
    - Retries up to `retries` times on HTTP 429 with exponential backoff.
    - Returns None if no match found or all retries exhausted.
    """
    # USDA FDC uses api_key query param, not X-Api-Key header
    headers = {}
    params = {
        "query": ingredient,
        "api_key": api_key,
        "pageSize": 5,
    }

    foods = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(USDA_SEARCH_URL, params=params, timeout=10)
        except requests.RequestException as e:
            log.warning("Network error for '%s': %s", ingredient, e)
            return None

        if resp.status_code == 429:
            wait = 2 ** attempt
            log.warning(
                "Rate limited (429) on attempt %d/%d — waiting %ds...",
                attempt, retries, wait,
            )
            time.sleep(wait)
            continue

        if not resp.ok:
            log.warning("HTTP %d for '%s': %s", resp.status_code, ingredient, resp.text[:120])
            return None

        foods = resp.json().get("foods", [])
        break  # successful response — exit retry loop

    if foods is None:
        log.warning("All %d retries exhausted for '%s'", retries, ingredient)
        return None

    if not foods:
        log.debug("No USDA result for: %s", ingredient)
        return None

    # Use first result — USDA sorts by relevance
    food = foods[0]
    nutrients_raw = {n["nutrientId"]: n.get("value", 0.0) for n in food.get("foodNutrients", [])}

    nutrition = {
        "fdc_id": food.get("fdcId"),
        "description": food.get("description", ""),
        "data_type": food.get("dataType", ""),
    }
    for nid, key in NUTRIENT_IDS.items():
        nutrition[key] = round(nutrients_raw.get(nid, 0.0), 2)

    return nutrition


# ── Cache ──────────────────────────────────────────────────────────────────────

def load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict, cache_path: Path) -> None:
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Enrichment ────────────────────────────────────────────────────────────────

def enrich_meal(meal: dict, cache: dict) -> dict:
    """
    Attach USDA nutrition to each ingredient and compute rough meal-level totals.

    Note: totals treat each ingredient as ~100g — we don't parse free-text
    measures to exact grams. Use as relative comparison, not precise counts.
    """
    enriched_ingredients = []
    totals = {key: 0.0 for key in NUTRIENT_IDS.values()}

    for ing in meal.get("ingredients", []):
        name_lower = ing["ingredient"].lower()
        nutrition = cache.get(name_lower)

        enriched = {**ing}
        if nutrition:
            enriched["nutrition_per_100g"] = {
                k: v for k, v in nutrition.items()
                if k not in ("fdc_id", "description", "data_type")
            }
            enriched["usda_match"] = nutrition["description"]
            for key in NUTRIENT_IDS.values():
                totals[key] = round(totals[key] + nutrition.get(key, 0.0), 2)
        else:
            enriched["nutrition_per_100g"] = None
            enriched["usda_match"] = None

        enriched_ingredients.append(enriched)

    return {
        **meal,
        "ingredients": enriched_ingredients,
        "nutrition_totals_estimate": totals,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Enrich meal corpus with USDA FDC nutrition data.")
    parser.add_argument("--key", required=True,
                        help="USDA FDC API key — get free at https://api.nal.usda.gov/")
    parser.add_argument("--corpus", type=Path, default=Path("data/raw/corpus.json"),
                        help="Input corpus JSON (default: data/raw/corpus.json)")
    parser.add_argument("--out", type=Path, default=Path("data/raw"),
                        help="Output directory (default: data/raw)")
    parser.add_argument("--delay", type=float, default=0.15,
                        help="Seconds between requests (default: 0.15 → ~400 req/min, well under 1000/hr)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / "nutrition_cache.json"

    # Load corpus
    log.info("Loading corpus from %s", args.corpus)
    meals = json.loads(args.corpus.read_text(encoding="utf-8"))
    log.info("Loaded %d meals", len(meals))

    # Collect unique ingredients
    all_ingredients: set[str] = set()
    for meal in meals:
        for ing in meal.get("ingredients", []):
            name = ing["ingredient"].strip().lower()
            if name:
                all_ingredients.add(name)
    log.info("Unique ingredients to look up: %d", len(all_ingredients))

    # Load cache; fetch only what's missing
    cache = load_cache(cache_path)
    missing = [i for i in all_ingredients if i not in cache]
    log.info("Cached: %d | To fetch: %d", len(cache), len(missing))

    for idx, ingredient in enumerate(missing, 1):
        cache[ingredient] = search_usda(ingredient, args.key)

        if idx % 20 == 0:
            save_cache(cache, cache_path)
            log.info("Progress: %d / %d", idx, len(missing))

        time.sleep(args.delay)

    save_cache(cache, cache_path)

    found = sum(1 for v in cache.values() if v is not None)
    log.info(
        "Done. USDA match rate: %d / %d (%.1f%%) — cache -> %s",
        found, len(cache), 100 * found / max(len(cache), 1), cache_path,
    )

    # Enrich and save corpus
    log.info("Enriching corpus...")
    enriched = [enrich_meal(meal, cache) for meal in meals]

    out_path = args.out / "enriched_corpus.json"
    out_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Enriched corpus -> %s (%d meals)", out_path, len(enriched))


if __name__ == "__main__":
    main()