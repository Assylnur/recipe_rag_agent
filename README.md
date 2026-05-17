# 🍳 Recipe RAG Agent

A multi-agent recipe recommendation system built with LangGraph, Milvus, and FastAPI. The system combines semantic search over a curated recipe corpus with real-time YouTube tutorial discovery via MCP.

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│ Ingredient Agent│  Parses intent, dietary restrictions, health goals
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Recipe Agent   │  Semantic search over recipes_main (Milvus)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Nutrition Agent │  Re-ranks by nutrition_profiles (Milvus)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ YouTube Agent   │  Fetches tutorials via YouTube MCP server
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Responder     │  Generates final answer with sources
└─────────────────┘
```

## Tech Stack

| Component | Technology | Reason |
|---|---|---|
| Agent orchestration | LangGraph | State machine with typed state, SSE streaming support |
| Vector store | Milvus | Two collections: semantic recipes + nutrition profiles |
| Embeddings | sentence-transformers/all-mpnet-base-v2 | Free, local, 768-dim, strong English performance |
| LLM | vLLM (local) | No API cost, full control, streaming support |
| MCP server | FastAPI + SSE | YouTube Data API v3 integration |
| API | FastAPI | Async, SSE streaming, rate limiting |
| UI | Streamlit | Fast to build, supports streaming |
| Data | TheMealDB + USDA FDC | Free APIs, 608 meals, full nutrition data |

## Project Structure

```
api/
├── server.py              # FastAPI backend
├── graph.py               # LangGraph StateGraph
├── state.py               # RecipeAgentState TypedDict
├── config.py              # Model + collection config
├── security.py            # Input validation, rate limiting, content filtering
├── cache.py               # Query cache + YouTube quota tracker
├── rag_quality.py         # Retrieval confidence, hallucination detection, source attribution
├── logger.py              # Audit logger
├── dashboard.py           # Streamlit metrics dashboard
├── ui.py                  # Streamlit chat UI
├── nodes/
│   ├── _db.py             # Milvus search connector singletons
│   ├── ingredient.py      # Ingredient Agent
│   ├── recipe.py          # Recipe Agent
│   ├── nutrition.py       # Nutrition Agent
│   ├── youtube.py         # YouTube Agent (MCP client)
│   └── responder.py       # Responder node
├── api/
│   └── db_connector.py    # Search-only Milvus connector
├── db/
│   └── milvus_connector.py  # Ingest connector
├── mcp_servers/
│   └── youtube_mcp.py     # YouTube MCP server
├── data_preparation/
│   ├── fetch_meals.py     # TheMealDB scraper
│   ├── fetch_nutrition.py # USDA FDC enrichment
│   └── ingest_recipes.py  # Milvus ingestion pipeline
└── tests/
    ├── conftest.py
    ├── test_dietary_filter.py
    ├── test_nutrition_agent.py
    ├── test_retrieval.py
    ├── test_ingredient_agent.py
    ├── test_mcp_youtube.py
    ├── test_pipeline.py
    ├── test_security.py
    ├── test_rag_quality.py
    └── test_cache.py
```


## Data Pipeline

```
data_preparation/
├── fetch_meals.py        # TheMealDB scraper — fetches all 608 meals across 14 categories
├── fetch_nutrition.py    # USDA FDC enrichment — looks up nutrition for 841 unique ingredients
├── ingest_recipes.py     # Milvus ingestion — embeds and loads both collections
└── milvus_connector.py   # Write-only connector used during ingestion

data/
└── raw/
    ├── meals/                  # one JSON file per meal (52772.json, etc.)
    ├── categories.json         # 14 TheMealDB categories
    ├── meal_index.json         # flat list of all meal stubs
    ├── corpus.json             # 608 meals combined
    ├── nutrition_cache.json    # USDA lookup cache — 841 ingredients
    └── enriched_corpus.json    # final corpus with nutrition — used for RAG
```

### Step 1 — Fetch meals from TheMealDB (free, no key)

```bash
python data_preparation/fetch_meals.py --out data/raw
# ~3 minutes, fetches 608 meals across 14 categories
# Saves per-meal JSON + combined corpus.json
```

### Step 2 — Enrich with USDA nutrition (free key required)

```bash
python data_preparation/fetch_nutrition.py --key $USDA_API_KEY
# ~2 minutes, 841 unique ingredients → USDA FoodData Central
# Results cached in nutrition_cache.json — safe to re-run
# Output: enriched_corpus.json with macros per ingredient + meal totals
```

### Step 3 — Ingest into Milvus

```bash
python data_preparation/ingest_recipes.py
# Creates two collections:
#   recipes_main       — embeds "name | category | area | ingredients | tags"
#   nutrition_profiles — embeds "450 kcal, 35g protein, high protein, low carb..."
# Resume-safe: skips already-ingested meal_ids
```

### Data quality notes

- **USDA match rate:** ~94% of ingredients matched (790/841). Unmatched ingredients (exotic spices, brand names) stored as null and excluded from nutrition totals.
- **Nutrition totals are estimates.** Each ingredient treated as ~100g regardless of actual measure. Suitable for relative comparison, not precise calorie counting.
- **Dual embedding strategy:** Recipe text captures cuisine identity and ingredient composition. Nutrition text captures macro profile in natural language so health-goal queries like "high protein low carb" match semantically.

## Setup

### Prerequisites

- Docker + Docker Compose
- Milvus running on `172.17.0.1:19530`
- vLLM running on `host.docker.internal:8080`
- YouTube Data API v3 key (free at console.cloud.google.com)

### 1. Environment

```bash
cp .env.example .env
# Edit .env and set:
# YOUTUBE_API_KEY=your_key_here
# VLLM_URL=http://host.docker.internal:8080/v1
# MODEL_NAME=your-model-name
```

### 2. Data preparation

```bash
# Collect recipes from TheMealDB
python data_preparation/fetch_meals.py --out data/raw

# Enrich with USDA nutrition data
python data_preparation/fetch_nutrition.py --key $USDA_API_KEY

# Ingest into Milvus (both collections)
python data_preparation/ingest_recipes.py
```

### 3. Start services

```bash
# Start YouTube MCP server (host)
YOUTUBE_API_KEY=your_key python mcp_servers/youtube_mcp.py &

# Start all containers
docker compose up -d
```

### 4. Access

| Service | URL |
|---|---|
| Chat UI | http://localhost:8541 |
| Metrics dashboard | http://localhost:8542 |
| API docs | http://localhost:8041/docs |
| YouTube MCP health | http://localhost:8801/health |

## Running Tests

```bash
# Fast unit tests (no API needed)
pytest tests/test_dietary_filter.py tests/test_nutrition_agent.py tests/test_security.py tests/test_cache.py tests/test_rag_quality.py -v

# RAG quality tests (needs Milvus)
pytest tests/test_retrieval.py -v

# Full pipeline tests (needs running backend)
TEST_API_URL=http://localhost:8010 pytest tests/test_pipeline.py -v

# MCP tests (needs youtube_mcp running)
pytest tests/test_mcp_youtube.py -v

# All tests
pytest tests/ -v
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/recommend` | Blocking recommendation |
| POST | `/recommend_stream` | SSE streaming recommendation |
| POST | `/feedback` | Submit thumbs up/down rating |
| GET | `/metrics` | Aggregated performance metrics |
| GET | `/health` | Liveness check |

## Data

- **608 meals** from TheMealDB (14 categories)
- **841 unique ingredients** enriched with USDA FDC nutrition data
- **Two Milvus collections**: `recipes_main` (semantic) + `nutrition_profiles` (nutritional)
- Corpus stored at `data/raw/enriched_corpus.json`