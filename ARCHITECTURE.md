# Architecture Blueprint — Recipe RAG Agent

## System Overview

The Recipe RAG Agent is a multi-agent system that answers natural language recipe requests by combining semantic search over a curated corpus with nutritional re-ranking and live YouTube tutorial discovery.

## Agent Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                        User Request                          │
│     "High protein chicken dinner, gluten-free please"        │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                   INGREDIENT AGENT                           │
│                                                              │
│  Input:  raw user message                                    │
│  LLM:    structured output → IngredientAgentOutput           │
│  Output: {                                                   │
│    available_ingredients: ["chicken"]                        │
│    dietary_restrictions:  ["gluten-free"]                    │
│    health_goal:           "high protein"                     │
│    recipe_query:          "high protein chicken gluten-free" │
│    nutrition_query:       "high protein low carb"            │
│  }                                                           │
└─────────────────────────┬────────────────────────────────────┘
                          │  recipe_query + user_intent
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                    RECIPE AGENT                              │
│                                                              │
│  Input:  recipe_query, dietary_restrictions                  │
│  Search: Milvus recipes_main (COSINE, top 20)               │
│  Filter: dietary restriction post-filter (Python)            │
│  QA:     check_retrieval_confidence() → log precision        │
│  Output: candidate_recipes (top 10 meal_ids + scores)        │
└─────────────────────────┬────────────────────────────────────┘
                          │  candidate_recipes
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                  NUTRITION AGENT                             │
│                                                              │
│  Input:  nutrition_query, candidate_ids from Recipe Agent    │
│  Search: Milvus nutrition_profiles (filtered to candidates)  │
│  Score:  combined = 0.5 × recipe_score + 0.5 × nutrition     │
│  Enrich: attach macros, generate health warnings             │
│  Output: final_recipes (top 5, re-ranked)                    │
└─────────────────────────┬────────────────────────────────────┘
                          │  final_recipes
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                   YOUTUBE AGENT (MCP)                        │
│                                                              │
│  Input:  final_recipes (top recipe names)                    │
│  Call:   YouTube MCP server → search_youtube() [1 API call]  │
│  Result: top 3 tutorial videos                               │
│  Fallback: TheMealDB corpus YouTube links if quota exceeded  │
│  Output: video_links attached to final_recipes               │
└─────────────────────────┬────────────────────────────────────┘
                          │  final_recipes + video_links
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                     RESPONDER                                │
│                                                              │
│  Input:  final_recipes, video_links, user_intent             │
│  Loads:  full instructions from enriched_corpus.json         │
│  LLM:    generate recommendation (streaming)                 │
│  Append: source attribution (TheMealDB #ID, relevance %)     │
│  Check:  detect_hallucination() → log warnings               │
│  Output: final_answer                                        │
└──────────────────────────────────────────────────────────────┘
```

## Data Architecture

### Two Milvus Collections

**recipes_main** — used by Recipe Agent
```
Embedded text: "Teriyaki Chicken | Chicken | Japanese | ingredients: soy sauce, ginger..."
Schema: {meal_id, name, category, area, tags, ingredients, embedding[768]}
Purpose: semantic similarity to user's dish/ingredient/cuisine query
```

**nutrition_profiles** — used by Nutrition Agent
```
Embedded text: "450 kcal, 35g protein, 12g fat. high protein, low carb, moderate sodium."
Schema: {meal_id, name, calories_kcal, protein_g, fat_g, carbs_g, fiber_g, sugar_g, sodium_mg, embedding[768]}
Purpose: semantic similarity to user's health goal query
```

**Why two collections instead of one?**
Each agent needs a different semantic space. Recipe Agent needs cuisine/ingredient similarity. Nutrition Agent needs health-goal similarity. Mixing them in one collection would make neither search work well.

### Corpus (enriched_corpus.json)
Full meal data kept out of Milvus (field size limits). Loaded into memory at startup. Accessed by Responder for instructions, thumbnails, YouTube links.

## Data Collection Pipeline

```
TheMealDB API (free)          USDA FoodData Central (free key)
       │                               │
       ▼                               ▼
fetch_meals.py              fetch_nutrition.py
  14 categories               841 unique ingredients
  608 meals                   ~94% match rate
  per-meal JSON               nutrition_cache.json (resumable)
       │                               │
       └──────────┬────────────────────┘
                  ▼
          enriched_corpus.json
          608 meals × 7 macros per ingredient
                  │
                  ▼
          ingest_recipes.py
          SentenceTransformer (all-mpnet-base-v2)
                  │
         ┌────────┴────────┐
         ▼                 ▼
   recipes_main     nutrition_profiles
   (608 vectors)    (608 vectors)
   cuisine+ingr.    macro profile
   embedding        embedding
```

### Why two different embedding texts

**recipes_main** embeds:
```
"Teriyaki Chicken | Chicken | Japanese | ingredients: soy sauce, ginger, honey | Meat Casserole"
```
This captures *what the dish is* — good for matching "Asian chicken with soy sauce".

**nutrition_profiles** embeds:
```
"450 kcal, 35g protein, 12g fat, 28g carbs. high protein, low carb, moderate calorie, low sodium."
```
This captures *how healthy it is* — good for matching "high protein low carb diet".

Same meal, two semantic representations, two collections. Neither works for the other's query type.

## MCP Integration

```
Backend Container          Host Process
       │                       │
       │  POST /messages        │
       │  {"method":           │
       │   "tools/call",       │
       │   "name":             │
       │   "search_youtube"}   │
       │──────────────────────►│
       │                       │  GET googleapis.com/youtube/v3/search
       │                       │──────────────────────────────────────►
       │                       │◄──────────────────────────────────────
       │  SSE response         │
       │  data: {"result":     │
       │   {"content": [...]}} │
       │◄──────────────────────│
```

YouTube MCP server implements the MCP protocol over SSE (Server-Sent Events). It exposes one tool: `search_youtube(query, max_results)`. The backend calls it via JSON-RPC 2.0.

**Why MCP instead of direct API call?**
MCP decouples the YouTube integration from the agent logic. The agent doesn't know or care about YouTube's API — it just calls a tool. The MCP server can be swapped for Vimeo, Dailymotion, or a mock without changing agent code.

## Security Architecture

```
Request → Rate Limiter (20/min/IP)
       → Input Sanitizer (strip HTML, null bytes, truncate)
       → Content Filter (block weapons/self-harm, log jailbreaks)
       → Agent Pipeline
       → Response
```

**Design decision:** Jailbreak and SQL injection attempts are sanitized and logged but not hard-blocked. The LLM's own safety training handles them more effectively than regex. We only hard-block two categories: weapons and self-harm.

## Observability Stack

```
Agent Nodes
    │
    ├── node_logger.py    → node_timings.jsonl (per-node duration)
    ├── llm_logger.py     → llm_calls.jsonl (tokens, TTFT, cost)
    └── logger.py         → recipe_rag_audit.jsonl (per-request audit)

User Feedback
    └── POST /feedback    → feedback.jsonl (ratings)

Quota Tracking
    └── QuotaTracker      → youtube_quota.json (daily API usage)

Dashboard
    └── GET /metrics      → aggregates all files → Streamlit dashboard
```

## Cost Analysis

| Component | Cost |
|---|---|
| vLLM (local) | $0 — runs on existing GPU |
| Milvus (local) | $0 — runs on existing server |
| Embeddings (local) | $0 — sentence-transformers |
| TheMealDB | $0 — free public API |
| USDA FDC | $0 — free government API |
| YouTube Data API v3 | $0 — 10,000 units/day free (100 searches) |
| **Total** | **$0/month** |

## Technology Rationale

**LangGraph over raw Python:**
State machine with typed state prevents data leakage between turns. Built-in SSE streaming support. Easy to add/remove nodes without restructuring code.

**Milvus over ChromaDB/FAISS:**
Production-grade, supports filtering expressions (`meta["meal_id"] in [...]`), handles 600+ documents easily, consistent with existing infrastructure.

**Two-stage search (semantic + nutritional):**
Single-collection search can't simultaneously optimize for "tastes like chicken teriyaki" and "high protein low carb." Two collections let each agent specialize.

**sentence-transformers over OpenAI embeddings:**
Free, local, no API dependency, 768-dim vectors are sufficient for 608 meals.

**FastAPI SSE over WebSocket:**
SSE is unidirectional (server → client) which matches our use case. Simpler than WebSocket, works through most proxies, native browser support.