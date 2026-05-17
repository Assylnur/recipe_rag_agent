# Executive Summary — Recipe RAG Agent

## Project Overview

The Recipe RAG Agent is an AI-powered recipe recommendation system that answers natural language food queries by combining semantic search, nutritional analysis, and real-time video discovery. The system is built as a multi-agent pipeline using LangGraph, deployed locally with zero recurring API costs.

**Target users:** Home cooks who want personalized recipe suggestions based on available ingredients, dietary restrictions, and health goals.

## Problem Statement

Existing recipe platforms (Allrecipes, Yummly) require users to navigate category trees and apply filters manually. They don't understand natural language like "high protein gluten-free dinner with what I have in my fridge." Users with specific dietary restrictions (halal, vegan, low-sodium) must cross-reference multiple sources. Recipe videos are linked but often stale or irrelevant.

## Solution

A conversational AI agent that understands intent, not just keywords:

| User says | System does |
|---|---|
| "I have chicken, garlic, soy sauce" | Extracts ingredients, searches semantically |
| "high protein, gluten-free" | Filters by restriction, re-ranks by nutrition |
| "something Asian" | Applies cuisine preference to vector search |
| Returns answer | Includes nutrition data, health warnings, fresh YouTube tutorials |

## Key Technical Achievements

**Multi-agent pipeline:** Four specialized agents collaborate through shared state — Ingredient Agent parses intent, Recipe Agent retrieves candidates, Nutrition Agent re-ranks by health fit, YouTube Agent adds tutorials via MCP.

**Dual RAG collections:** Two separate Milvus vector stores — one for semantic recipe similarity, one for nutritional profile similarity — enable simultaneous optimization for taste preference and health goals.

**MCP integration:** YouTube tutorial discovery via a custom MCP server implementing the JSON-RPC 2.0 protocol over SSE, with TheMealDB corpus links as fallback when API quota is exceeded.

**Zero cloud cost:** vLLM, Milvus, and sentence-transformers run locally. Only external dependency is YouTube Data API (100 free searches/day).

## Data Quality

- **608 meals** from TheMealDB covering 14 cuisine categories
- **841 unique ingredients** enriched with USDA FoodData Central nutrition data (official US government source)
- **Dual embedding strategy:** Recipe text embeds cuisine+ingredient identity; nutrition text embeds macro profile in natural language for health-goal matching
- **Dietary filter coverage:** 7 restriction types (vegan, vegetarian, gluten-free, halal, kosher, dairy-free, nut-free) with ingredient-level signal matching

## Test Results

**96/96 tests passing** across 7 test suites:

| Suite | Tests | Coverage |
|---|---|---|
| Dietary filter | 16 | Vegan, halal, gluten-free, multi-restriction logic |
| Nutrition agent | 14 | Warning thresholds, score merging, macro extraction |
| RAG retrieval | 13 | Milvus search accuracy, score ranges, field validation |
| Security | 31 | Input sanitization, content filtering, rate limiting |
| RAG quality | 17 | Confidence scoring, hallucination detection, source attribution |
| Cache | 17 | LRU eviction, TTL expiry, quota persistence |
| MCP YouTube | 9 | JSON-RPC protocol, video fields, quota enforcement |
| Pipeline E2E | 24 | 12 positive + 12 negative/adversarial scenarios |

## Non-Functional Compliance

| Requirement | Implementation |
|---|---|
| Observability | node_logger, llm_logger, audit log, Streamlit dashboard |
| Security | Rate limiting (20/min), input sanitization, content filtering, API key masking |
| RAG quality | Confidence thresholds, hallucination detection, source attribution |
| Caching | In-memory LRU query cache (1hr TTL), USDA nutrition cache |
| Quota management | Daily YouTube API counter with 80% warning and hard block |
| Graceful degradation | TheMealDB fallback links when YouTube quota exhausted |

## Business Value

**For users:** Natural language interface removes the friction of filter navigation. Dietary restrictions are respected automatically. Nutritional transparency builds trust. Video tutorials reduce cooking failure rate.

**For operators:** Zero cloud cost makes the system economically viable at any scale. Local-first architecture means no data leaves the server. Modular agent design allows adding new agents (meal planning, grocery list, cost estimation) without restructuring the pipeline.

**Market opportunity:** The global recipe app market is valued at $300M+ annually. Differentiation through AI-native conversational interface and nutrition-aware recommendations addresses an underserved segment of health-conscious home cooks.

## Trade-offs and Limitations

**Nutrition totals are estimates.** USDA lookup treats each ingredient as ~100g regardless of actual measure. Precise calculation would require NLP-based quantity parsing (future work).

**YouTube quota is 100 searches/day** on the free tier. The system uses 1 API call per user query (not per recipe) and falls back to TheMealDB corpus links when exhausted.

**Recipe corpus is static.** TheMealDB data was collected at ingestion time. New recipes require re-running the data pipeline. A scheduled re-ingestion job would address this.

**English-only optimization.** The embedding model is optimized for English. Russian/Kazakh queries work but with lower retrieval precision.

## Conclusion

The Recipe RAG Agent demonstrates that a production-quality multi-agent RAG system can be built at zero operational cost using open-source components. The architecture is extensible, observable, and secure — ready for further development into a full meal planning platform.
