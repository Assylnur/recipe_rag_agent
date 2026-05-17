# Self-Review — Architecture Decisions and Trade-offs

## What I built and why

### Decision 1: Two Milvus collections instead of one

**What I did:** Created `recipes_main` (semantic recipe search) and `nutrition_profiles` (nutrition-based re-ranking) as separate collections with different embedding texts.

**Why:** A single collection can't simultaneously serve two different semantic spaces. "Teriyaki Chicken | Japanese | soy sauce, ginger" and "450 kcal, 35g protein, high protein, low carb" need different embeddings because they answer different questions. Mixing them in one collection would force a compromise that serves neither agent well.

**Trade-off:** Two ingest pipelines, two search calls, slightly more complexity. Worth it because the Nutrition Agent's re-ranking is the core differentiator of this system.

---

### Decision 2: Sequential pipeline instead of parallel fan-out

**What I did:** `ingredient → recipe → nutrition → youtube → responder` runs sequentially.

**Why:** Each agent's output is the next agent's input. Recipe Agent's `candidate_ids` are required by Nutrition Agent's filtered search. This isn't parallelizable without architectural changes. In my Legal RAG project I used parallel fan-out for independent branches — here the dependency chain is strict.

**Trade-off:** Total latency = sum of all agent latencies (~30s). Parallel execution isn't possible given the dependency structure, but caching frequent queries mitigates this for repeated requests.

---

### Decision 3: LangGraph over raw asyncio

**What I did:** Used LangGraph's `StateGraph` with a `TypedDict` state.

**Why:** LangGraph gives typed state that prevents agents from accidentally reading stale data from previous turns. `reset_per_turn_state()` explicitly clears scratch fields on every new request. This solved a class of bugs I encountered in early development where the Nutrition Agent read stale `candidate_recipes` from a previous query.

**Trade-off:** LangGraph adds overhead and a learning curve. For a simple sequential pipeline it's arguably overkill — plain asyncio would work. I chose it for the streaming support via `astream_events()` and for consistency with the existing Legal RAG codebase.

---

### Decision 4: Corpus JSON in memory instead of PostgreSQL

**What I did:** `enriched_corpus.json` is loaded into memory at startup by `responder.py`. Full meal details (instructions, thumbnail, youtube) are looked up by `meal_id`.

**Why:** 608 meals × ~5KB each ≈ 3MB. Trivial for memory. Avoids the PostgreSQL + PgBouncer setup that the Legal RAG system requires for its chunk store. The requirement for this project explicitly said no PostgreSQL needed.

**Trade-off:** Memory is cleared on restart. Corpus updates require restarting the container. For 608 records this is acceptable. At 100K+ recipes, PostgreSQL or a key-value store would be necessary.

---

### Decision 5: 1 YouTube API call per query, not per recipe

**Initial implementation:** Called YouTube once per recipe in `final_recipes` — 5 calls per user query. Burned through 100 search quota in ~20 queries during testing.

**Fix:** One call for the top recipe name, distribute top 3 results across recipes, fall back to TheMealDB corpus links for the rest.

**Lesson:** API quota management must be designed upfront, not retrofitted. The quota tracker (`QuotaTracker`) persists counts to disk so they survive container restarts, which the in-memory approach missed.

---

### Decision 6: Content filter allows jailbreaks through

**What I did:** `security.py` hard-blocks weapons and self-harm requests but lets jailbreak/SQL injection attempts pass through (sanitized and logged).

**Why:** Regex-based jailbreak detection is an arms race you can't win. Patterns like "ignore previous instructions" have thousands of variations. The LLM's own RLHF/safety training is more robust than any regex. My job is to log the attempt for audit purposes and sanitize the input — not to out-regex sophisticated adversarial prompts.

**Trade-off:** A sophisticated jailbreak might succeed at the LLM layer. Mitigated by the fact that this is a recipe recommendation system — the blast radius of a successful jailbreak is "gives a recipe for something weird," not "executes code."

---

### Decision 7: Dietary restriction filter in Python, not Milvus

**What I did:** `_passes_restrictions()` in `recipe_node.py` filters results after Milvus search using a keyword-to-ingredient lookup table.

**Why:** Milvus JSON field filtering with `meta["ingredients"] like "%pork%"` is possible but slow and requires knowing every synonym. A Python post-filter over 20 results is faster and more maintainable. The lookup table (`_RESTRICTION_SIGNALS`) is explicit, testable, and easy to extend.

**Trade-off:** We over-fetch from Milvus (top 20) and filter down. If all 20 results are blocked (e.g. very strict restrictions), the agent returns no results. A smarter approach would expand the search window dynamically.

---


---

### Decision 8: Application observability vs infrastructure monitoring

**What I did:** Split observability into two layers. Application-level metrics (token usage, node timing, request audit, user feedback, API quota) are handled by four dedicated loggers writing JSONL to `logs/` — `llm_logger.py`, `node_logger.py`, `logger.py`, and `cache.py`. These are aggregated by the `/metrics` endpoint and visualized in the project's Streamlit dashboard. System-level metrics (CPU, memory, GPU utilization) are delegated to the existing Grafana/Prometheus stack already running on the server.

**Why:** Infrastructure monitoring and application observability are different concerns. Grafana is the right tool for GPU utilization graphs and memory pressure alerts across multiple services. Embedding `psutil` calls into agent nodes would mix business logic with infrastructure concerns and duplicate what Grafana already does better. The application loggers focus on what matters for debugging agent behavior: which node was slow, how many tokens the LLM used, did the user rate the response positively.

**Trade-off:** Two separate dashboards to check. A unified observability platform (Datadog, Grafana with custom panels) would consolidate both. For a single-server deployment this is acceptable — the Grafana dashboard covers infrastructure, the Streamlit dashboard covers agent behavior.

---

## What I would do differently

1. **Parse ingredient quantities.** Currently nutrition totals treat every ingredient as ~100g. NLP-based quantity parsing ("3/4 cup soy sauce" → 180g) would make nutrition data accurate enough to trust.

2. **Add a reranker LLM node.** Like the Legal RAG system, a reranker between Recipe Agent and Nutrition Agent would improve precision. Currently we rely on cosine similarity alone.

3. **Scheduled re-ingestion.** TheMealDB data is ingested once. A daily cron job that detects new meals and ingests them would keep the corpus fresh.

4. **Redis for cache.** The current `QueryCache` is in-memory per-container. Multiple backend instances would have separate caches. Redis would give a shared cache across instances.

5. **Structured logging from day one.** I retrofitted `node_logger.py` and `llm_logger.py` from the Legal RAG project. Designing the logging schema upfront would have saved time.

## What worked well

- **Two-collection RAG** is the strongest technical decision. The combination of semantic recipe search + health-goal nutritional re-ranking is genuinely useful and not found in off-the-shelf solutions.
- **MCP architecture** cleanly separates YouTube integration from agent logic. Swapping YouTube for another video provider requires zero changes to agent code.
- **`reset_per_turn_state()`** pattern eliminates an entire class of stateful bugs. Every agent starts with a clean slate.
- **Test pyramid** — 96 tests with fast unit tests at the bottom, slow E2E tests at the top — means the fast tests run in seconds and catch most bugs before the expensive pipeline tests run.