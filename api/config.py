"""
config.py — model and collection configuration for the Recipe RAG Agent.
"""

import os
from typing import Literal
from langchain_openai import ChatOpenAI
from llm_logger import LLMCallbackLogger

# ── Collections ────────────────────────────────────────────────────────────────
COLLECTION_RECIPES   = "recipes_main"        # Recipe Agent
COLLECTION_NUTRITION = "nutrition_profiles"  # Nutrition Agent

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM        = 768

MAX_RECIPE_RESULTS   = 10   # Recipe Agent returns top-N candidates
MAX_NUTRITION_RERANK = 5    # Nutrition Agent re-ranks down to top-N

# ── Corpus path (for full meal details: instructions, thumbnail, etc.) ─────────
CORPUS_PATH = os.getenv("CORPUS_PATH", "data/raw/enriched_corpus.json")

# ── LLM ───────────────────────────────────────────────────────────────────────
VLLM_URL = os.getenv("VLLM_URL", "http://host.docker.internal:8080/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss-120b")

NODE_CONFIGS = {
    "ingredient_agent": {"temperature": 0},      # structured extraction — deterministic
    "recipe_agent":     {"temperature": 0},      # filter building — deterministic
    "nutrition_agent":  {"temperature": 0},      # health analysis — deterministic
    "responder":        {"temperature": 0.4},    # final answer — some creativity
}

MAX_TOKENS = {
    "ingredient_agent": 1024,
    "recipe_agent":     512,
    "nutrition_agent":  1024,
    "responder":        4096,
}


def get_llm(
    node_type: Literal["ingredient_agent", "recipe_agent", "nutrition_agent", "responder"],
    streaming: bool = False,
) -> ChatOpenAI:
    config = NODE_CONFIGS.get(node_type, {"temperature": 0})
    extra_body = {
        "max_completion_tokens": MAX_TOKENS.get(node_type, 1024),
        **({"stream_options": {"include_usage": True}} if streaming else {}),
    }
    return ChatOpenAI(
        base_url=VLLM_URL,
        model=MODEL_NAME,
        api_key="EMPTY",
        temperature=config["temperature"],
        streaming=streaming,
        timeout=None,
        max_retries=2,
        callbacks=[LLMCallbackLogger(node_type=node_type, model=MODEL_NAME)],
        extra_body=extra_body,
    )
