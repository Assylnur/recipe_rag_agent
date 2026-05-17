"""
milvus_connector.py — Milvus connector for recipe RAG ingestion.

Two collections, same class:
    recipes_main       — Recipe Agent  (semantic: name + cuisine + ingredients)
    nutrition_profiles — Nutrition Agent (semantic: natural-language macro description)

Schema (both collections)
--------------------------
id         INT64         primary key, auto_id
embedding  FLOAT_VECTOR  dim=1024
text       VARCHAR(65535) embedded text
meta       JSON          all scalar fields (meal_id, name, category, nutrition, etc.)

Usage:
    from db.milvus_connector import MilvusRecipeConnector, COLLECTION_RECIPES, COLLECTION_NUTRITION

    connector = MilvusRecipeConnector(COLLECTION_RECIPES, model)
    connector.add_documents(texts, metas)
"""

from typing import List, Optional

import numpy as np
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from sentence_transformers import SentenceTransformer

MILVUS_HOST = "172.17.0.1"
MILVUS_PORT = "19530"

COLLECTION_RECIPES   = "recipes_main"
COLLECTION_NUTRITION = "nutrition_profiles"


class MilvusRecipeConnector:
    def __init__(
        self,
        collection_name: str,
        embedding_model: SentenceTransformer,
        dim: int = 1024,
    ):
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.dim = dim

        if utility.has_collection(collection_name):
            self.collection = Collection(collection_name)
        else:
            fields = [
                FieldSchema(name="id",        dtype=DataType.INT64,         is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,  dim=self.dim),
                FieldSchema(name="text",      dtype=DataType.VARCHAR,       max_length=65535),
                FieldSchema(name="meta",      dtype=DataType.JSON),
            ]
            schema = CollectionSchema(fields, description=f"Recipe RAG — {collection_name}")
            self.collection = Collection(name=collection_name, schema=schema)
            self.collection.create_index(
                field_name="embedding",
                index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"},
            )
            self.collection.load()

    # ── Embeddings ─────────────────────────────────────────────────────────────

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        embs = self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        embs = np.array(embs)
        if len(embs.shape) == 1:
            embs = embs.reshape(1, -1)
        return embs

    # ── Insert ─────────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[str], metas: List[dict], batch_size: int = 512):
        """
        Embed and insert documents with their metadata.

        Parameters
        ----------
        documents : list of texts to embed (recipe text or nutrition description)
        metas     : list of dicts stored in the JSON meta field
        batch_size: number of records per Milvus insert call
        """
        assert len(documents) == len(metas), "documents and metas must have the same length"

        for i in range(0, len(documents), batch_size):
            docs_batch  = documents[i : i + batch_size]
            metas_batch = metas[i : i + batch_size]
            embeddings  = self.get_embeddings(docs_batch)

            entities = [embeddings.tolist(), docs_batch, metas_batch]
            try:
                self.collection.insert(entities)
            except Exception as e:
                print(f"[{self.collection_name}] Error inserting batch {i}: {e}")
            self.collection.flush()

        print(f"[{self.collection_name}] Total entities: {self.collection.num_entities}")

    # ── Delete by meal_id ──────────────────────────────────────────────────────

    def delete_by_meal_id(self, meal_id: str):
        """Delete all records for a given meal_id (e.g. before re-ingesting)."""
        self.collection.load()
        results = self.collection.query(
            expr=f'meta["meal_id"] == "{meal_id}"',
            output_fields=["id"],
        )
        if not results:
            print(f"[{self.collection_name}] No rows found for meal_id='{meal_id}'")
            return

        ids = [r["id"] for r in results]
        self.collection.delete(expr=f"id in {ids}")
        self.collection.flush()
        print(f"[{self.collection_name}] Deleted {len(ids)} rows for meal_id='{meal_id}'")

    # ── Existing IDs ───────────────────────────────────────────────────────────

    def get_existing_meal_ids(self) -> set[str]:
        """Return set of meal_ids already in the collection (for resume support)."""
        if self.collection.num_entities == 0:
            return set()
        results = self.collection.query(
            expr='meta["meal_id"] != ""',
            output_fields=["meta"],
            limit=16384,
        )
        return {r["meta"]["meal_id"] for r in results}