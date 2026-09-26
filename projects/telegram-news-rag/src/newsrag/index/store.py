"""Embeddings + Qdrant vector store, shared by the indexer and the query side."""

from functools import lru_cache

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from newsrag.config import get_settings

# Metadata fields we filter on; Qdrant needs a payload index for fast filtering
FILTER_FIELDS = {
    "metadata.edition_date": models.PayloadSchemaType.KEYWORD,
    "metadata.publication": models.PayloadSchemaType.KEYWORD,
    "metadata.source": models.PayloadSchemaType.KEYWORD,
}


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    device = "mps" if torch.backends.mps.is_available() else "cpu"  # Apple Silicon GPU
    return HuggingFaceEmbeddings(
        model_name=get_settings().embed_model,
        model_kwargs={"device": device},
        # Normalized vectors make cosine similarity a plain dot product
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )


@lru_cache
def get_client() -> QdrantClient:
    settings = get_settings()
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.qdrant_path))


def get_vector_store() -> QdrantVectorStore:
    settings = get_settings()
    client, embeddings = get_client(), get_embeddings()

    if not client.collection_exists(settings.qdrant_collection):
        dim = len(embeddings.embed_query("dimension probe"))
        client.create_collection(
            settings.qdrant_collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        for field, schema in FILTER_FIELDS.items():
            client.create_payload_index(settings.qdrant_collection, field, schema)

    return QdrantVectorStore(client=client, collection_name=settings.qdrant_collection, embedding=embeddings)
