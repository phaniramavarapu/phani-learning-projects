"""Cross-encoder reranking: re-score a wider candidate pool for real relevance.

Embedding similarity (bi-encoder) scores the query and each chunk independently,
so it's fast but coarse - our scores all sit in a tight 0.49-0.54 band with no
clear winner. A cross-encoder reads the query and chunk together in one pass,
which is far more accurate at judging relevance, but too slow to run over the
whole index - so it only re-scores the top candidates retrieval already found.
"""

from functools import lru_cache

import torch
from langchain_core.documents import Document

from newsrag.config import get_settings


@lru_cache
def get_reranker():
    from sentence_transformers import CrossEncoder

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return CrossEncoder(get_settings().reranker_model, device=device)


def rerank(query: str, hits: list[tuple[Document, float]], top_n: int) -> list[tuple[Document, float]]:
    """Re-score `hits` with the cross-encoder and return the best `top_n`, highest first."""
    if not hits:
        return hits
    model = get_reranker()
    pairs = [(query, doc.page_content) for doc, _ in hits]
    scores = model.predict(pairs)
    reranked = sorted(zip((doc for doc, _ in hits), scores), key=lambda pair: pair[1], reverse=True)
    return [(doc, float(score)) for doc, score in reranked[:top_n]]
