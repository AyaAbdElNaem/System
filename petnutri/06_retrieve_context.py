"""
06_retrieve_context.py
========================
Stage 6 of the RAG pipeline: CONTEXT RETRIEVAL.

Reproduces the original notebook's hybrid retriever (BM25 lexical score +
SentenceTransformer semantic score, min-max normalized and combined with
weight ``alpha``) on top of ChromaDB as the storage layer, then packs the
top, deduplicated chunks into a citation-ready context block - identical
selection logic to the original ``build_context_package`` (word budget +
max chunk count).

Public API
----------
retrieve_hybrid(query, collection, k, alpha) -> list[dict]
build_context_package(query, collection, ...) -> dict
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from pipeline_utils import load_stage

_vectors = load_stage("04_vector_representation")
embed_query = _vectors.embed_query

try:
    from config import DEFAULT_ALPHA, DEFAULT_MAX_CHUNKS, DEFAULT_TOP_K, DEFAULT_WORD_BUDGET
except ImportError:
    DEFAULT_TOP_K = 6
    DEFAULT_ALPHA = 0.5
    DEFAULT_MAX_CHUNKS = 4
    DEFAULT_WORD_BUDGET = 350


def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min == 0:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


class RetrievalError(Exception):
    """Raised when the vector store cannot be queried (e.g. empty database)."""


def _fetch_corpus(collection):
    """
    Pull the full collection contents (documents, metadatas, embeddings)
    into memory once. The knowledge base is small (a few hundred chunks),
    so this mirrors the original notebook's in-memory hybrid search while
    still persisting everything durably in ChromaDB.
    """
    result = collection.get(include=["documents", "metadatas", "embeddings"])
    if not result["ids"]:
        raise RetrievalError(
            "The vector database is empty. Build it first from the sidebar "
            "('Rebuild Database') before asking a question."
        )
    return result


def retrieve_hybrid(
    query: str,
    collection,
    k: int = DEFAULT_TOP_K,
    alpha: float = DEFAULT_ALPHA,
) -> List[dict]:
    """
    Rank every chunk in the collection against ``query`` using a weighted
    combination of BM25 (lexical) and semantic (embedding cosine
    similarity) scores, and return the top ``k``.

    Parameters
    ----------
    query : str
        The user's question.
    collection : chromadb.Collection
        An already-built collection (see 05_create_chroma_store.py).
    k : int
        Number of top-ranked chunks to return.
    alpha : float
        Weight given to the lexical (BM25) score; ``1 - alpha`` is given to
        the semantic score. 0.5 weighs both equally.

    Returns
    -------
    List[dict]
        Chunks sorted by descending hybrid score, each with its metadata
        and a ``score`` field.
    """
    if not query or not query.strip():
        raise RetrievalError("Empty query - nothing to retrieve.")

    corpus = _fetch_corpus(collection)
    documents = corpus["documents"]
    metadatas = corpus["metadatas"]
    embeddings = np.array(corpus["embeddings"])

    tokenized_corpus = [_simple_tokenize(text) for text in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = np.array(bm25.get_scores(_simple_tokenize(query)))

    query_embedding = embed_query(query)
    semantic_scores = embeddings @ query_embedding  # both are L2-normalized -> cosine similarity

    lexical_norm = _min_max_normalize(bm25_scores)
    semantic_norm = _min_max_normalize(semantic_scores)
    hybrid_scores = alpha * lexical_norm + (1 - alpha) * semantic_norm

    ranking = np.argsort(hybrid_scores)[::-1][:k]

    results = []
    for rank in ranking:
        meta = metadatas[rank]
        results.append(
            {
                **meta,
                "score": float(hybrid_scores[rank]),
            }
        )
    return results


def build_context_package(
    query: str,
    collection,
    k: int = DEFAULT_TOP_K,
    alpha: float = DEFAULT_ALPHA,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    word_budget: int = DEFAULT_WORD_BUDGET,
) -> dict:
    """
    Retrieve, deduplicate, and word-budget-pack context for a query -
    identical selection strategy to the original notebook's
    ``build_context_package``.

    Returns
    -------
    dict with keys:
        - context_text: str, ready to drop into the LLM prompt
        - sources: List[dict], one per selected chunk (for citation display)
        - used_words: int
    """
    candidates = retrieve_hybrid(query, collection, k=k, alpha=alpha)

    selected = []
    seen_texts = set()
    used_words = 0

    for row in candidates:
        normalized_text = re.sub(r"\s+", " ", row["chunk_text"]).strip().lower()
        if normalized_text in seen_texts:
            continue

        chunk_words = len(row["chunk_text"].split())
        if selected and used_words + chunk_words > word_budget:
            continue

        selected.append(row)
        seen_texts.add(normalized_text)
        used_words += chunk_words

        if len(selected) >= max_chunks:
            break

    context_blocks = []
    sources = []
    for idx, row in enumerate(selected, start=1):
        context_blocks.append(
            f"[Source {idx}] {row['title']} | category={row['category']} | "
            f"source={row['source']} | score={row['score']:.4f}\n{row['chunk_text']}"
        )
        sources.append(
            {
                "index": idx,
                "title": row["title"],
                "category": row["category"],
                "source": row["source"],
                "last_updated": row.get("last_updated", ""),
                "url_reference": row.get("url_reference", ""),
                "score": row["score"],
                "excerpt": row.get("raw_chunk_text", row["chunk_text"])[:400],
            }
        )

    return {
        "context_text": "\n\n".join(context_blocks),
        "sources": sources,
        "used_words": used_words,
    }


if __name__ == "__main__":
    _store = load_stage("05_create_chroma_store")
    coll = _store.get_or_build_collection()
    package = build_context_package("What are the main factors for developing obesity in dogs?", coll)
    print(package["context_text"])
    print("\nSources:", [s["title"] for s in package["sources"]])
