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
    from config import (
        DEFAULT_ALPHA,
        DEFAULT_MAX_CHUNKS,
        DEFAULT_MIN_ABSOLUTE_SCORE,
        DEFAULT_MIN_SCORE_RATIO,
        DEFAULT_TOP_K,
        DEFAULT_WORD_BUDGET,
    )
except ImportError:
    DEFAULT_TOP_K = 6
    DEFAULT_ALPHA = 0.5
    DEFAULT_MAX_CHUNKS = 4
    DEFAULT_WORD_BUDGET = 350
    DEFAULT_MIN_ABSOLUTE_SCORE = 0.12
    DEFAULT_MIN_SCORE_RATIO = 0.4



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
    min_absolute_score: float = DEFAULT_MIN_ABSOLUTE_SCORE,
    min_score_ratio: float = DEFAULT_MIN_SCORE_RATIO,
) -> dict:
    
    candidates = retrieve_hybrid(query, collection, k=k, alpha=alpha)

    if not candidates or candidates[0]["score"] < min_absolute_score:
        return {
            "context_text": "",
            "sources": [],
            "used_words": 0,
            "has_sufficient_context": False,
        }

    top_score = candidates[0]["score"]
    score_floor = max(min_absolute_score, min_score_ratio * top_score)
    candidates = [row for row in candidates if row["score"] >= score_floor]

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
        "has_sufficient_context": bool(selected),
    }


if __name__ == "__main__":
    _store = load_stage("05_create_chroma_store")
    coll = _store.get_or_build_collection()
    package = build_context_package("What are the main factors for developing obesity in dogs?", coll)
    print(package["context_text"])
    print("\nSources:", [s["title"] for s in package["sources"]])
