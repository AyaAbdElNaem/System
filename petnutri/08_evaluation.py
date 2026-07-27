from __future__ import annotations

import re
from typing import Callable, List

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pipeline_utils import load_stage

_store = load_stage("05_create_chroma_store")
_vectors = load_stage("04_vector_representation")
embed_query = _vectors.embed_query

try:
    from config import DEFAULT_ALPHA
except ImportError:
    DEFAULT_ALPHA = 0.5

K_EVAL = 3

# --------------------------------------------------------------------------
GROUND_TRUTH = {
    "What is the nutritional requirements of dogs ?": [0],
    "what are the body condition score scales?  ": [0],
    "what is the meaning of malnutrition?": [0],
    "how protein requirements of dogs and cats vary vary ?": [1],
    "what are the different life stages in nutrient requirements?": [1],
    "what is the importance of water to pets?": [1],
    "what are the signs of protein deficiency?": [1],
    "How many times per day should puppies between weaning and 6 months old be fed?": [2],
    "What are the main factors for developing obesity in dogs?": [2],
    "What is the pros and cons of protein?": [1, 2],
    "What are the three most common food allergens in domestic dogs?": [3],
    "what is the caloric requirements of lactating queen?": [3],
    "What is the best feeding regimen to adult dogs?": [3],
    "what is the impact of overfeeding": [2, 3],
    "How does excessive fat tissue actively contribute to chronic joint inflammation and metabolic diseases?": [3],
    "Which amino acid deficiency causes dilated cardiomyopathy and central retinal degeneration in cats?": [4],
    "Why are plant-based ingredients incapable of fulfilling a cat's organic requirement for Vitamin A and fatty acids?": [4],
    "are dog foods satisfactory for cats?": [4],
    "How does an excessive intake of a single mineral negatively impact the utilization of other essential elements?": [4],
    "Write down the exact exponential formula used to calculate a pet's Resting Energy Requirement (RER), and explain how "
    "this calculation be modified (by what specific percentage) when starting a weight loss program for an overweight "
    "feline.": [3, 4],
    "Why is it incorrect to view adult maintenance as a single, unchanging life stage as a pet ages, and how must the "
    "digestibility of macronutrients be modified when feeding older felines compared to middle-aged ones?": [2, 4],
}


def build_queries_df() -> pd.DataFrame:
    df = pd.DataFrame(
        {"query": list(GROUND_TRUTH.keys()), "relevant_document_ids": list(GROUND_TRUTH.values())}
    )
    df.insert(0, "query_id", [f"q{i + 1}" for i in range(len(df))])
    return df


def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

# Metrics 
# --------------------------------------------------------------------------
def precision_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    hits = [doc_id for doc_id in retrieved_ids[:k] if doc_id in relevant_ids]
    return len(hits) / k


def recall_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    hits = set(retrieved_ids[:k]).intersection(set(relevant_ids))
    return len(hits) / len(relevant_ids)


def hit_rate_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> int:
    hits = set(retrieved_ids[:k]).intersection(set(relevant_ids))
    return int(len(hits) > 0)


def reciprocal_rank(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in set(relevant_ids):
            return 1 / rank
    return 0.0


def evaluate_retriever_on_df(
    retriever_name: str,
    retrieval_function: Callable[[str, int], List[dict]],
    eval_df: pd.DataFrame,
    k: int = K_EVAL,
) -> pd.DataFrame:
    rows = []
    for _, row_data in eval_df.iterrows():
        query_text = row_data["query"]
        results = retrieval_function(query_text, k)
        retrieved_doc_ids = [int(r["document_id"]) for r in results]
        relevant_ids = row_data["relevant_document_ids"]
        relevant_ids = relevant_ids if isinstance(relevant_ids, list) else [relevant_ids]
        relevant_ids = [int(x) for x in relevant_ids]

        rows.append(
            {
                "retriever": retriever_name,
                "query": query_text,
                f"precision@{k}": precision_at_k(retrieved_doc_ids, relevant_ids, k),
                f"recall@{k}": recall_at_k(retrieved_doc_ids, relevant_ids, k),
                f"hit_rate@{k}": hit_rate_at_k(retrieved_doc_ids, relevant_ids, k),
                "reciprocal_rank": reciprocal_rank(retrieved_doc_ids, relevant_ids),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    print("Loading the corpus from ChromaDB (building it first if needed)...")
    collection = _store.get_or_build_collection()
    corpus = collection.get(include=["documents", "metadatas", "embeddings"])

    search_texts = corpus["documents"]
    metadatas = corpus["metadatas"]
    embeddings = np.array(corpus["embeddings"])

    def _row(idx: int, score: float) -> dict:
        return {**metadatas[idx], "score": score}

    # --- TF-IDF ---
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(search_texts)

    def retrieve_top_k_tfidf(query: str, k: int) -> List[dict]:
        query_vector = tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
        ranking = np.argsort(scores)[::-1][:k]
        return [_row(i, float(scores[i])) for i in ranking]

    # --- BM25 ---
    tokenized_corpus = [_simple_tokenize(t) for t in search_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    def retrieve_top_k_bm25(query: str, k: int) -> List[dict]:
        scores = np.array(bm25.get_scores(_simple_tokenize(query)))
        ranking = np.argsort(scores)[::-1][:k]
        return [_row(i, float(scores[i])) for i in ranking]

    # --- Semantic ---
    def retrieve_top_k_semantic(query: str, k: int) -> List[dict]:
        query_embedding = embed_query(query)
        scores = embeddings @ query_embedding
        ranking = np.argsort(scores)[::-1][:k]
        return [_row(i, float(scores[i])) for i in ranking]

    # --- Hybrid (the retriever actually used by the app) ---
    def retrieve_top_k_hybrid(query: str, k: int) -> List[dict]:
        bm25_scores = np.array(bm25.get_scores(_simple_tokenize(query)))
        semantic_scores = embeddings @ embed_query(query)

        def _min_max(scores: np.ndarray) -> np.ndarray:
            s_min, s_max = scores.min(), scores.max()
            if s_max - s_min == 0:
                return np.zeros_like(scores)
            return (scores - s_min) / (s_max - s_min)

        hybrid_scores = DEFAULT_ALPHA * _min_max(bm25_scores) + (1 - DEFAULT_ALPHA) * _min_max(semantic_scores)
        ranking = np.argsort(hybrid_scores)[::-1][:k]
        return [_row(i, float(hybrid_scores[i])) for i in ranking]

    queries_df = build_queries_df()

    print(f"Evaluating {len(queries_df)} ground-truth queries at k={K_EVAL}...\n")
    tfidf_results = evaluate_retriever_on_df("TF-IDF", retrieve_top_k_tfidf, queries_df, k=K_EVAL)
    bm25_results = evaluate_retriever_on_df("BM25", retrieve_top_k_bm25, queries_df, k=K_EVAL)
    semantic_results = evaluate_retriever_on_df("Semantic (Embeddings)", retrieve_top_k_semantic, queries_df, k=K_EVAL)
    hybrid_results = evaluate_retriever_on_df("Hybrid Search (used by the app)", retrieve_top_k_hybrid, queries_df, k=K_EVAL)

    all_results = pd.concat([tfidf_results, bm25_results, semantic_results, hybrid_results])

    comparison_dashboard = (
        all_results.groupby("retriever")
        .agg(
            **{
                f"precision@{K_EVAL}": (f"precision@{K_EVAL}", "mean"),
                f"recall@{K_EVAL}": (f"recall@{K_EVAL}", "mean"),
                f"hit_rate@{K_EVAL}": (f"hit_rate@{K_EVAL}", "mean"),
                "Mean Reciprocal Rank (MRR)": ("reciprocal_rank", "mean"),
            }
        )
        .reset_index()
    )

    print("=== RETRIEVER PERFORMANCE COMPARISON ===")
    print(comparison_dashboard.to_string(index=False))

    output_path = "evaluation_results.csv"
    comparison_dashboard.to_csv(output_path, index=False)
    print(f"\nSaved comparison table to '{output_path}'.")


if __name__ == "__main__":
    main()
