from __future__ import annotations

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

try:
    from config import EMBEDDING_MODEL_NAME
except ImportError:
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_sentence_transformer(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    
    return SentenceTransformer(model_name)


class SentenceTransformerEmbeddingFunction:
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002 (chroma's expected arg name)
        model = get_sentence_transformer(self.model_name)
        embeddings = model.encode(
            list(input), convert_to_numpy=True, normalize_embeddings=True
        )
        return embeddings.tolist()

    def name(self) -> str:
        return f"sentence-transformers:{self.model_name}"


def embed_texts(texts: List[str], model_name: str = EMBEDDING_MODEL_NAME):
    """Embed a batch of texts, returning a normalized numpy array."""
    model = get_sentence_transformer(model_name)
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def embed_query(query: str, model_name: str = EMBEDDING_MODEL_NAME):
    """Embed a single query string, returning a normalized numpy vector."""
    model = get_sentence_transformer(model_name)
    return model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]


if __name__ == "__main__":
    vec = embed_query("What do puppies need to eat?")
    print(f"Model: {EMBEDDING_MODEL_NAME}")
    print(f"Embedding dimension: {vec.shape[0]}")
