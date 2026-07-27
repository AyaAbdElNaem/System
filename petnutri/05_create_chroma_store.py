from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import chromadb

from pipeline_utils import load_stage

_documents = load_stage("01_documents")
_chunking = load_stage("03_chunking")
_vectors = load_stage("04_vector_representation")

load_documents = _documents.load_documents
compute_documents_hash = _documents.compute_documents_hash
chunk_documents = _chunking.chunk_documents
SentenceTransformerEmbeddingFunction = _vectors.SentenceTransformerEmbeddingFunction

try:
    from config import (
        CHROMA_PERSIST_DIR,
        COLLECTION_NAME,
        DATA_DIR,
        EMBEDDING_MODEL_NAME,
        SOURCE_HASH_FILE,
        ensure_directories,
    )
except ImportError:
    CHROMA_PERSIST_DIR = Path(__file__).resolve().parent / "database" / "chroma_db"
    SOURCE_HASH_FILE = CHROMA_PERSIST_DIR.parent / "source_hash.json"
    COLLECTION_NAME = "petnutri_nutrition"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    DATA_DIR = Path(__file__).resolve().parent / "data" / "nutrition"

    def ensure_directories() -> None:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)


class VectorStoreError(Exception):
    """Raised when the ChromaDB store cannot be built or accessed."""


def _get_client() -> chromadb.ClientAPI:
    ensure_directories()
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))


def _read_stored_hash() -> Optional[dict]:
    if not SOURCE_HASH_FILE.exists():
        return None
    try:
        return json.loads(SOURCE_HASH_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_stored_hash(hash_value: str, num_documents: int, num_chunks: int) -> None:
    SOURCE_HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_HASH_FILE.write_text(
        json.dumps(
            {
                "hash": hash_value,
                "num_documents": num_documents,
                "num_chunks": num_chunks,
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "embedding_model": EMBEDDING_MODEL_NAME,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def get_or_build_collection(force_rebuild: bool = False):
    
    try:
        client = _get_client()
        documents = load_documents(DATA_DIR)
        current_hash = compute_documents_hash(documents)
        stored = _read_stored_hash()

        existing_collections = {c.name for c in client.list_collections()}
        needs_rebuild = (
            force_rebuild
            or COLLECTION_NAME not in existing_collections
            or stored is None
            or stored.get("hash") != current_hash
        )

        embedding_function = SentenceTransformerEmbeddingFunction(EMBEDDING_MODEL_NAME)

        if needs_rebuild:
            if COLLECTION_NAME in existing_collections:
                client.delete_collection(COLLECTION_NAME)

            collection = client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"},
            )

            chunks = chunk_documents(documents)
            if not chunks:
                raise VectorStoreError(
                    "Chunking produced zero chunks - check that your Markdown "
                    "files contain readable text."
                )

            batch_size = 100
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                collection.add(
                    ids=[c["chunk_id"] for c in batch],
                    documents=[c["search_text"] for c in batch],
                    metadatas=[
                        {
                            "document_id": c["document_id"],
                            "title": c["title"],
                            "source": c["source"],
                            "category": c["category"],
                            "last_updated": c["last_updated"],
                            "target_species": c["target_species"],
                            "url_reference": c["url_reference"],
                            "chunk_index": c["chunk_index"],
                            "chunk_text": c["chunk_text"],
                            "raw_chunk_text": c["raw_chunk_text"],
                        }
                        for c in batch
                    ],
                )

            _write_stored_hash(current_hash, len(documents), len(chunks))
        else:
            collection = client.get_collection(
                name=COLLECTION_NAME, embedding_function=embedding_function
            )

        return collection

    except VectorStoreError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a friendly error
        raise VectorStoreError(f"Failed to build or load the vector store: {exc}") from exc


def get_collection_stats() -> dict:

    stored = _read_stored_hash()
    client = None
    collection_exists = False
    chunk_count = 0

    try:
        client = _get_client()
        existing_collections = {c.name for c in client.list_collections()}
        collection_exists = COLLECTION_NAME in existing_collections
        if collection_exists:
            embedding_function = SentenceTransformerEmbeddingFunction(EMBEDDING_MODEL_NAME)
            collection = client.get_collection(
                name=COLLECTION_NAME, embedding_function=embedding_function
            )
            chunk_count = collection.count()
    except Exception:
        collection_exists = False

    num_documents = 0
    try:
        num_documents = len(list(DATA_DIR.glob("*.md")))
    except Exception:
        pass

    return {
        "database_ready": collection_exists,
        "num_documents": num_documents,
        "num_chunks": chunk_count if collection_exists else (stored or {}).get("num_chunks", 0),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "last_built": (stored or {}).get("built_at", "Never"),
        "persist_dir": str(CHROMA_PERSIST_DIR),
    }


if __name__ == "__main__":
    collection = get_or_build_collection()
    stats = get_collection_stats()
    print("=== ChromaDB status ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
