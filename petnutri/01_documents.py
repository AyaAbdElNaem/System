"""
01_documents.py
================
Stage 1 of the RAG pipeline: DOCUMENTS.

Loads the raw Markdown knowledge base and attaches curated metadata to each
file. This preserves the exact metadata scheme from the original project
code (title, source, category, last_updated, target_species, url_reference)
so downstream citations keep working unchanged.

Public API
----------
load_documents(data_dir) -> List[dict]
compute_documents_hash(documents) -> str
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Union

from langchain_community.document_loaders import DirectoryLoader, TextLoader

try:
    from config import DATA_DIR
except ImportError:  # allow running this file standalone (python 01_documents.py)
    DATA_DIR = Path(__file__).resolve().parent / "data" / "nutrition"

# --------------------------------------------------------------------------
# Curated metadata per source file (unchanged from the original notebook)
# --------------------------------------------------------------------------
CUSTOM_METADATA: Dict[str, Dict] = {
    "01_overview_of_nutrition.md": {
        "title": "Overview of Nutrition: Small Animals",
        "source": "MSD Veterinary Manual",
        "category": "Nutrition",
        "last_updated": "2024-09",
        "target_species": ["Dogs", "Cats"],
        "url_reference": "https://www.msdvetmanual.com/management-and-nutrition/nutrition-small-animals/overview-of-nutrition-small-animals",
    },
    "02_nutritional_req.md": {
        "title": "Nutritional Requirements of Small Animals",
        "source": "MSD Veterinary Manual",
        "category": "Nutritional Requirements",
        "last_updated": "2024-09",
        "target_species": ["Dogs", "Cats"],
        "url_reference": "https://www.msdvetmanual.com/management-and-nutrition/nutrition-small-animals/nutritional-requirements-of-small-animals",
    },
    "03_nutrition_disease_managment.md": {
        "title": "Nutrition in Disease Management in Small Animals",
        "source": "MSD Veterinary Manual",
        "category": "Management and Nutrition",
        "last_updated": "2025-06",
        "target_species": ["Dogs", "Cats"],
        "url_reference": "https://www.msdvetmanual.com/management-and-nutrition/nutrition-small-animals/nutrition-in-disease-management-in-small-animals",
    },
    "04_feeding_practices.md": {
        "title": "Feeding Practices in Small Animals",
        "source": "MSD Veterinary Manual",
        "category": "Management and Nutrition",
        "last_updated": "2024-10",
        "target_species": ["Dogs", "Cats"],
        "url_reference": "https://www.msdvetmanual.com/management-and-nutrition/nutrition-small-animals/feeding-practices-in-small-animals",
    },
    "05_foods_managment.md": {
        "title": "Dog and Cat Foods",
        "source": "MSD Veterinary Manual",
        "category": "Dog and Cat Foods",
        "last_updated": "2025-06",
        "target_species": ["Dogs", "Cats"],
        "url_reference": "https://www.msdvetmanual.com/management-and-nutrition/nutrition-small-animals/dog-and-cat-foods",
    },
}


class DocumentLoadError(Exception):
    """Raised when the knowledge base directory is missing or empty."""


def load_documents(data_dir: Union[str, Path] = DATA_DIR) -> List[dict]:
    """
    Load every ``*.md`` file under ``data_dir`` and attach curated metadata.

    Parameters
    ----------
    data_dir : str | Path
        Folder containing the Markdown knowledge base files.

    Returns
    -------
    List[dict]
        One dict per document with keys: document_id, file_name, title,
        source, category, last_updated, target_species, url_reference, text.

    Raises
    ------
    DocumentLoadError
        If the directory does not exist or contains no Markdown files.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise DocumentLoadError(
            f"Knowledge base folder not found: '{data_dir}'. "
            "Create it and add your Markdown files, or check config.DATA_DIR."
        )

    md_files = sorted(data_dir.glob("*.md"))
    if not md_files:
        raise DocumentLoadError(
            f"No Markdown (*.md) files found in '{data_dir}'. "
            "Add at least one knowledge base file before building the database."
        )

    loader = DirectoryLoader(
        str(data_dir),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    raw_documents = loader.load()
    # DirectoryLoader does not guarantee file order; sort for reproducibility.
    raw_documents.sort(key=lambda d: os.path.basename(d.metadata["source"]))

    documents: List[dict] = []
    for index, doc in enumerate(raw_documents):
        file_name = os.path.basename(doc.metadata["source"])
        file_metadata = CUSTOM_METADATA.get(file_name, {})

        documents.append(
            {
                "document_id": index,
                "file_name": file_name,
                "title": file_metadata.get("title", file_name),
                "source": file_metadata.get("source", "Unknown source"),
                "category": file_metadata.get("category", "General"),
                "last_updated": file_metadata.get("last_updated", ""),
                "target_species": file_metadata.get("target_species", []),
                "url_reference": file_metadata.get("url_reference", ""),
                "text": doc.page_content,
            }
        )

    return documents


def compute_documents_hash(documents: List[dict]) -> str:
    """
    Compute a stable hash representing the current state of the knowledge
    base (file names + content). Used by 05_create_chroma_store.py to decide
    whether the vector database needs to be rebuilt.
    """
    hasher = hashlib.sha256()
    for doc in sorted(documents, key=lambda d: d["file_name"]):
        hasher.update(doc["file_name"].encode("utf-8"))
        hasher.update(doc["text"].encode("utf-8"))
    return hasher.hexdigest()


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} document(s) from '{DATA_DIR}':")
    for d in docs:
        print(f"  - [{d['document_id']}] {d['title']} ({d['file_name']}, {len(d['text'])} chars)")
    print(f"Documents hash: {compute_documents_hash(docs)}")
