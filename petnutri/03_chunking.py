"""
03_chunking.py
===============
Stage 3 of the RAG pipeline: CHUNKING.

Two-pass, markdown-aware chunking (identical parameters to the original
notebook so retrieval behavior is preserved):

1. ``MarkdownHeaderTextSplitter`` splits each document along its
   #, ##, ### headers, so every chunk keeps track of the section it came
   from.
2. ``RecursiveCharacterTextSplitter`` (chunk_size=800, chunk_overlap=100)
   further splits any oversized sections, using paragraph/line/word
   boundaries in that order of preference.

Each resulting chunk keeps the full document metadata (title, source,
category, last_updated, target_species, url_reference) plus a
``search_text`` field that prepends header + title + category context,
exactly like the original ``search_text`` column - this is what improved
retrieval quality in the original evaluation.
"""

from __future__ import annotations

from typing import List

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

try:
    from config import CHUNK_OVERLAP, CHUNK_SIZE
except ImportError:
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100

from pipeline_utils import load_stage

_preprocessing = load_stage("02_preprocessing")
normalize_raw_text = _preprocessing.normalize_raw_text
clean_chunk_text = _preprocessing.clean_chunk_text


HEADERS_TO_SPLIT_ON = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
]

_markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False
)

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""],
)


def chunk_documents(documents: List[dict]) -> List[dict]:
    """
    Split every document into retrieval-ready chunks.

    Parameters
    ----------
    documents : List[dict]
        Output of ``01_documents.load_documents``.

    Returns
    -------
    List[dict]
        One dict per chunk with keys: chunk_id, document_id, title, source,
        category, last_updated, target_species, url_reference, chunk_index,
        chunk_text (cleaned), raw_chunk_text (original), search_text
        (header + title + category + cleaned content, used for embedding
        and lexical search).
    """
    chunk_rows: List[dict] = []

    for doc in documents:
        normalized_text = normalize_raw_text(doc["text"])
        md_header_splits = _markdown_splitter.split_text(normalized_text)
        final_chunks = _text_splitter.split_documents(md_header_splits)

        for chunk_index, chunk in enumerate(final_chunks):
            raw_chunk_text = chunk.page_content.strip()
            clean_text = clean_chunk_text(raw_chunk_text)
            if not clean_text:
                continue  # skip empty/whitespace-only chunks

            h1 = chunk.metadata.get("header_1", "")
            h2 = chunk.metadata.get("header_2", "")
            h3 = chunk.metadata.get("header_3", "")
            headers_context = " ".join(part for part in (h1, h2, h3) if part).strip()

            target_species = doc["target_species"]
            target_species_str = (
                ", ".join(target_species)
                if isinstance(target_species, list)
                else target_species
            )

            chunk_rows.append(
                {
                    "chunk_id": f"doc{doc['document_id']}_chunk{chunk_index}",
                    "document_id": doc["document_id"],
                    "title": doc["title"],
                    "source": doc["source"],
                    "category": doc["category"],
                    "last_updated": doc["last_updated"],
                    "target_species": target_species_str,
                    "url_reference": doc["url_reference"],
                    "chunk_index": chunk_index,
                    "chunk_text": clean_text,
                    "raw_chunk_text": raw_chunk_text,
                    "search_text": (
                        f"Context: {headers_context} | Title: {doc['title']} | "
                        f"Category: {doc['category']} | Content: {clean_text}"
                    ),
                }
            )

    return chunk_rows


if __name__ == "__main__":
    _documents = load_stage("01_documents")
    docs = _documents.load_documents()
    chunks = chunk_documents(docs)
    print(f"Produced {len(chunks)} chunks from {len(docs)} document(s).")
    if chunks:
        print("\nSample chunk:")
        print(f"  chunk_id   : {chunks[0]['chunk_id']}")
        print(f"  title      : {chunks[0]['title']}")
        print(f"  search_text: {chunks[0]['search_text'][:200]}...")
