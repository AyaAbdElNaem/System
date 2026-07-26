"""
config.py
=========
Central configuration for the PetNutri RAG application.

Every other module (01_documents.py ... streamlit_app.py) imports paths and
constants from here instead of hard-coding them, so the whole project has a
single source of truth for file locations and model/API settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent

DATA_DIR: Path = BASE_DIR / "data" / "nutrition"
DATABASE_DIR: Path = BASE_DIR / "database"
CHROMA_PERSIST_DIR: Path = DATABASE_DIR / "chroma_db"
SOURCE_HASH_FILE: Path = DATABASE_DIR / "source_hash.json"
ASSETS_DIR: Path = BASE_DIR / "assets"
STYLE_CSS_PATH: Path = ASSETS_DIR / "style.css"

# --------------------------------------------------------------------------
# Retrieval / embedding settings (kept identical to the original notebook
# so retrieval behavior does not change: chunk_size=800, overlap=100,
# model="all-MiniLM-L6-v2")
# --------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 100
COLLECTION_NAME: str = "petnutri_nutrition"

DEFAULT_TOP_K: int = 6
DEFAULT_ALPHA: float = 0.5          # hybrid weighting: alpha*lexical + (1-alpha)*semantic
DEFAULT_MAX_CHUNKS: int = 4         # max chunks packed into the final context
DEFAULT_WORD_BUDGET: int = 350      # max words packed into the final context

# --------------------------------------------------------------------------
# OpenRouter (LLM generation) settings
# --------------------------------------------------------------------------
OPENROUTER_API_URL: str = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL: str = "openai/gpt-4o-mini"
APP_TITLE: str = "PetNutri"
APP_REFERRER: str = "https://petnutri.streamlit.app"


def get_openrouter_credentials() -> Tuple[Optional[str], str]:
    """
    Resolve the OpenRouter API key and model name.

    Resolution order (per the assignment's Streamlit Secrets rules):
      1. Streamlit secrets (``st.secrets``) - used when deployed on
         Streamlit Cloud.
      2. Environment variables (``.env`` loaded via python-dotenv) - used
         for local development.

    Returns
    -------
    Tuple[Optional[str], str]
        (api_key, model_name). api_key is None if not configured anywhere,
        which callers must handle gracefully (see 07_prompting.py).
    """
    api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    model: str = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)

    try:
        import streamlit as st  # imported lazily so non-Streamlit scripts still work

        if not api_key:
            api_key = st.secrets.get("OPENROUTER_API_KEY", None)
        secret_model = st.secrets.get("OPENROUTER_MODEL", None)
        if secret_model:
            model = secret_model
    except Exception:
        # No Streamlit context, no secrets.toml, or secrets not configured.
        pass

    return api_key, model


def ensure_directories() -> None:
    """Create any runtime directories that must exist before the app runs."""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
