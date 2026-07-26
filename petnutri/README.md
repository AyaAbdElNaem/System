# 🐾 PetNutri — AI Pet Nutrition Assistant

A retrieval-augmented generation (RAG) Streamlit app that answers dog and cat
nutrition questions using a curated veterinary knowledge base (MSD Veterinary
Manual excerpts), with grounded, cited answers only — no hallucinated advice.

> An Arabic step-by-step operations guide is available in
> [`README_AR.md`](README_AR.md).

---

## 1. Pipeline overview

The project follows the required lab sequence, one file per stage:

```
documents → preprocessing → chunking → vector representation → vector store
    → context retrieval → prompting → Streamlit UI
```

| File | Stage | Responsibility |
|---|---|---|
| `01_documents.py` | Documents | Loads the 5 Markdown files + curated metadata (title, source, category, species, URL) |
| `02_preprocessing.py` | Preprocessing | Text normalization (CRLF fix) + aggressive chunk cleaning |
| `03_chunking.py` | Chunking | Markdown-header-aware split, then recursive character split (800 chars / 100 overlap) |
| `04_vector_representation.py` | Vector representation | SentenceTransformer (`all-MiniLM-L6-v2`) embedding wrapper, Chroma-compatible |
| `05_create_chroma_store.py` | Vector store | Builds/loads a **persistent ChromaDB** collection; rebuilds only when the knowledge base changes |
| `06_retrieve_context.py` | Context retrieval | Hybrid BM25 + semantic scoring, dedup + word-budget context packing, citation metadata |
| `07_prompting.py` | Prompting | Casual-vs-knowledge classification, strict grounded prompt, streaming OpenRouter call |
| `streamlit_app.py` | UI | PetNutri landing + chat experience |

Supporting files: `config.py` (paths/constants/secrets resolution),
`pipeline_utils.py` (helper to import the numerically-prefixed stage files),
`data/nutrition/*.md` (knowledge base), `assets/style.css` (theme).

## 2. Requirements

- Python 3.10+
- An [OpenRouter](https://openrouter.ai) API key (free tier available)

## 3. Local setup

```bash
git clone <your-repo-url>
cd petnutri
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your real OPENROUTER_API_KEY

streamlit run streamlit_app.py
```

The first run (or the sidebar's **Rebuild Database** button) builds the
Chroma vector store from `data/nutrition/*.md` under `database/chroma_db/`.
Subsequent runs reuse it unless the Markdown files change.

## 4. Google Colab

```python
!git clone <your-repo-url>
%cd petnutri
!pip install -r requirements.txt -q

import os
os.environ["OPENROUTER_API_KEY"] = "your_openrouter_key_here"

!pip install -q pyngrok
from pyngrok import ngrok
!streamlit run streamlit_app.py &>/content/log.txt &
public_url = ngrok.connect(8501)
print(public_url)
```

## 5. Deploying to Streamlit Community Cloud (public URL)

1. Push this repo to GitHub **without** `.env` or any real key (`.gitignore`
   already excludes them).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at `streamlit_app.py`.
3. Open **Manage app → Secrets** and paste:
   ```toml
   OPENROUTER_API_KEY = "your_openrouter_key_here"
   OPENROUTER_MODEL = "openai/gpt-4o-mini"
   ```
4. Deploy. The resulting `*.streamlit.app` URL is public — anyone with the
   link can use the app without signing in.

### Known public-access limitations (not caused by this app's code)

- **Streamlit Community Cloud** free tier apps sleep after inactivity and
  cold-start on the next visit (10-30s delay) — unavoidable on the free tier.
- **OpenRouter** enforces per-key rate limits and, on free/low-tier models,
  request throughput caps; heavy simultaneous public traffic can hit 429s.
- ChromaDB's persistent store lives on the container's local disk; on
  Streamlit Cloud this disk is **ephemeral** across redeploys, so the app
  rebuilds the database automatically on cold start (this is why rebuilds
  are cheap and hash-checked rather than assumed to persist forever).

## 6. Updating the knowledge base

1. Add or edit `.md` files inside `data/nutrition/`.
2. If a new file needs custom citation metadata, add an entry to
   `CUSTOM_METADATA` in `01_documents.py`.
3. Click **🔄 Rebuild Database** in the sidebar (or delete
   `database/chroma_db/` and `database/source_hash.json` and rerun) — the
   app detects the content hash changed and rebuilds automatically even
   without a manual click.

## 7. Error handling

The app degrades gracefully instead of crashing for:
- Missing/empty `data/nutrition/` folder
- Missing `OPENROUTER_API_KEY`
- Empty or not-yet-built vector database
- Empty/invalid user queries (silently ignored)
- ChromaDB or embedding failures (caught and shown as a friendly sidebar/chat error)

## 8. Tech stack

Python · Streamlit · ChromaDB (persistent) · LangChain (loaders/splitters) ·
SentenceTransformers · rank-bm25 · OpenRouter API.
