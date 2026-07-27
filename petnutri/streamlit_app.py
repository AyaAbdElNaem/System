from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from pipeline_utils import load_stage

load_dotenv()  # local .env support; no-op if the file doesn't exist

_store = load_stage("05_create_chroma_store")
_retrieve = load_stage("06_retrieve_context")
_prompting = load_stage("07_prompting")

get_or_build_collection = _store.get_or_build_collection
get_collection_stats = _store.get_collection_stats
VectorStoreError = _store.VectorStoreError

build_context_package = _retrieve.build_context_package
RetrievalError = _retrieve.RetrievalError

classify_query = _prompting.classify_query
get_casual_response = _prompting.get_casual_response
stream_answer = _prompting.stream_answer
GenerationError = _prompting.GenerationError
NO_ANSWER_SENTENCE = _prompting.NO_ANSWER_SENTENCE

try:
    from config import APP_TITLE, STYLE_CSS_PATH, get_openrouter_credentials
except ImportError:
    APP_TITLE = "PetNutri"
    STYLE_CSS_PATH = Path(__file__).resolve().parent / "assets" / "style.css"

    def get_openrouter_credentials():
        return os.getenv("OPENROUTER_API_KEY"), os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


# --------------------------------------------------------------------------
# Page config + styling
# --------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🐾", layout="centered")


def _inject_css() -> None:
    if STYLE_CSS_PATH.exists():
        st.markdown(f"<style>{STYLE_CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


_inject_css()

_PAW_SVG_HERO = """
<div class="pn-hero-image">
<svg viewBox="0 0 700 320" xmlns="http://www.w3.org/2000/svg">
  <rect width="700" height="320" fill="#EFE6D8"/>
  <circle cx="120" cy="90" r="46" fill="#D9A67E" opacity="0.55"/>
  <circle cx="600" cy="250" r="70" fill="#5C6B2F" opacity="0.18"/>
  <g transform="translate(230,60)">
    <ellipse cx="120" cy="150" rx="95" ry="80" fill="#C98B5E"/>
    <circle cx="70" cy="70" r="34" fill="#C98B5E"/>
    <circle cx="170" cy="70" r="34" fill="#C98B5E"/>
    <ellipse cx="120" cy="150" rx="60" ry="50" fill="#F3E3D2"/>
    <circle cx="98" cy="140" r="7" fill="#2B2A28"/>
    <circle cx="142" cy="140" r="7" fill="#2B2A28"/>
    <ellipse cx="120" cy="165" rx="10" ry="7" fill="#2B2A28"/>
  </g>
  <g fill="#5C6B2F" opacity="0.35">
    <circle cx="600" cy="60" r="10"/>
    <circle cx="625" cy="80" r="7"/>
    <circle cx="580" cy="80" r="7"/>
    <circle cx="615" cy="45" r="6"/>
    <circle cx="590" cy="45" r="6"/>
  </g>
</svg>
</div>
"""

_PAW_ICON = "🐾"

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, sources?}
if "db_ready" not in st.session_state:
    st.session_state.db_ready = False


def go_to(page: str) -> None:
    st.session_state.page = page


# --------------------------------------------------------------------------
# Sidebar - knowledge base status + controls (always visible)
# --------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="pn-sidebar-title">🐾 PetNutri</div>', unsafe_allow_html=True)

        # Knowledge-base status (documents/chunks/embedding model/last built)
        # is intentionally not rendered in the UI - kept internal only, so
        # we still know whether the database is ready to gate chat.
        try:
            stats = get_collection_stats()
            st.session_state.db_ready = stats["database_ready"]
        except Exception as exc:  # noqa: BLE001
            st.session_state.db_ready = False
            st.error(f"Could not read database status: {exc}")

        if st.button("🔄 Rebuild Database", use_container_width=True, type="secondary"):
            with st.spinner("Rebuilding the vector database..."):
                try:
                    get_or_build_collection(force_rebuild=True)
                    st.success("Database rebuilt successfully.")
                except VectorStoreError as exc:
                    st.error(str(exc))
            st.rerun()

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🧹 Clear Chat", use_container_width=True, type="secondary"):
                st.session_state.messages = []
                st.rerun()
        with col_b:
            if st.button("↩️ Reset", use_container_width=True, type="secondary"):
                st.session_state.messages = []
                st.session_state.page = "home"
                st.rerun()


# --------------------------------------------------------------------------
# Home page
# --------------------------------------------------------------------------
def render_home() -> None:
    st.markdown(
        f"""
        <div class="pn-topbar">
            <div class="pn-logo">{_PAW_ICON} {APP_TITLE}</div>
            <div>{_PAW_ICON}</div>
        </div>
        <div class="pn-hero-title">
            Personalized Nutrition<br>for a <span class="accent">Happier Paw</span> {_PAW_ICON}
        </div>
        <div class="pn-hero-subtitle">
            Every pet is unique. Our AI-driven platform crafts perfectly balanced
            diets tailored to your companion's breed, age, and health needs.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Start AI Consultation", type="primary"):
        go_to("chat")
        st.rerun()

    st.markdown(_PAW_SVG_HERO, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Chat page
# --------------------------------------------------------------------------
def render_message(message: dict) -> None:
    role = message["role"]
    content = message["content"]

    if role == "user":
        st.markdown(f'<div class="pn-bubble-user">{content}</div>', unsafe_allow_html=True)
    elif message.get("casual"):
        st.markdown(f'<div class="pn-bubble-assistant pn-bubble-casual">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="pn-bubble-assistant">{content}</div>', unsafe_allow_html=True)
        sources = message.get("sources") or []
        if sources:
            with st.expander(f"📚 Sources ({len(sources)})"):
                for src in sources:
                    link = (
                        f'<a href="{src["url_reference"]}" target="_blank">reference</a>'
                        if src.get("url_reference")
                        else ""
                    )
                    st.markdown(
                        f"""<div class="pn-source-card">
                        <b>Source {src['index']}: {src['title']}</b><br>
                        {src['category']} · {src['source']} · relevance {src['score']:.2f} {link}
                        </div>""",
                        unsafe_allow_html=True,
                    )


def handle_submission(user_text: str) -> None:
    user_text = user_text.strip()
    if not user_text:
        return  # ignore empty/invalid queries

    st.session_state.messages.append({"role": "user", "content": user_text})

    category = classify_query(user_text)

    if category == "casual":
        reply = get_casual_response(user_text)
        st.session_state.messages.append({"role": "assistant", "content": reply, "casual": True})
        return

    if not st.session_state.db_ready:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "The knowledge base hasn't been built yet. Please click **Rebuild "
                    "Database** in the sidebar, then ask your question again."
                ),
                "casual": True,
            }
        )
        return

    try:
        collection = get_or_build_collection()
        package = build_context_package(user_text, collection)
    except (VectorStoreError, RetrievalError) as exc:
        st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {exc}", "casual": True})
        return
    except Exception as exc:  # noqa: BLE001
        st.session_state.messages.append(
            {"role": "assistant", "content": f"⚠️ Unexpected retrieval error: {exc}", "casual": True}
        )
        return

    if not package["has_sufficient_context"]:
        st.session_state.messages.append({"role": "assistant", "content": NO_ANSWER_SENTENCE})
        return

    try:
        answer_text = "".join(
            stream_answer(user_text, package["context_text"])
        )
    except GenerationError as exc:
        st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {exc}", "casual": True})
        return
    except Exception as exc:  # noqa: BLE001
        st.session_state.messages.append(
            {"role": "assistant", "content": f"⚠️ Unexpected generation error: {exc}", "casual": True}
        )
        return


    _FALLBACK_MARKER = "do not contain enough information"
    is_fallback_answer = _FALLBACK_MARKER in answer_text.strip().lower()
    message = {"role": "assistant", "content": answer_text}
    if not is_fallback_answer:
        message["sources"] = package["sources"]
    st.session_state.messages.append(message)



def render_chat() -> None:
    st.markdown(
        f"""
        <div class="pn-topbar" style="justify-content: flex-end;">
            <div class="pn-logo">{_PAW_ICON} {APP_TITLE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("← Back to Home", key="back_home_btn", type="secondary"):
        go_to("home")
        st.rerun()

    st.markdown(
        f"""
        <div class="pn-badge">✨ Nutrition Expert AI</div>
        <div class="pn-chat-title">How can I help your pet today?</div>
        <div class="pn-chat-subtitle">Expert guidance for a healthier, happier life.</div>
        """,
        unsafe_allow_html=True,
    )

    for message in st.session_state.messages:
        render_message(message)

    if not st.session_state.messages:
        st.caption("Ask about ingredients, feeding schedules, allergies, weight management, and more.")

    with st.form("chat_form", clear_on_submit=True):
        col_input, col_button = st.columns([5, 1.3])
        with col_input:
            user_text = st.text_input(
                "message",
                placeholder="Ask about ingredients, safety, or diet plans...",
                label_visibility="collapsed",
            )
        with col_button:
            submitted = st.form_submit_button("Consult AI →", use_container_width=True)

    if submitted and user_text.strip():
        with st.spinner("PetNutri is thinking..."):
            handle_submission(user_text)
        st.rerun()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    render_sidebar()

    if st.session_state.page == "home":
        render_home()
    else:
        render_chat()


if __name__ == "__main__":
    main()
