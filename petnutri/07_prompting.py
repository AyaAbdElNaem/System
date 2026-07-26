"""
07_prompting.py
=================
Stage 7 of the RAG pipeline: PROMPTING & GENERATION.

Responsibilities:
1. Classify incoming messages as "casual" (greetings/small talk) or
   "knowledge" (an actual nutrition question) so the retriever is only
   invoked when it's actually needed.
2. Build the strict, grounded RAG prompt (same rules as the original
   notebook: answer only from context, refuse if insufficient, always cite
   sources).
3. Call OpenRouter's chat-completions endpoint (streaming) using the API
   key from Streamlit secrets / environment variables.
"""

from __future__ import annotations

import json
import re
from typing import Generator, Iterable, Optional

import requests

try:
    from config import (
        APP_REFERRER,
        APP_TITLE,
        DEFAULT_OPENROUTER_MODEL,
        OPENROUTER_API_URL,
        get_openrouter_credentials,
    )
except ImportError:
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
    APP_TITLE = "PetNutri"
    APP_REFERRER = "https://petnutri.streamlit.app"

    def get_openrouter_credentials():
        import os

        return os.getenv("OPENROUTER_API_KEY"), os.getenv(
            "OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL
        )


# --------------------------------------------------------------------------
# 1. Casual vs. knowledge-question classification
# --------------------------------------------------------------------------
_GREETING_PATTERNS = [
    r"^\s*hi+\s*[!.]*\s*$",
    r"^\s*hello+\s*[!.]*\s*$",
    r"^\s*hey+\s*[!.]*\s*$",
    r"^\s*good\s+(morning|afternoon|evening|night)\s*[!.]*\s*$",
    r"^\s*how\s+are\s+you\??\s*$",
    r"^\s*what'?s\s+up\??\s*$",
    r"^\s*yo+\s*[!.]*\s*$",
]

_THANKS_PATTERNS = [
    r"^\s*thanks?\s*(you)?\s*[!.]*\s*$",
    r"^\s*thank\s+you\s*(so much|a lot)?\s*[!.]*\s*$",
    r"^\s*(ok|okay|great|cool|nice|awesome)\s*[!.]*\s*$",
    r"^\s*(bye|goodbye|see\s+you)\s*[!.]*\s*$",
]

_ALL_CASUAL_PATTERNS = _GREETING_PATTERNS + _THANKS_PATTERNS


def classify_query(text: str) -> str:
    """
    Classify a user message as ``"casual"`` or ``"knowledge"``.

    Casual conversation (greetings, thanks, small talk) never triggers the
    retriever - only genuine nutrition questions do. When in doubt (e.g.
    the message is longer than a few words, or asks something), we treat
    it as a knowledge question so we never silently skip retrieval for a
    real question.
    """
    normalized = text.strip().lower()
    if not normalized:
        return "casual"

    for pattern in _ALL_CASUAL_PATTERNS:
        if re.match(pattern, normalized):
            return "casual"

    # Very short inputs with no question mark and no nutrition-ish keywords
    # are more likely small talk than a real question.
    word_count = len(normalized.split())
    has_question_mark = "?" in normalized
    if word_count <= 3 and not has_question_mark:
        return "casual"

    return "knowledge"


_CASUAL_RESPONSES = {
    "greeting": (
        "Welcome to PetNutri 👋\n\n"
        "I'm your AI Pet Nutrition Assistant. I can help answer questions about "
        "dog and cat nutrition using trusted veterinary knowledge - things like "
        "feeding schedules, body condition, life-stage diets, food allergies, and "
        "weight management.\n\n"
        "What would you like to know about your pet's nutrition?"
    ),
    "thanks": (
        "You're welcome! 🐾 Let me know if you have any other questions about your "
        "pet's nutrition - I'm happy to help."
    ),
    "farewell": (
        "Take care, and give your pet a treat from me! 🐾 Come back anytime you "
        "have a nutrition question."
    ),
}


def get_casual_response(text: str) -> str:
    """Return a canned, friendly reply for a casual message (no retrieval)."""
    normalized = text.strip().lower()
    for pattern in _THANKS_PATTERNS[:2]:
        if re.match(pattern, normalized):
            return _CASUAL_RESPONSES["thanks"]
    for pattern in _THANKS_PATTERNS[2:]:
        if re.match(pattern, normalized):
            return _CASUAL_RESPONSES["farewell"]
    return _CASUAL_RESPONSES["greeting"]


# --------------------------------------------------------------------------
# 2. Strict, grounded prompt template (unchanged rules from the original)
# --------------------------------------------------------------------------
_NO_ANSWER_SENTENCE = (
    "The provided sources do not contain enough information to answer this question."
)

_SYSTEM_PROMPT = (
    "You are PetNutri, a grounded RAG assistant specialized in veterinary "
    "nutrition for dogs and cats.\n\n"
    "Rules:\n"
    "1. Use ONLY the provided context to answer the question. Never add "
    "background knowledge or external facts.\n"
    "2. If the answer cannot be fully found within the provided context, "
    f'reply exactly with: "{_NO_ANSWER_SENTENCE}"\n'
    "3. Do not assume or extrapolate. If the context is ambiguous, state "
    "only what is explicitly written.\n"
    "4. Always cite which source number(s) you used at the end of your "
    "answer, e.g. '(Source 1, Source 2)'.\n"
    "5. Keep a warm, professional, plain-language tone suitable for a pet owner."
)


def build_user_prompt(query: str, context_text: str) -> str:
    """Build the user-turn prompt combining the question and retrieved context."""
    if not context_text.strip():
        return (
            f"Question: {query}\n\n"
            "Context: (no relevant context was retrieved)\n\n"
            f'Since there is no context, reply exactly with: "{_NO_ANSWER_SENTENCE}"'
        )

    return f"Question: {query}\n\nContext:\n{context_text}"


# --------------------------------------------------------------------------
# 3. OpenRouter streaming call
# --------------------------------------------------------------------------
class GenerationError(Exception):
    """Raised when the OpenRouter API call fails or is misconfigured."""


def stream_answer(
    query: str,
    context_text: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> Generator[str, None, None]:
    """
    Stream the LLM's answer token-by-token (as text chunks) from OpenRouter.

    Yields
    ------
    str
        Incremental text chunks. Concatenate them to get the full answer.

    Raises
    ------
    GenerationError
        If no API key is configured, or the HTTP request fails.
    """
    resolved_key, resolved_model = get_openrouter_credentials()
    api_key = api_key or resolved_key
    model = model or resolved_model or DEFAULT_OPENROUTER_MODEL

    if not api_key:
        raise GenerationError(
            "No OpenRouter API key configured. Add OPENROUTER_API_KEY to your "
            "Streamlit secrets (deployed) or your .env file (local)."
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(query, context_text)},
        ],
        "temperature": temperature,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": APP_REFERRER,
        "X-Title": APP_TITLE,
    }

    try:
        response = requests.post(
            OPENROUTER_API_URL, headers=headers, json=payload, stream=True, timeout=60
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise GenerationError(f"Could not reach OpenRouter: {exc}") from exc

    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            data_str = raw_line[len("data:") :].strip()
            if data_str == "[DONE]":
                break
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = event.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            token = delta.get("content")
            if token:
                yield token
    except requests.exceptions.RequestException as exc:
        raise GenerationError(f"Streaming interrupted: {exc}") from exc


def generate_answer_sync(query: str, context_text: str, **kwargs) -> str:
    """Non-streaming convenience wrapper - collects the full answer at once."""
    return "".join(stream_answer(query, context_text, **kwargs))


if __name__ == "__main__":
    print(classify_query("hi there"))          # casual
    print(classify_query("thanks a lot!"))     # casual
    print(classify_query("What should I feed a lactating queen?"))  # knowledge
    print(build_user_prompt("test?", "[Source 1] ... context ..."))
