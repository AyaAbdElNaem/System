from __future__ import annotations

import re

# Symbols kept because they appear in nutrition figures, e.g. "10%-20%",
# "2-4 g/day/kg", "< 5%".
_KEEP_SYMBOLS = r"a-z0-9\s\-%\+/=\><"


def normalize_raw_text(text: str) -> str:
    """
    Normalize a raw document's text prior to chunking.

    - Converts Windows-style CRLF/escaped ``\\r\\n`` sequences to ``\\n``.
    - Collapses 3+ consecutive blank lines to a single blank line.
    - Strips trailing whitespace from each line.

    Markdown headers (#, ##, ###) are left untouched so
    ``MarkdownHeaderTextSplitter`` can still split on them correctly.
    """
    if not text:
        return ""

    text = text.replace("\\r\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_chunk_text(text: str) -> str:
    """
    Aggressively clean a single chunk's text for use in lexical/semantic
    search (lowercased, punctuation stripped except a few meaningful
    symbols). This mirrors the original ``clean_final_chunk`` function so
    retrieval quality and behavior stay identical.
    """
    if not text:
        return ""

    text = text.replace("\\n", "\n").lower()
    text = re.sub(rf"[^{_KEEP_SYMBOLS}]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    sample = "## Obesity\r\n\r\nDogs overweight/obese: 59%\r\n\r\n\r\nCats: 61%"
    normalized = normalize_raw_text(sample)
    print("Normalized:\n", repr(normalized))
    print("Cleaned chunk:\n", clean_chunk_text(normalized))
