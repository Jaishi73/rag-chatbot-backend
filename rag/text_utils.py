from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class TextChunk:
    id: str
    text: str
    metadata: dict


_WS_RE = re.compile(r"[ \t]+")


def normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = _WS_RE.sub(" ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(
    *,
    text: str,
    chunk_size_chars: int = 3500,
    chunk_overlap_chars: int = 400,
) -> List[str]:
    """
    Simple char-based chunker that is stable across Python versions and doesn't
    require tokenizers. Works well enough for 1-90 page business docs.
    """
    text = normalize_text(text)
    if not text:
        return []

    if chunk_overlap_chars >= chunk_size_chars:
        raise ValueError("chunk_overlap_chars must be < chunk_size_chars")

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size_chars, n)

        # Prefer to break on paragraph boundary when possible
        window = text[start:end]
        cut = window.rfind("\n\n")
        if cut != -1 and cut > int(chunk_size_chars * 0.6):
            end = start + cut

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(0, end - chunk_overlap_chars)

    return chunks


def safe_join_snippets(snippets: Iterable[str], max_chars: int = 8000) -> str:
    out: List[str] = []
    total = 0
    for snip in snippets:
        snip = normalize_text(snip)
        if not snip:
            continue
        if total + len(snip) + 2 > max_chars:
            remaining = max(0, max_chars - total - 2)
            if remaining > 0:
                out.append(snip[:remaining])
            break
        out.append(snip)
        total += len(snip) + 2
    return "\n\n".join(out).strip()

