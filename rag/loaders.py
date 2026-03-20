from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from docx import Document
from pypdf import PdfReader

from rag.text_utils import normalize_text


@dataclass(frozen=True)
class LoadedDocument:
    source_file: str
    text: str
    metadata: dict


def load_pdf(path: Path) -> List[LoadedDocument]:
    reader = PdfReader(str(path))
    out: List[LoadedDocument] = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        txt = normalize_text(txt)
        if not txt:
            continue
        out.append(
            LoadedDocument(
                source_file=path.name,
                text=txt,
                metadata={"source_file": path.name, "page": i, "type": "pdf"},
            )
        )
    return out


def load_docx(path: Path) -> List[LoadedDocument]:
    doc = Document(str(path))
    # Keep paragraph boundaries
    parts: List[str] = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)

    # Tables to TSV-like lines
    for table in doc.tables:
        for row in table.rows:
            cells = [normalize_text((c.text or "")) for c in row.cells]
            cells = [c for c in cells if c]
            if cells:
                parts.append(" | ".join(cells))

    txt = normalize_text("\n\n".join(parts))
    if not txt:
        return []

    return [
        LoadedDocument(
            source_file=path.name,
            text=txt,
            metadata={"source_file": path.name, "type": "docx"},
        )
    ]


def _sniff_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return ","


def load_csv(path: Path, max_rows: int = 5000) -> List[LoadedDocument]:
    raw = path.read_bytes()
    # best-effort decoding
    text = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="ignore")

    sample = text[:4096]
    delim = _sniff_delimiter(sample)

    df = pd.read_csv(io.StringIO(text), sep=delim, engine="python")
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)

    # Serialize rows into readable lines; keep column names
    cols = [str(c) for c in df.columns]
    lines: List[str] = []
    lines.append("COLUMNS: " + " | ".join(cols))
    for idx, row in df.iterrows():
        values = []
        for c in cols:
            v = row.get(c)
            if pd.isna(v):
                values.append("")
            else:
                values.append(str(v))
        lines.append(f"ROW {idx}: " + " | ".join(values))

    txt = normalize_text("\n".join(lines))
    if not txt:
        return []

    return [
        LoadedDocument(
            source_file=path.name,
            text=txt,
            metadata={"source_file": path.name, "type": "csv"},
        )
    ]


def load_any(path: Path) -> List[LoadedDocument]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".docx":
        return load_docx(path)
    if suffix == ".csv":
        return load_csv(path)
    raise ValueError(f"Unsupported file type: {suffix}")

