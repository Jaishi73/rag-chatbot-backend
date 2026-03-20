from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple

from rag.embeddings import embed_texts
from rag.loaders import LoadedDocument, load_any
from rag.text_utils import chunk_text, normalize_text
from rag.store import upsert_chunks


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_fingerprint(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def ingest_file(path: Path) -> Tuple[str, int]:
    """
    Returns: (file_hash, num_chunks_upserted)
    """
    docs: List[LoadedDocument] = load_any(path)
    file_hash = file_fingerprint(path)

    chunk_ids: List[str] = []
    chunk_texts: List[str] = []
    chunk_metas: List[dict] = []

    for doc_idx, d in enumerate(docs):
        chunks = chunk_text(text=d.text)
        for chunk_idx, ch in enumerate(chunks):
            ch = normalize_text(ch)
            if not ch:
                continue

            # Stable IDs so re-ingest updates instead of duplicates.
            base = f"{path.name}:{file_hash}:{doc_idx}:{chunk_idx}"
            cid = hashlib.sha256(base.encode("utf-8")).hexdigest()
            meta = dict(d.metadata)
            meta.update(
                {
                    "source_file": path.name,
                    "file_hash": file_hash,
                    "doc_part": doc_idx,
                    "chunk_index": chunk_idx,
                }
            )
            chunk_ids.append(cid)
            chunk_texts.append(ch)
            chunk_metas.append(meta)

    if not chunk_texts:
        return file_hash, 0

    vectors = embed_texts(chunk_texts)
    upsert_chunks(ids=chunk_ids, documents=chunk_texts, embeddings=vectors, metadatas=chunk_metas)
    return file_hash, len(chunk_texts)

