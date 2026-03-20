from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rag.paths import index_dir


INDEX_FILE = index_dir() / "rag_index.pkl"


@dataclass
class _Index:
    ids: List[str]
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    embeddings: np.ndarray  # shape (n, d), float32, L2-normalized


def _load_index() -> _Index:
    if not INDEX_FILE.exists():
        return _Index(ids=[], documents=[], metadatas=[], embeddings=np.zeros((0, 0), dtype=np.float32))
    with INDEX_FILE.open("rb") as f:
        obj = pickle.load(f)
    # Basic validation / forward compatibility
    ids = list(obj.get("ids", []))
    documents = list(obj.get("documents", []))
    metadatas = list(obj.get("metadatas", []))
    emb = obj.get("embeddings")
    if emb is None:
        emb = np.zeros((0, 0), dtype=np.float32)
    emb = np.asarray(emb, dtype=np.float32)
    return _Index(ids=ids, documents=documents, metadatas=metadatas, embeddings=emb)


def _save_index(ix: _Index) -> None:
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_FILE.with_suffix(".tmp")
    payload = {
        "ids": ix.ids,
        "documents": ix.documents,
        "metadatas": ix.metadatas,
        "embeddings": ix.embeddings,
    }
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(INDEX_FILE)


def upsert_chunks(
    *,
    ids: List[str],
    documents: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
) -> None:
    if not ids:
        return

    ix = _load_index()
    new_emb = np.asarray(embeddings, dtype=np.float32)
    if new_emb.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")

    # Ensure consistent dimension
    if ix.embeddings.size == 0:
        ix.embeddings = np.zeros((0, new_emb.shape[1]), dtype=np.float32)
    elif ix.embeddings.shape[1] != new_emb.shape[1]:
        raise ValueError("Embedding dimension mismatch. Clear data/index/ to reindex.")

    pos = {cid: i for i, cid in enumerate(ix.ids)}
    for cid, doc, meta, emb_row in zip(ids, documents, metadatas, new_emb):
        if cid in pos:
            i = pos[cid]
            ix.documents[i] = doc
            ix.metadatas[i] = meta
            ix.embeddings[i] = emb_row
        else:
            ix.ids.append(cid)
            ix.documents.append(doc)
            ix.metadatas.append(meta)
            ix.embeddings = np.vstack([ix.embeddings, emb_row.reshape(1, -1)])

    _save_index(ix)


def delete_by_source_file(source_file: str) -> None:
    ix = _load_index()
    if not ix.ids:
        return

    keep_idx = [i for i, m in enumerate(ix.metadatas) if (m or {}).get("source_file") != source_file]
    if len(keep_idx) == len(ix.ids):
        return

    ix.ids = [ix.ids[i] for i in keep_idx]
    ix.documents = [ix.documents[i] for i in keep_idx]
    ix.metadatas = [ix.metadatas[i] for i in keep_idx]
    ix.embeddings = ix.embeddings[keep_idx, :] if keep_idx else np.zeros((0, ix.embeddings.shape[1]), dtype=np.float32)
    _save_index(ix)


def query(
    *,
    query_embedding: List[float],
    n_results: int = 6,
    where: Optional[Dict[str, Any]] = None,
):
    ix = _load_index()
    if not ix.ids:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    if ix.embeddings.size == 0 or ix.embeddings.shape[1] != q.shape[1]:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    # Optional metadata filter (simple equality on fields)
    candidates = list(range(len(ix.ids)))
    if where:
        def _match(meta: Dict[str, Any]) -> bool:
            for k, v in where.items():
                if (meta or {}).get(k) != v:
                    return False
            return True
        candidates = [i for i in candidates if _match(ix.metadatas[i])]
        if not candidates:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    E = ix.embeddings[candidates, :]
    # Cosine distance = 1 - cosine similarity (embeddings assumed L2-normalized)
    sims = (E @ q.T).reshape(-1)
    k = min(max(1, int(n_results)), sims.shape[0])
    top = np.argpartition(-sims, kth=k - 1)[:k]
    # Sort top-k by similarity desc
    top = top[np.argsort(-sims[top])]

    ids = [ix.ids[candidates[i]] for i in top.tolist()]
    docs = [ix.documents[candidates[i]] for i in top.tolist()]
    metas = [ix.metadatas[candidates[i]] for i in top.tolist()]
    dists = [float(1.0 - sims[i]) for i in top.tolist()]

    return {"ids": [ids], "documents": [docs], "metadatas": [metas], "distances": [dists]}

