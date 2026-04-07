from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache
import hashlib

from rag.embeddings import embed_texts
from rag.store import query
from rag.text_utils import normalize_text
from rag.logger import get_logger

logger = get_logger()

@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: float


# -------------------------------
# 🔹 CACHE: Query Embedding
# -------------------------------
@lru_cache(maxsize=200)
def _cached_embedding(question: str) -> Tuple[float, ...]:
    emb = embed_texts([question])[0]
    return tuple(emb)  # hashable for cache


# -------------------------------
# 🔹 CACHE: Retrieval Results
# -------------------------------
@lru_cache(maxsize=200)
def _cached_retrieval(
    question: str,
    top_k: int,
    where_key: str,
) -> Tuple[Tuple[str, str, Dict[str, Any], float], ...]:

    qemb = list(_cached_embedding(question))
    where = eval(where_key) if where_key else None

    res = query(query_embedding=qemb, n_results=top_k, where=where)

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    results = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        if not doc:
            continue
        results.append((str(cid), str(doc), dict(meta or {}), float(dist or 0.0)))

    return tuple(results)


def retrieve(
    *,
    question: str,
    top_k: int = 6,
    where: Optional[Dict[str, Any]] = None,
) -> List[RetrievedChunk]:

    question = normalize_text(question)
    if not question:
        return []

    where_key = str(where) if where else ""

    cached = _cached_retrieval(question, top_k, where_key)

    return [
        RetrievedChunk(id=c[0], text=c[1], metadata=c[2], distance=c[3])
        for c in cached
    ]