from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rag.embeddings import embed_texts
from rag.store import query
from rag.text_utils import normalize_text


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: float


def retrieve(
    *,
    question: str,
    top_k: int = 6,
    where: Optional[Dict[str, Any]] = None,
) -> List[RetrievedChunk]:
    question = normalize_text(question)
    if not question:
        return []

    qemb = embed_texts([question])[0]
    res = query(query_embedding=qemb, n_results=top_k, where=where)

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: List[RetrievedChunk] = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        if not doc:
            continue
        out.append(
            RetrievedChunk(
                id=str(cid),
                text=str(doc),
                metadata=dict(meta or {}),
                distance=float(dist) if dist is not None else 0.0,
            )
        )
    return out

