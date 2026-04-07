from __future__ import annotations

from typing import List, Tuple
import numpy as np
from rag.data_state import get_data_hash
from rag.embeddings import embed_texts


# In-memory store
_CACHE = []  # [(embedding, answer, question)]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def find_similar(question: str, threshold: float = 0.65):
    if not _CACHE:
        return False, ""

    q_emb = np.array(embed_texts([question])[0])
    current_hash = get_data_hash()

    best_score = -1
    best_answer = ""

    for emb, answer, q, data_hash in _CACHE:

        # 🔥 IMPORTANT: skip old data
        if data_hash != current_hash:
            continue

        score = _cosine_similarity(q_emb, emb)
        print("Similarity score:", score)
        
        if score > best_score:
            best_score = score
            best_answer = answer

    if best_score >= threshold:
        return True, best_answer

    return False, ""

def store(question: str, answer: str):
    emb = np.array(embed_texts([question])[0])
    data_hash = get_data_hash()

    _CACHE.append((emb, answer, question, data_hash))

    if "cannot find" in answer.lower():
        return 