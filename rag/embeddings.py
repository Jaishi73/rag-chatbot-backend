from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


@lru_cache(maxsize=1)
def get_vectorizer() -> HashingVectorizer:
    """
    Lightweight, stateless text embedding using a hashing-based bag-of-words.
    This avoids heavy Torch/ONNX native dependencies and works fully locally.
    """
    return HashingVectorizer(
        n_features=4096,
        alternate_sign=False,
        norm="l2",
        stop_words="english",
    )


def embed_texts(texts: List[str]) -> List[List[float]]:
    vec = get_vectorizer()
    X = vec.transform(texts)
    dense = X.toarray().astype("float32")
    return dense.tolist()

