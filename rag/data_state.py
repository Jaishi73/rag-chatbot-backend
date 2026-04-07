from __future__ import annotations

import hashlib
from rag.paths import uploads_dir


def get_data_hash() -> str:
    """
    Generates a hash based on current uploaded files
    """
    p = uploads_dir()
    files = sorted([
   f"{f.name}:{f.stat().st_mtime}"
    for f in p.iterdir()
    if f.is_file()
])

    raw = "|".join(files)
    return hashlib.sha256(raw.encode()).hexdigest()