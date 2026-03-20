from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    p = project_root() / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def uploads_dir() -> Path:
    p = data_dir() / "uploads"
    p.mkdir(parents=True, exist_ok=True)
    return p


def chroma_dir() -> Path:
    p = data_dir() / "chroma"
    p.mkdir(parents=True, exist_ok=True)
    return p


def index_dir() -> Path:
    p = data_dir() / "index"
    p.mkdir(parents=True, exist_ok=True)
    return p


def chats_dir() -> Path:
    p = data_dir() / "chats"
    p.mkdir(parents=True, exist_ok=True)
    return p

