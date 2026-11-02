"""Central configuration. Env vars locally, Streamlit secrets in deployment."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def get_setting(name: str, default: str | None = None) -> str | None:
    """Environment variable first, then Streamlit secrets, then default."""
    value = os.environ.get(name)
    if value not in (None, ""):
        return value
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets file / not running in Streamlit
        pass
    return default


def _path(value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def _float_or_none(value: str | None) -> float | None:
    if value is None or str(value).strip().lower() in ("", "none", "off"):
        return None
    return float(value)


def _bool(value: str | None) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    gemini_api_key: str
    gemini_model: str
    embedding_model: str
    pdf_input_dir: Path
    chroma_dir: Path
    data_dir: Path
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_distance: float | None
    max_retries: int
    min_support: float
    history_turns: int
    ocr_enabled: bool
    chroma_cloud_api_key: str
    chroma_tenant: str
    chroma_database: str
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"


def load_settings() -> Settings:
    g = get_setting
    return Settings(
        llm_provider=(g("LLM_PROVIDER", "gemini") or "gemini").lower(),
        gemini_api_key=g("GEMINI_API_KEY", "") or "",
        # Default is a sensible starting point; set GEMINI_MODEL to whichever current model you use.
        gemini_model=g("GEMINI_MODEL", "") or "gemini-2.5-flash",
        groq_api_key=g("GROQ_API_KEY", "") or "",
        groq_model=g("GROQ_MODEL", "") or "llama-3.3-70b-versatile",
        embedding_model=g("EMBEDDING_MODEL", "") or "sentence-transformers/all-MiniLM-L6-v2",
        pdf_input_dir=_path(g("PDF_INPUT_DIR", "Interview_Experiences")),
        chroma_dir=_path(g("CHROMA_PERSIST_DIR", "chroma_db")),
        data_dir=_path(g("DATA_DIR", "data")),
        collection_name=g("CHROMA_COLLECTION", "campusprep") or "campusprep",
        chunk_size=int(g("CHUNK_SIZE", "1000")),
        chunk_overlap=int(g("CHUNK_OVERLAP", "150")),
        top_k=int(g("RETRIEVAL_TOP_K", "5")),
        max_distance=_float_or_none(g("RETRIEVAL_MAX_DISTANCE", "0.85")),
        max_retries=int(g("MAX_ANSWER_RETRIES", "1")),
        min_support=float(g("CITATION_MIN_SUPPORT", "0.5")),
        history_turns=int(g("HISTORY_TURNS", "3")),
        ocr_enabled=_bool(g("OCR_ENABLED", "false")),
        chroma_cloud_api_key=g("CHROMA_CLOUD_API_KEY", "") or "",
        chroma_tenant=g("CHROMA_TENANT", "") or "",
        chroma_database=g("CHROMA_DATABASE", "") or "",
    )
