"""Single reusable initialisation of embeddings, vector store, LLM and graph."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from backend.config import Settings, load_settings


class ConfigError(RuntimeError):
    pass


def get_llm(settings: Settings | None = None):
    """Chat model for the configured provider. Add providers here to keep the app provider-agnostic."""
    s = settings or load_settings()
    if s.llm_provider == "gemini":
        if not s.gemini_api_key:
            raise ConfigError("GEMINI_API_KEY is not set (env var or Streamlit secret).")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=s.gemini_model, google_api_key=s.gemini_api_key, temperature=0)
    elif s.llm_provider == "groq":
        if not s.groq_api_key:
            raise ConfigError("GROQ_API_KEY is not set (env var or Streamlit secret).")
        from langchain_groq import ChatGroq

        return ChatGroq(model=s.groq_model, api_key=s.groq_api_key, temperature=0)
    raise ConfigError(f"Unsupported LLM_PROVIDER '{s.llm_provider}'. Implement it in backend/services.py:get_llm.")


@dataclass
class Services:
    settings: Settings
    vector_store: Any
    llm: Any
    graph: Any


def get_services(settings: Settings | None = None) -> Services:
    """Load embeddings (once), open the vector store, build the LLM and the LangGraph workflow."""
    from backend.graph import build_graph
    from ingestion.pipeline import manifest_path
    from ingestion.vector_store import get_embeddings, get_vector_store

    s = settings or load_settings()
    embeddings = get_embeddings(s.embedding_model)
    mp = manifest_path(s.data_dir)
    if mp.exists():
        recorded = (json.loads(mp.read_text(encoding="utf-8")).get("embedding") or {}).get("model")
        if recorded and recorded != s.embedding_model:
            raise ConfigError(f"Index was built with '{recorded}' but EMBEDDING_MODEL is '{s.embedding_model}'. Rebuild with --reset.")
    store = get_vector_store(s, embeddings)
    llm = get_llm(s)
    return Services(s, store, llm, build_graph(store, llm, s))
