"""Single reusable initialisation of embeddings, vector store, LLM and graph."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any

from backend.config import Settings, load_settings


class ConfigError(RuntimeError):
    pass


log = logging.getLogger(__name__)


def _is_rate_limit_error(error: Exception) -> bool:
    """Recognize common Gemini and Groq quota/rate-limit responses."""
    for value in (getattr(error, "status_code", None), getattr(getattr(error, "response", None), "status_code", None)):
        if str(value) == "429":
            return True
    message = str(error).lower()
    return any(token in message for token in (
        "429", "rate limit", "rate_limit", "too many requests", "resource_exhausted",
        "quota exceeded", "requests per minute", "tokens per minute", "daily limit",
    ))


class _AlternatingStructuredOutput:
    def __init__(self, providers, coordinator):
        self.providers = providers
        self.coordinator = coordinator

    def invoke(self, value, config=None, **kwargs):
        with self.coordinator._lock:
            start = self.coordinator._next_index
        last_rate_limit = None
        for offset in range(len(self.providers)):
            provider_index = (start + offset) % len(self.providers)
            name, runnable = self.providers[provider_index]
            try:
                log.info("Running structured ingestion extraction with %s", name)
                result = runnable.invoke(value, config=config, **kwargs)
                with self.coordinator._lock:
                    self.coordinator._next_index = (provider_index + 1) % len(self.providers)
                return result
            except Exception as error:
                if not _is_rate_limit_error(error):
                    raise
                last_rate_limit = error
                log.warning("%s rate limit reached during ingestion; trying the other configured provider", name)
        with self.coordinator._lock:
            self.coordinator._next_index = (start + 1) % len(self.providers)
        raise last_rate_limit


class AlternatingIngestionLLM:
    """Build structured-output runnables that alternate providers and fail over on 429s."""
    def __init__(self, providers):
        self.providers = providers
        self._next_index = 0
        self._lock = threading.Lock()

    def with_structured_output(self, schema, **kwargs):
        runnables = []
        for name, model in self.providers:
            try:
                runnables.append((name, model.with_structured_output(schema, **kwargs)))
            except Exception as error:
                log.warning("%s does not support the requested structured extraction: %s", name, error)
        if not runnables:
            raise ConfigError("No configured provider supports structured ingestion output.")
        return _AlternatingStructuredOutput(runnables, self)


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



def get_ingestion_llm(settings: Settings | None = None) -> AlternatingIngestionLLM:
    """Create Gemini and Groq clients for round-robin ingestion with rate-limit fallback."""
    s = settings or load_settings()
    providers = []
    if s.gemini_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            providers.append(("Gemini", ChatGoogleGenerativeAI(
                model=s.gemini_model, google_api_key=s.gemini_api_key, temperature=0,
            )))
        except Exception as error:
            log.warning("Gemini ingestion client unavailable: %s", error)
    if s.groq_api_key:
        try:
            from langchain_groq import ChatGroq

            providers.append(("Groq", ChatGroq(
                model=s.groq_model, api_key=s.groq_api_key, temperature=0,
            )))
        except Exception as error:
            log.warning("Groq ingestion client unavailable: %s", error)
    if not providers:
        raise ConfigError("Set GEMINI_API_KEY and/or GROQ_API_KEY to extract ingestion insights.")
    if len(providers) == 1:
        log.warning("Only %s is configured for ingestion; set both GEMINI_API_KEY and GROQ_API_KEY for alternation and fallback.", providers[0][0])
    return AlternatingIngestionLLM(providers)

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
