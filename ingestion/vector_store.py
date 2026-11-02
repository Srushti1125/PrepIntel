"""Embedding model + ChromaDB access. The embedding model is loaded once and reused."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma


@lru_cache(maxsize=4)
def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
    """Local CPU sentence-transformers embeddings (no external API). Cached per process."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def embedding_dimension(embeddings: Any) -> int:
    return len(embeddings.embed_query("dimension probe"))


def _client(settings):
    if settings.chroma_cloud_api_key:
        import chromadb

        return chromadb.CloudClient(
            tenant=settings.chroma_tenant or None,
            database=settings.chroma_database or None,
            api_key=settings.chroma_cloud_api_key,
        )
    return None


def get_vector_store(settings, embeddings: Any) -> Chroma:
    """Open (or create) the collection. Cosine space so distances are in [0, 2]."""
    kwargs: dict[str, Any] = dict(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    client = _client(settings)
    if client is not None:
        kwargs["client"] = client
    else:
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        kwargs["persist_directory"] = str(settings.chroma_dir)
    return Chroma(**kwargs)


def delete_source(store: Chroma, source_path: str) -> int:
    ids = store.get(where={"source_path": source_path}, include=[])["ids"]
    if ids:
        store.delete(ids=ids)
    return len(ids)


def add_chunks(store: Chroma, chunks, ids, batch: int = 64) -> None:
    for i in range(0, len(chunks), batch):
        store.add_documents(chunks[i : i + batch], ids=ids[i : i + batch])
