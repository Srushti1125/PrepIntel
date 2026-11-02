"""Company-scoped retriever. The company filter is applied inside the vector query itself."""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class CompanyRetriever(BaseRetriever):
    vector_store: Any
    company_id: str
    top_k: int = 5
    max_distance: Optional[float] = None
    years: Optional[list[str]] = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        if not self.company_id:
            raise ValueError("CompanyRetriever requires a company_id; refusing to search all companies.")
        flt: dict = {"company_id": self.company_id}
        if self.years:
            flt = {"$and": [flt, {"year": {"$in": list(self.years)}}]}
        results = self.vector_store.similarity_search_with_score(query, k=self.top_k, filter=flt)
        docs, seen = [], set()
        for doc, distance in results:
            if doc.metadata.get("company_id") != self.company_id:  # defence in depth
                continue
            if self.max_distance is not None and distance > self.max_distance:
                continue
            key = (doc.metadata.get("source_path"), doc.metadata.get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)
            docs.append(Document(page_content=doc.page_content, metadata={**doc.metadata, "distance": float(distance)}))
        return docs
