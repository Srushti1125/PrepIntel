"""LangGraph workflow: classify_query -> retrieve_documents -> generate_answer -> verify_citations."""
from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from backend.citations import (
    build_citations, format_response, format_sources, number_refs, verify_answer,
)
from backend.prompts import INSUFFICIENT, RETRY_FEEDBACK
from backend.rag_chain import (
    answer_inputs, build_answer_chain, build_condense_chain, history_to_text, label_documents,
)
from backend.retriever import CompanyRetriever

log = logging.getLogger(__name__)

MAX_RETRIEVALS = 2
BROAD_TYPES = {"study_plan", "process", "topics", "resources", "tips"}

_PATTERNS = [
    ("source_request", r"\b(sources?|citations?|references?|where did you (?:get|find)|which (?:pdf|document)s?)\b"),
    ("study_plan", r"\b(study plan|roadmap|schedule|\d+[- ]day|(?:seven|five|ten|three|two|one|four|six)[- ]day|week plan)\b"),
    ("resources", r"\b(resources?|links?|websites?|courses?|books?|sheets?|youtube|channels?)\b"),
    ("tips", r"\b(tips?|advice|how to prepare|preparation|strategy)\b"),
    ("questions", r"\b(questions?|asked|problems?)\b"),
    ("process", r"\b(process|rounds?|procedure|stages?|pattern|differences?|compare|across)\b"),
    ("topics", r"\b(topics?|focus|concepts?|syllabus|subjects?)\b"),
]
_FOLLOWUP = re.compile(r"\b(it|that|this|those|them|they|these|above|previous|same|more|also|else|another|again)\b", re.I)


class GraphState(TypedDict, total=False):
    company_id: str
    company_name: str
    years: list[str]
    question: str
    history: list[dict]
    search_query: str
    query_type: str
    documents: list[Document]
    answer: str
    citations: list[dict]
    verification_status: str
    attempts: int
    retrieval_attempts: int
    feedback: str
    error: str


def classify(question: str) -> str:
    q = question.lower()
    for name, pat in _PATTERNS:
        if re.search(pat, q):
            return name
    return "general"


def _last_citations(history: list[dict]) -> list[dict]:
    for m in reversed(history or []):
        if m["role"] == "assistant" and m.get("citations"):
            return m["citations"]
    return []


def build_graph(vector_store: Any, llm: Any, settings: Any):
    answer_chain = build_answer_chain(llm)
    condense_chain = build_condense_chain(llm)

    # ---- nodes -------------------------------------------------------------------------
    def classify_query(state: GraphState) -> dict:
        question, history = state["question"], state.get("history", [])
        qtype = classify(question)
        search_query = question
        if history and qtype != "source_request" and (len(question.split()) <= 10 or _FOLLOWUP.search(question)):
            try:
                rewritten = condense_chain.invoke({"history": history_to_text(history, settings.history_turns), "question": question}).strip()
                if rewritten:
                    search_query = rewritten
            except Exception as e:  # fall back to the raw question
                log.warning("condense failed: %s", e)
        return {"query_type": qtype, "search_query": search_query, "attempts": 0, "retrieval_attempts": 0,
                "feedback": "", "error": "", "citations": []}

    def reuse_sources(state: GraphState) -> dict:
        cites = _last_citations(state.get("history", []))
        return {"answer": f"These are the sources for my previous answer:\n\n{format_sources(cites)}",
                "citations": cites, "verification_status": "reused"}

    def retrieve_documents(state: GraphState) -> dict:
        n = state.get("retrieval_attempts", 0)
        k = settings.top_k * (2 if state.get("query_type") in BROAD_TYPES or n > 0 else 1)
        query = state["search_query"] if n == 0 else state["question"]
        retriever = CompanyRetriever(vector_store=vector_store, company_id=state["company_id"], top_k=k,
                                     max_distance=settings.max_distance, years=state.get("years"))
        docs = retriever.invoke(query)
        update: dict = {"documents": docs, "retrieval_attempts": n + 1}
        if not docs and n + 1 >= MAX_RETRIEVALS:
            update.update(answer=INSUFFICIENT, verification_status="no_evidence", citations=[])
        return update

    def generate_answer(state: GraphState) -> dict:
        docs_by_id = label_documents(state["documents"])
        inputs = answer_inputs(state["company_name"], state["question"], docs_by_id,
                               history_to_text(state.get("history", []), settings.history_turns), state.get("feedback", ""))
        try:
            text = answer_chain.invoke(inputs)
        except Exception as e:
            log.exception("generation failed")
            return {"error": str(e), "answer": "", "verification_status": "error", "attempts": state.get("attempts", 0) + 1}
        return {"answer": text, "attempts": state.get("attempts", 0) + 1, "error": ""}

    def verify_citations(state: GraphState) -> dict:
        docs_by_id = label_documents(state["documents"])
        passages = {sid: d.page_content for sid, d in docs_by_id.items()}
        res = verify_answer(state["answer"], passages, settings.min_support)
        if res.status == "no_evidence":
            return {"answer": INSUFFICIENT, "citations": [], "verification_status": "no_evidence"}
        if res.status == "failed" and state["attempts"] <= settings.max_retries:
            items = "\n".join(f"- {u}" for u in res.unsupported[:8])
            return {"verification_status": "retry", "feedback": RETRY_FEEDBACK.format(items=items)}
        if not res.used_ids:  # nothing survived verification
            return {"answer": INSUFFICIENT, "citations": [], "verification_status": "no_evidence"}
        citations = build_citations(res.used_ids, docs_by_id)
        body = number_refs(res.body, citations)
        status = "verified"
        if res.status == "failed":
            body += "\n\n_Some statements were removed because they could not be verified against the source passages._"
            status = "partially_verified"
        return {"answer": body, "citations": citations, "verification_status": status}

    # ---- routing -----------------------------------------------------------------------
    def after_classify(state: GraphState) -> str:
        if state["query_type"] == "source_request" and _last_citations(state.get("history", [])):
            return "reuse_sources"
        return "retrieve_documents"

    def after_retrieve(state: GraphState) -> str:
        if state.get("verification_status") == "no_evidence":
            return END
        return "generate_answer" if state.get("documents") else "retrieve_documents"

    def after_generate(state: GraphState) -> str:
        return END if state.get("error") else "verify_citations"

    def after_verify(state: GraphState) -> str:
        return "generate_answer" if state.get("verification_status") == "retry" else END

    g = StateGraph(GraphState)
    for name, fn in [("classify_query", classify_query), ("reuse_sources", reuse_sources),
                     ("retrieve_documents", retrieve_documents), ("generate_answer", generate_answer),
                     ("verify_citations", verify_citations)]:
        g.add_node(name, fn)
    g.add_edge(START, "classify_query")
    g.add_conditional_edges("classify_query", after_classify, ["reuse_sources", "retrieve_documents"])
    g.add_edge("reuse_sources", END)
    g.add_conditional_edges("retrieve_documents", after_retrieve, ["generate_answer", "retrieve_documents", END])
    g.add_conditional_edges("generate_answer", after_generate, ["verify_citations", END])
    g.add_conditional_edges("verify_citations", after_verify, ["generate_answer", END])
    return g.compile()


def answer_question(graph: Any, company_id: str, company_name: str, question: str,
                    history: list[dict] | None = None, years: list[str] | None = None) -> dict:
    """Run the workflow for one question. History must belong to the same company."""
    try:
        payload: dict[str, Any] = {"company_id": company_id, "company_name": company_name,
                                   "question": question, "history": history or []}
        if years:
            payload["years"] = years
        out = graph.invoke(payload)
    except Exception as e:
        log.exception("graph failed")
        return {"content": f"Sorry, something went wrong while answering: {e}", "citations": [], "status": "error", "error": str(e)}
    if out.get("error"):
        return {"content": "The language model could not be reached or returned an error. Please try again in a moment.\n\n"
                           f"`{out['error'][:300]}`", "citations": [], "status": "error", "error": out["error"]}
    status = out.get("verification_status", "")
    content = out["answer"] if status in ("reused", "no_evidence") else format_response(out["answer"], out.get("citations", []))
    return {"content": content, "citations": out.get("citations", []), "status": status, "query_type": out.get("query_type")}
