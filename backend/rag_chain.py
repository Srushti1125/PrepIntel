"""LangChain pieces: source-aware context formatting and prompt | llm | parser chains."""
from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.prompts import CONDENSE_PROMPT, HUMAN_PROMPT, INSUFFICIENT, SYSTEM_PROMPT


def label_documents(docs: list[Document]) -> dict[str, Document]:
    return {f"S{i}": d for i, d in enumerate(docs, start=1)}


def format_context(docs_by_id: dict[str, Document]) -> str:
    blocks = []
    for sid, d in docs_by_id.items():
        m = d.metadata
        page = m.get("page") if m.get("page") == m.get("page_end") else f"{m.get('page')}-{m.get('page_end')}"
        blocks.append(
            f"[{sid}] Company: {m.get('company')} | Year: {m.get('year')} | File: {m.get('filename')} | Page: {page}\n"
            f"{d.page_content}"
        )
    return "\n\n".join(blocks)


def history_to_text(history: list[dict], turns: int = 3) -> str:
    """Last ``turns`` exchanges as plain text. Assistant source lists are removed."""
    lines = []
    for m in history[-2 * turns:]:
        content = re.split(r"\n\*\*Sources\*\*", m["content"])[0].replace("**Answer**", "").strip()
        who = "Student" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {content[:600]}")
    return "\n".join(lines) or "(none)"


def build_answer_chain(llm: Any):
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    return prompt | llm | StrOutputParser()


def build_condense_chain(llm: Any):
    return ChatPromptTemplate.from_template(CONDENSE_PROMPT) | llm | StrOutputParser()


def answer_inputs(company: str, question: str, docs_by_id, history_text: str, feedback: str = "") -> dict:
    return {
        "company": company, "insufficient": INSUFFICIENT, "context": format_context(docs_by_id),
        "history": history_text, "feedback": feedback, "question": question,
    }
