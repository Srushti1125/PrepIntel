"""Shared fixtures.

The real embedding model (HuggingFace hub) and Gemini are not available in CI/sandbox, so tests use:
  * HashingEmbeddings - deterministic bag-of-words vectors (lexical similarity, 256-dim, L2-normalised)
  * ScriptedExtractionLLM / FakeListChatModel - deterministic stand-ins for the LLM
Everything else (PyMuPDF, splitter, ChromaDB, LangGraph, citation verification) is the real code.
"""
from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path

import pymupdf
import pytest
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import Settings  # noqa: E402
from schemas.experience import LLMExperience, LLMExtraction, LLMQuestion, LLMResource, LLMRound, LLMTip  # noqa: E402

_STOP = {"the", "a", "an", "of", "in", "and", "to", "was", "were", "is", "are", "what", "who", "which", "for", "on", "did", "me", "it"}


class HashingEmbeddings(Embeddings):
    DIM = 256

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.DIM
        for tok in re.findall(r"[a-z0-9+#]+", text.lower()):
            if tok in _STOP:
                continue
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def make_pdf(path: Path, pages: list[str], image_only: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if image_only:  # no text layer, only a picture
            pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 80, 40), False)
            pix.clear_with(200)
            page.insert_image(pymupdf.Rect(50, 50, 250, 150), pixmap=pix)
        else:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=10)
    doc.save(path)
    doc.close()
    return path


ACME_2024 = [
    "Experience 1\nRound 1 Online Assessment\nThe online assessment had two coding questions on arrays and dynamic programming. "
    "I was asked to find the longest increasing subsequence. The test lasted ninety minutes.",
    "Round 2 Technical Interview\nThe technical interviewer asked about database normalization and SQL joins. "
    "He also asked me to explain my machine learning project in detail.\n"
    "Tip: Practice dynamic programming on LeetCode before the online assessment.\n"
    "Resource: https://leetcode.com/problemset/ for practice.",
]
ACME_2025 = [
    "Round 1 Online Assessment\nThe assessment contained aptitude questions and one coding problem about binary trees.",
    "Round 2 HR Interview\nThe HR round asked why I want to join and about my strengths and weaknesses.",
]
GLOBEX_2025 = [
    "Round 1 Technical Interview\nGlobex interviewers asked about operating systems deadlock and process scheduling. "
    "Computer networks questions covered TCP and the OSI model.",
]


@pytest.fixture()
def corpus(tmp_path):
    root = tmp_path / "Interview_Experiences"
    make_pdf(root / "2024" / "Acme_Corp" / "exp_1.pdf", ACME_2024)
    make_pdf(root / "2025" / "Acme_Corp" / "exp_2.pdf", ACME_2025)
    make_pdf(root / "2025" / "Globex" / "exp_1.pdf", GLOBEX_2025)
    return root


@pytest.fixture()
def settings(tmp_path, corpus):
    return Settings(
        llm_provider="gemini", gemini_api_key="", gemini_model="test", embedding_model="hashing-test",
        pdf_input_dir=corpus, chroma_dir=tmp_path / "chroma", data_dir=tmp_path / "data",
        collection_name="test", chunk_size=400, chunk_overlap=60, top_k=5, max_distance=0.9,
        max_retries=1, min_support=0.5, history_turns=3, ocr_enabled=False,
        chroma_cloud_api_key="", chroma_tenant="", chroma_database="",
    )


class ScriptedExtractionLLM:
    """Pretends to be Gemini structured output. Includes deliberately *invented* content that grounding must drop."""

    def with_structured_output(self, schema):
        acme24 = LLMExperience(
            role="Software Engineer Intern",  # NOT in the PDF -> must be dropped
            rounds=[
                LLMRound(name="Round 1 Online Assessment", category="Online Assessment", pages=[1],
                         description="The online assessment had two coding questions on arrays and dynamic programming.",
                         topics=["arrays", "dynamic programming", "Kubernetes"],  # Kubernetes is invented
                         questions=[LLMQuestion(text="find the longest increasing subsequence", kind="coding", page=1)]),
                LLMRound(name="Round 2 Technical Interview", category="Technical Interview", pages=[2],
                         topics=["database normalization", "SQL joins"],
                         questions=[LLMQuestion(text="explain my machine learning project in detail", kind="project", page=2),
                                    LLMQuestion(text="Implement an LRU cache with O(1) operations", kind="coding", page=2)]),  # invented
            ],
            tips=[LLMTip(text="Practice dynamic programming on LeetCode before the online assessment.", category="OA preparation", page=2),
                  LLMTip(text="Solve 500 problems on HackerRank every week.", category="Coding preparation", page=2)],  # invented
            resources=[LLMResource(title="LeetCode", url="https://leetcode.com/problemset/", page=2),
                       LLMResource(title="Fake", url="https://invented-resource.example.org/guide", page=2)],  # invented
        )
        acme25 = LLMExperience(rounds=[
            LLMRound(name="Round 1 Online Assessment", category="Online Assessment", pages=[1],
                     topics=["aptitude", "binary trees"], questions=[LLMQuestion(text="one coding problem about binary trees", kind="coding", page=1)]),
            LLMRound(name="Round 2 HR Interview", category="HR Interview", pages=[2], topics=["strengths and weaknesses"]),
        ])
        globex = LLMExperience(rounds=[
            LLMRound(name="Round 1 Technical Interview", category="Technical Interview", pages=[1],
                     topics=["operating systems", "deadlock", "OSI model"])])
        return _PerPath({"2024/Acme_Corp": acme24, "2025/Acme_Corp": acme25, "2025/Globex": globex})


class _PerPath:
    """exp_1.pdf exists under two companies, so route by the page text instead of the filename."""

    def __init__(self, by_marker):
        self.by_marker = by_marker

    def invoke(self, messages):
        text = messages[-1].content
        if "Globex interviewers" in text:
            return LLMExtraction(experiences=[self.by_marker["2025/Globex"]])
        if "longest increasing" in text or "ninety minutes" in text:
            return LLMExtraction(experiences=[self.by_marker["2024/Acme_Corp"]])
        return LLMExtraction(experiences=[self.by_marker["2025/Acme_Corp"]])


@pytest.fixture()
def embeddings():
    return HashingEmbeddings()


@pytest.fixture()
def ingested(settings, embeddings):
    from ingestion.pipeline import run_ingestion

    report = run_ingestion(settings, embeddings, ScriptedExtractionLLM(), progress=lambda m: None)
    return settings, embeddings, report
