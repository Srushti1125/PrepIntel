"""Round-aware chunking with page tracking and light PII redaction."""
from __future__ import annotations

import re
from bisect import bisect_right

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.metadata_extractor import DocMeta
from ingestion.pdf_loader import PageText

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?<![\d.])(?:\+\d{1,3}[\s-]?)?(?:[6-9]\d{9}|\d{3}[\s-]\d{3}[\s-]\d{4})(?!\d)")

# Short lines that look like a round / experience heading start a new section, so a question
# stays together with the round it belongs to.
_HEADING_RE = re.compile(
    r"^\W*(?:"
    r"(?:round|stage|phase|level)\s*[-:#]?\s*(?:\d+|one|two|three|four|five|i{1,3}|iv|v)\b"
    r"|online\s+assessment|online\s+test|coding\s+(?:round|test)|aptitude\s+round"
    r"|(?:technical|managerial|hr|manager|group\s+discussion|gd|cultural\s+fit)\s+(?:round|interview)"
    r"|(?:interview\s+)?experience\s*[-:#]?\s*\d*\s*[:\-]?\s*$"
    r"|(?:candidate|student)\s*[-:#]?\s*\d+"
    r")",
    re.IGNORECASE,
)
MIN_SECTION_CHARS = 200


def redact_pii(text: str) -> str:
    return PHONE_RE.sub("[phone removed]", EMAIL_RE.sub("[email removed]", text))


def _is_heading(line: str) -> bool:
    s = line.strip()
    return 0 < len(s) <= 80 and bool(_HEADING_RE.match(s))


def _sections(full: str) -> list[tuple[int, int]]:
    starts, pos = [0], 0
    for line in full.split("\n"):
        if _is_heading(line) and pos > 0:
            starts.append(pos)
        pos += len(line) + 1
    sections, cur = [], starts[0]
    for b in starts[1:] + [len(full)]:
        if b == len(full) or b - cur >= MIN_SECTION_CHARS:
            sections.append((cur, b))
            cur = b
    return sections


def build_chunks(
    pages: list[PageText],
    meta: DocMeta,
    sha256: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split a document into chunks; every chunk carries company/year/file/page metadata."""
    full, spans = "", []
    for p in pages:
        if p.text.strip():
            start = len(full)
            full += p.text
            spans.append((start, len(full), p.page))
            full += "\n\n"
    if not spans:
        return []
    starts = [s[0] for s in spans]

    def page_at(pos: int) -> int:
        return spans[max(bisect_right(starts, pos) - 1, 0)][2]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "? ", " ", ""],
    )
    chunks: list[Document] = []
    for s, e in _sections(full):
        for d in splitter.create_documents([full[s:e]]):
            content = d.page_content.strip()
            if not content:
                continue
            a = s + max(d.metadata.get("start_index", 0), 0)
            b = a + len(d.page_content)
            idx = len(chunks)
            chunks.append(
                Document(
                    page_content=content,
                    metadata={
                        "company": meta.company,
                        "company_id": meta.company_id,
                        "year": meta.year,
                        "filename": meta.filename,
                        "page": page_at(a),
                        "page_end": page_at(max(b - 1, a)),
                        "source_path": meta.source_path,
                        "chunk_index": idx,
                        "content_sha256": sha256,
                    },
                )
            )
    return chunks


def chunk_ids(sha256: str, n: int) -> list[str]:
    """Deterministic ids: re-ingesting the same file overwrites rather than duplicates."""
    return [f"{sha256[:16]}-{i:04d}" for i in range(n)]
