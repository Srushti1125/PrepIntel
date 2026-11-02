"""Citation parsing, verification and rendering.

The LLM marks claims with [S#]. Here we (1) check each claim sentence has valid markers,
(2) check the cited passages lexically support the sentence, (3) drop unsupported sentences,
and (4) build the Sources list ourselves from the passages that were actually cited.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.documents import Document

from backend.prompts import INSUFFICIENT
from schemas.textmatch import support_ratio

MARKER_RE = re.compile(r"[\[【]\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*[\]】]", re.IGNORECASE)
_BULLET_RE = re.compile(r"^(\s*(?:[-*•]|\d+[.)])\s+)")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MIN_CLAIM_WORDS = 4


def marker_ids(text: str) -> list[str]:
    ids: list[str] = []
    for m in MARKER_RE.finditer(text):
        ids += [s.upper() for s in re.findall(r"S\d+", m.group(1), flags=re.I)]
    return ids


def strip_markers(text: str) -> str:
    return re.sub(r"\s+", " ", MARKER_RE.sub("", text)).strip()


def _plain(text: str) -> str:
    return re.sub(r"[*_`#>]+", "", strip_markers(text)).strip()


def _is_heading(line: str) -> bool:
    s = _plain(line)
    if not s or s.startswith("---"):
        return True
    if s.endswith(":") and not MARKER_RE.search(line):
        return True
    if re.fullmatch(r"(?i)(answer|sources?)", s):
        return True
    return bool(re.fullmatch(r"(?i)(day|week|step)\s*\d+.{0,40}", s)) and len(s.split()) <= 7 and not MARKER_RE.search(line)


@dataclass
class VerificationResult:
    status: str                       # passed | failed | no_evidence
    body: str                         # answer with unsupported sentences removed (markers still [S#])
    unsupported: list[str] = field(default_factory=list)
    used_ids: list[str] = field(default_factory=list)


def verify_answer(answer: str, passages: dict[str, str], min_support: float = 0.5) -> VerificationResult:
    """Check every claim sentence in ``answer`` against the passages it cites."""
    answer = (answer or "").replace("【", "[").replace("】", "]").strip()
    if not answer or answer.startswith(INSUFFICIENT[:40]):
        return VerificationResult("no_evidence", INSUFFICIENT)

    unsupported: list[str] = []
    kept_lines: list[str] = []
    n_claims = 0
    for raw in answer.splitlines():
        if _is_heading(raw):
            kept_lines.append(raw)
            continue
        m = _BULLET_RE.match(raw)
        prefix = m.group(1) if m else ""
        text = raw[len(prefix):]
        text = re.sub(r"([.!?])\s*((?:\[\s*S[\d\s,;S]*\]\s*)+)", r"\2\1 ", text)  # ". [S1]" -> " [S1]."
        kept: list[str] = []
        for sent in _SENT_SPLIT.split(text.strip()):
            plain = _plain(sent)
            if len(plain.split()) < _MIN_CLAIM_WORDS:
                continue
            n_claims += 1
            ids = marker_ids(sent)
            ok = bool(ids) and all(i in passages for i in ids)
            if ok:
                ok = support_ratio(plain, " ".join(passages[i] for i in ids)) >= min_support
            if ok:
                kept.append(sent.strip())
            else:
                unsupported.append(plain)
        if kept:
            kept_lines.append(prefix + " ".join(kept))

    body = "\n".join(kept_lines).strip()
    if n_claims == 0 or not marker_ids(body):
        return VerificationResult("failed" if n_claims else "no_evidence", INSUFFICIENT if not n_claims else body,
                                  unsupported or ["(answer contained no cited claims)"], [])
    used = list(dict.fromkeys(marker_ids(body)))
    return VerificationResult("failed" if unsupported else "passed", body, unsupported, used)


def build_citations(used_ids: list[str], docs_by_id: dict[str, Document]) -> list[dict]:
    """Merge cited passages that point at the same file/page range; number them 1..n."""
    citations: list[dict] = []
    for sid in used_ids:
        d = docs_by_id[sid]
        md = d.metadata
        key = (md.get("source_path"), md.get("page"), md.get("page_end"))
        existing = next((c for c in citations if (c["source_path"], c["page"], c["page_end"]) == key), None)
        if existing:
            existing["source_ids"].append(sid)
            continue
        citations.append({
            "ref": len(citations) + 1, "source_ids": [sid], "company": md.get("company"),
            "year": md.get("year"), "filename": md.get("filename"), "source_path": md.get("source_path"),
            "page": md.get("page"), "page_end": md.get("page_end", md.get("page")),
            "excerpt": d.page_content[:700],
        })
    return citations


def number_refs(body: str, citations: list[dict]) -> str:
    """Replace [S#] markers with [n] references that match the Sources list."""
    ref_of = {sid: c["ref"] for c in citations for sid in c["source_ids"]}

    def repl(m: re.Match) -> str:
        refs = sorted({ref_of[s.upper()] for s in re.findall(r"S\d+", m.group(1), flags=re.I) if s.upper() in ref_of})
        return "".join(f"[{r}]" for r in refs)

    return re.sub(r"[ \t]+([.!?,;])", r"\1", MARKER_RE.sub(repl, body))


def page_label(c: dict) -> str:
    p, pe = c.get("page"), c.get("page_end")
    return f"Page {p}" if not pe or pe == p else f"Pages {p}–{pe}"


def format_sources(citations: list[dict]) -> str:
    return "\n".join(
        f"- [{c['ref']}] {c['filename']} — {page_label(c)} ({c['company']}, {c['year']})" for c in citations
    )


def format_response(body: str, citations: list[dict]) -> str:
    if not citations:
        return body
    return f"**Answer**\n\n{body}\n\n**Sources**\n\n{format_sources(citations)}"
