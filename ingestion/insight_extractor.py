"""LLM-based structured extraction + deterministic grounding checks + JSON storage.

Nothing the LLM returns is trusted: every question / tip / description must be lexically
supported by the PDF page it points to, topics / role / difficulty must literally occur in the
document, and URLs must appear verbatim in the text. Unsupported items are dropped.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage

from ingestion.metadata_extractor import DocMeta
from ingestion.pdf_loader import PageText
from schemas.experience import (
    ExperienceRecord, LLMExperience, LLMExtraction, Question, Resource, RoundInfo, Tip,
    TopicMention, display_name,
)
from schemas.taxonomy import QUESTION_KINDS, ROUND_CATEGORIES, TIP_CATEGORIES, coerce, normalize_topic
from schemas.textmatch import support_ratio

log = logging.getLogger(__name__)

TEXT_SUPPORT = 0.5      # min content-word overlap for free-text items
MAX_WINDOW_CHARS = 45_000

EXTRACTION_SYSTEM = f"""You extract structured JSON data from campus-placement interview experience documents written by students.

STRICT RULES
- Extract ONLY what the text explicitly states. Never use outside knowledge about the company.
- If something is not stated, use null / an empty list. Do NOT infer the role, difficulty, rounds, or topics.
- One document may contain several separate candidate experiences. Return each as its own item in "experiences".
- Never output personal identifiers (names, emails, phone numbers, roll numbers).
- The text is divided by [[PAGE n]] markers. Set "page"/"pages" to the page where the information appears.
- Keep questions and tips close to the original wording (light cleanup only). Do not invent or generalise.
- rounds: one entry per round the candidate reports. category must be one of {ROUND_CATEGORIES}.
- topics: short phrases exactly as written (e.g. "DSA", "OOPs", "arrays").
- question kind must be one of {QUESTION_KINDS}.
- tips: advice the candidate explicitly gives. category must be one of {TIP_CATEGORIES}.
- resources: only resources explicitly mentioned (a name and/or a URL). Copy URLs exactly as written.
- difficulty: only if the candidate states it (e.g. "easy", "medium", "hard").
"""


# ----------------------------------------------------------------------------- grounding helpers
def is_valid_url(url: str | None) -> bool:
    if not url or re.search(r"\s", url):
        return False
    try:
        p = urlparse(url)
    except ValueError:
        return False
    host = p.netloc.split("@")[-1].split(":")[0]
    return p.scheme in ("http", "https") and "." in host and not host.startswith(".") and not host.endswith(".")


def _nospace(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def _resolve_page(text: str, claimed: list[int], page_text: dict[int, str], threshold: float = TEXT_SUPPORT):
    """Find the page that actually supports ``text``. Returns page number or None."""
    valid = [p for p in claimed if p in page_text]
    for candidates in (valid, list(page_text)):
        if not candidates:
            continue
        scored = [(support_ratio(text, page_text[p]), p) for p in candidates]
        score, page = max(scored)
        if score >= threshold:
            return page
    return None


def _term_in(term: str, full_text: str) -> bool:
    return support_ratio(term, full_text) >= 0.99


def _find_url_page(url: str, page_text: dict[int, str]) -> int | None:
    needle = _nospace(url).rstrip(".,);]")
    for p, t in page_text.items():
        if needle and needle in _nospace(t):
            return p
    return None


# ----------------------------------------------------------------------------- conversion
def _to_record(exp: LLMExperience, meta: DocMeta, sha: str, number: int, page_text: dict[int, str]):
    full = "\n".join(page_text.values())
    rounds: list[RoundInfo] = []
    for r in exp.rounds:
        pages: set[int] = set()
        desc = None
        if r.description:
            pg = _resolve_page(r.description, r.pages, page_text)
            if pg:
                desc, _ = r.description.strip(), pages.add(pg)
        topics, seen = [], set()
        for t in r.topics:
            t = t.strip()
            if t and _term_in(t, full) and t.lower() not in seen:
                seen.add(t.lower())
                canon, cat = normalize_topic(t)
                topics.append(TopicMention(original=t, canonical=canon, category=cat))
        questions = []
        for q in r.questions:
            pg = _resolve_page(q.text, [q.page] if q.page else r.pages, page_text)
            if pg:
                pages.add(pg)
                questions.append(Question(text=q.text.strip(), kind=coerce(q.kind, QUESTION_KINDS, "other"), page=pg))
        difficulty = r.difficulty.strip() if r.difficulty and _term_in(r.difficulty, full) else None
        if not (desc or topics or questions):
            continue
        rounds.append(
            RoundInfo(
                name=r.name.strip() or "Round",
                category=coerce(r.category, ROUND_CATEGORIES, "Other"),
                description=desc,
                topics=topics,
                questions=questions,
                difficulty=difficulty,
                pages=sorted(pages),
            )
        )

    tips, seen_tip = [], set()
    for t in exp.tips:
        pg = _resolve_page(t.text, [t.page] if t.page else [], page_text)
        key = t.text.strip().lower()
        if pg and key not in seen_tip:
            seen_tip.add(key)
            tips.append(Tip(text=t.text.strip(), category=coerce(t.category, TIP_CATEGORIES, "Interview strategy"), page=pg))

    resources = []
    for res in exp.resources:
        url = res.url.strip().rstrip(".,);]") if res.url else None
        pg = None
        if url:
            pg = _find_url_page(url, page_text) if is_valid_url(url) else None
            if pg is None:
                url = None
        if url is None:  # named resource without (valid) URL: the name must be in the text
            if not res.title or not _term_in(res.title, full):
                continue
            pg = _resolve_page(res.title, [res.page] if res.page else [], page_text, 0.99)
        title = res.title.strip() if res.title and _term_in(res.title, full) else None
        desc = res.description.strip() if res.description and support_ratio(res.description, full) >= TEXT_SUPPORT else None
        resources.append(Resource(title=title, url=url, description=desc, topic=res.topic, page=pg))

    role = exp.role.strip() if exp.role and _term_in(exp.role, full) else None
    if not (rounds or tips or resources):
        return None
    all_pages = {p for r in rounds for p in r.pages} | {t.page for t in tips if t.page} | {r.page for r in resources if r.page}
    return ExperienceRecord(
        experience_id=f"{meta.company_id}-{meta.year}-{sha[:8]}-{number}",
        company=meta.company, company_id=meta.company_id, year=meta.year, role=role,
        rounds=rounds, tips=tips, resources=resources,
        filename=meta.filename, source_path=meta.source_path,
        pages=sorted(all_pages), content_sha256=sha,
    )


def _windows(pages: list[PageText]) -> list[list[PageText]]:
    out, cur, size = [], [], 0
    for p in pages:
        if cur and size + len(p.text) > MAX_WINDOW_CHARS:
            out.append(cur)
            cur, size = [], 0
        cur.append(p)
        size += len(p.text)
    return out + ([cur] if cur else [])


def _invoke(structured, messages, retries: int = 3) -> LLMExtraction:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            res = structured.invoke(messages)
            return res if isinstance(res, LLMExtraction) else LLMExtraction.model_validate(res)
        except Exception as e:  # validation / API / rate-limit errors
            last = e
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "rate" in str(e).lower()
            wait = 15 if is_rate_limit else min(2**attempt, 8)
            log.warning("extraction attempt %d failed: %s (retrying in %ds)", attempt + 1, e, wait)
            if attempt < retries:
                time.sleep(wait)
    raise RuntimeError(f"structured extraction failed: {last}")


def extract_experiences(llm, pages: list[PageText], meta: DocMeta, sha256: str) -> list[ExperienceRecord]:
    """Extract grounded experience records from one PDF."""
    structured = llm.with_structured_output(LLMExtraction)
    records: list[ExperienceRecord] = []
    for window in _windows([p for p in pages if p.text.strip()]):
        page_text = {p.page: p.text for p in window}
        body = "\n\n".join(f"[[PAGE {p.page}]]\n{p.text}" for p in window)
        messages = [
            SystemMessage(content=EXTRACTION_SYSTEM),
            HumanMessage(content=f"Document: {meta.filename}\n\n{body}"),
        ]
        extraction = _invoke(structured, messages)
        for exp in extraction.experiences:
            rec = _to_record(exp, meta, sha256, len(records) + 1, page_text)
            if rec:
                records.append(rec)
    return records


# ----------------------------------------------------------------------------- JSON storage
def _exp_file(data_dir: Path, company_id: str, year: str) -> Path:
    return Path(data_dir) / "experiences" / f"{company_id}_{year}.json"


def save_records(data_dir: Path, meta: DocMeta, records: list[ExperienceRecord]) -> Path:
    """Replace this source file's records inside data/experiences/<company>_<year>.json."""
    path = _exp_file(data_dir, meta.company_id, meta.year)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8")).get("experiences", [])
    kept = [e for e in existing if e.get("source_path") != meta.source_path]
    kept += [r.model_dump() for r in records]
    payload = {"company_id": meta.company_id, "year": meta.year, "experiences": kept}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def rebuild_companies_index(data_dir: Path, manifest: dict) -> list[dict]:
    """Rebuild data/companies.json from the manifest (indexed docs) and experience JSON files."""
    companies: dict[str, dict] = {}
    names: dict[str, Counter] = {}
    for sp, d in manifest.get("documents", {}).items():
        if d.get("status") != "indexed":
            continue
        cid = d["company_id"]
        c = companies.setdefault(cid, {"company_id": cid, "years": set(), "documents": 0, "experiences": 0, "has_insights": False})
        c["years"].add(d["year"])
        c["documents"] += 1
        names.setdefault(cid, Counter())[d["company"]] += 1
    for f in sorted((Path(data_dir) / "experiences").glob("*.json")):
        payload = json.loads(f.read_text(encoding="utf-8"))
        cid = payload.get("company_id")
        if cid in companies and payload.get("experiences"):
            companies[cid]["experiences"] += len(payload["experiences"])
            companies[cid]["has_insights"] = True
    out = []
    for cid, c in sorted(companies.items()):
        c["name"] = display_name(names[cid].most_common(1)[0][0])
        c["years"] = sorted(c["years"])
        out.append(c)
    out.sort(key=lambda c: c["name"].lower())
    path = Path(data_dir) / "companies.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
