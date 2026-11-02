"""Dashboard calculations. Everything is derived from the validated JSON records - no LLM, no placeholders."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from ingestion.insight_extractor import is_valid_url
from schemas.experience import ExperienceRecord
from schemas.taxonomy import TIP_CATEGORIES, TOPIC_CATEGORIES, display_topic_name

TOPIC_METRIC_LABEL = "Number of distinct experiences mentioning the topic"
TOPIC_DISCLAIMER = (
    "Counts are historical: how many distinct reported experiences mention each topic. "
    "They are not a probability that a topic will be asked in a future interview."
)


def load_companies(data_dir: Path) -> list[dict]:
    p = Path(data_dir) / "companies.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def load_manifest(data_dir: Path) -> dict:
    p = Path(data_dir) / "ingestion_manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"documents": {}}


def load_company_experiences(data_dir: Path, company_id: str) -> list[ExperienceRecord]:
    out: list[ExperienceRecord] = []
    for f in sorted((Path(data_dir) / "experiences").glob(f"{company_id}_*.json")):
        payload = json.loads(f.read_text(encoding="utf-8"))
        if payload.get("company_id") != company_id:  # '<id>_*' glob could match a longer id
            continue
        out += [ExperienceRecord.model_validate(e) for e in payload.get("experiences", [])]
    return sorted(out, key=lambda e: (e.year, e.filename, e.experience_id))


def src(filename: str, page: int | None) -> str:
    return f"{filename} · p.{page}" if page else filename


def overview(exps: list[ExperienceRecord]) -> dict:
    years = sorted({e.year for e in exps})
    roles = sorted({e.role for e in exps if e.role})
    topics = {display_topic_name(t.canonical) for e in exps for r in e.rounds for t in r.topics}
    return {
        "experiences": len(exps),
        "years": years,
        "roles": roles,
        "topics": len(topics),
        "documents": len({e.source_path for e in exps}),
    }


def procedure_summary(exps: list[ExperienceRecord]) -> str:
    """Deterministic summary built only from extracted counts."""
    n = len(exps)
    if not n:
        return ""
    years = sorted({e.year for e in exps})
    span = years[0] if len(years) == 1 else f"{years[0]}–{years[-1]}"
    cats = Counter()
    for e in exps:
        cats.update({r.category for r in e.rounds})
    parts = [f"{c} ({k} of {n})" for c, k in cats.most_common()]
    counts = [len(e.rounds) for e in exps if e.rounds]
    text = f"Across {n} reported experience{'s' if n != 1 else ''} ({span}), the round types mentioned were: " + ", ".join(parts) + "."
    if counts:
        lo, hi = min(counts), max(counts)
        text += f" Experiences described {lo if lo == hi else f'{lo} to {hi}'} round{'s' if hi != 1 else ''} each."
    return text + " Processes differ between candidates and years; see the Interview Rounds tab."


def round_rows(exps: list[ExperienceRecord]) -> pd.DataFrame:
    rows = []
    for e in exps:
        for r in e.rounds:
            rows.append({
                "Year": e.year, "Role": e.role or "—", "Round": r.name, "Category": r.category,
                "Topics": ", ".join(display_topic_name(t.canonical) for t in r.topics) or "—",
                "Questions": len(r.questions), "Difficulty (stated)": r.difficulty or "—",
                "Source": e.filename, "Pages": ", ".join(map(str, r.pages)) or "—",
            })
    return pd.DataFrame(rows)


def _topic_sets(exps):
    """experience_id -> {canonical: (category, set(original wordings))}"""
    result = {}
    for e in exps:
        d = defaultdict(lambda: ["Other", set()])
        for r in e.rounds:
            for t in r.topics:
                canon = display_topic_name(t.canonical)
                d[canon][0] = t.category
                d[canon][1].add(t.original)
        result[e.experience_id] = d
    return result


def topic_stats(exps: list[ExperienceRecord], category: str | None = None) -> pd.DataFrame:
    sets, n = _topic_sets(exps), max(len(exps), 1)
    agg: dict[str, dict] = {}
    year_of = {e.experience_id: e.year for e in exps}
    for eid, d in sets.items():
        for canon, (cat, originals) in d.items():
            a = agg.setdefault(canon, {"Topic": canon, "Category": cat, "Experiences": 0, "Years": set(), "Wording in sources": set()})
            a["Experiences"] += 1
            a["Years"].add(year_of[eid])
            a["Wording in sources"] |= originals
    rows = [{**a, "Share of experiences": f"{100 * a['Experiences'] / n:.0f}%",
             "Years": ", ".join(sorted(a["Years"])), "Wording in sources": ", ".join(sorted(a["Wording in sources"]))}
            for a in agg.values()]
    df = pd.DataFrame(rows, columns=["Topic", "Category", "Experiences", "Share of experiences", "Years", "Wording in sources"])
    if category and category != "All":
        df = df[df["Category"] == category]
    return df.sort_values(["Experiences", "Topic"], ascending=[False, True]).reset_index(drop=True)


def topic_by_round(exps: list[ExperienceRecord]) -> pd.DataFrame:
    seen: dict[tuple[str, str], set] = defaultdict(set)
    for e in exps:
        for r in e.rounds:
            for t in r.topics:
                seen[(display_topic_name(t.canonical), r.category)].add(e.experience_id)
    if not seen:
        return pd.DataFrame()
    df = pd.DataFrame([{"Topic": t, "Round category": c, "Experiences": len(v)} for (t, c), v in seen.items()])
    return df.pivot_table(index="Topic", columns="Round category", values="Experiences", fill_value=0, aggfunc="sum").astype(int)


def topic_by_year(exps: list[ExperienceRecord], top: int = 15) -> tuple[pd.DataFrame, dict[str, int]]:
    """Counts of distinct experiences per topic per year, plus total experiences per year (the denominators)."""
    per_year = Counter(e.year for e in exps)
    top_topics = list(topic_stats(exps)["Topic"].head(top))
    sets = _topic_sets(exps)
    rows = {t: Counter() for t in top_topics}
    for e in exps:
        for t in top_topics:
            if t in sets[e.experience_id]:
                rows[t][e.year] += 1
    years = sorted(per_year)
    df = pd.DataFrame([[rows[t][y] for y in years] for t in top_topics], index=top_topics, columns=years)
    return df, dict(sorted(per_year.items()))


def sample_questions(exps: list[ExperienceRecord], kind: str | None = None, limit: int | None = None) -> list[dict]:
    out = []
    for e in exps:
        for r in e.rounds:
            for q in r.questions:
                if kind in (None, "All") or q.kind == kind:
                    out.append({"question": q.text, "kind": q.kind, "round": r.name, "category": r.category,
                                "year": e.year, "source": src(e.filename, q.page)})
    return out[:limit] if limit else out


def tips_by_category(exps: list[ExperienceRecord]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {c: [] for c in TIP_CATEGORIES}
    for e in exps:
        for t in e.tips:
            grouped.setdefault(t.category, []).append({"text": t.text, "year": e.year, "source": src(e.filename, t.page)})
    return {k: v for k, v in grouped.items() if v}


def resource_rows(exps: list[ExperienceRecord]) -> pd.DataFrame:
    rows, seen = [], set()
    for e in exps:
        for r in e.resources:
            url = r.url if is_valid_url(r.url) else None  # re-validate at display time
            key = (url or "", (r.title or "").lower(), e.source_path)
            if key in seen or not (url or r.title):
                continue
            seen.add(key)
            rows.append({"Resource": r.title or url, "URL": url, "Description": r.description or "—",
                         "Topic": r.topic or "—", "Year": e.year, "Source": src(e.filename, r.page)})
    return pd.DataFrame(rows, columns=["Resource", "URL", "Description", "Topic", "Year", "Source"])


def categories_present(exps) -> list[str]:
    present = set(topic_stats(exps)["Category"]) if exps else set()
    return [c for c in TOPIC_CATEGORIES if c in present]
