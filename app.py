"""CampusPrep AI - Company Placement Intelligence (Streamlit UI)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # Streamlit Cloud: make chromadb happy with an old system sqlite3
    import pysqlite3  # type: ignore

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import html
import re

import streamlit as st

from backend import dashboard as dash
from backend.citations import page_label
from backend.config import load_settings

st.set_page_config(page_title="CampusPrep AI", page_icon=":material/school:", layout="wide")


ICON_PATHS = {
    "school": '<path d="m3 9 9-5 9 5-9 5-9-5Z"/><path d="M7 11.2v4.1c3 2.3 7 2.3 10 0v-4.1"/><path d="M21 9v6"/>',
    "clipboard": '<rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 4.5h6M8 10h8M8 14h8M8 18h5"/>',
    "laptop": '<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M2 20h20M8 9l-2 2 2 2m4-4 2 2-2 2m5-4v4"/>',
    "code": '<path d="m8 8-4 4 4 4m8-8 4 4-4 4m-2-10-4 12"/>',
    "folder": '<path d="M3 6a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6Z"/><path d="M3 9h18"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="10" cy="7" r="4"/><path d="M20 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "lightbulb": '<path d="M9 18h6m-5 4h4m-5-7a7 7 0 1 1 6 0c-.8.6-1 1.1-1 3h-4c0-1.9-.2-2.4-1-3Z"/>',
    "chat": '<path d="M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8v.5Z"/>',
}


def svg_icon(name: str, size: int = 20) -> str:
    path = ICON_PATHS.get(name, ICON_PATHS["lightbulb"])
    return f'<svg aria-hidden="true" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{path}</svg>'

CSS = """
<style>
.stApp {background:#f5f7fb;color:#172033;}
.block-container {padding:2rem 2.4rem 4rem;max-width:1380px;}
.cp-hero {padding:1.4rem 1.7rem;border-radius:20px;margin-bottom:1.4rem;background:linear-gradient(120deg,#102a56,#1e4d8d 58%,#3277c8);color:white;box-shadow:0 12px 32px rgba(24,55,102,.16);}
.cp-eyebrow {font-size:.76rem;letter-spacing:.13em;text-transform:uppercase;opacity:.78;font-weight:700;}
.cp-title {display:flex;align-items:center;gap:.65rem;font-size:2rem;font-weight:760;color:white;margin:.25rem 0 0;letter-spacing:-.03em;}
.cp-title-icon {display:inline-flex;color:#dceaff;}
.cp-title-icon svg {width:1.8rem;height:1.8rem;}
.cp-sub {color:#e2edff;font-size:1rem;margin:.2rem 0 0;}
.cp-section {color:#182944;font-weight:750;letter-spacing:-.02em;}
.cp-card {background:#fff;border:1px solid #e3e9f2;border-radius:16px;padding:1rem 1.2rem;margin:.3rem 0 .9rem;box-shadow:0 4px 14px rgba(23,32,51,.035);}
.cp-card h4 {margin:0 0 .45rem;color:#172033;font-size:1rem;}
.cp-card p {margin:0;color:#40516b;line-height:1.65;}
.cp-note {background:#edf4ff;border:1px solid #d9e8ff;border-radius:12px;padding:.75rem 1rem;color:#34506f;font-size:.9rem;margin-bottom:1rem;}
div[data-testid="stMetric"] {background:white;border:1px solid #e3e9f2;border-radius:15px;padding:.8rem 1rem;box-shadow:0 3px 12px rgba(23,32,51,.03);}
div[data-testid="stMetricLabel"] p {color:#64748b;font-weight:600;}
div[data-testid="stMetricValue"] {color:#183b69;font-weight:750;}
button[data-baseweb="tab"] {font-weight:650;}
div[data-testid="stTabs"] [role="tablist"] {gap:.3rem;border-bottom:1px solid #e2e8f0;}
div[data-testid="stTabs"] button[role="tab"] {padding:.8rem .95rem;}
div[data-testid="stChatMessage"] {border:1px solid #e7edf5;border-radius:16px;padding:.35rem .85rem;margin:.65rem 0;background:#fff;}
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {line-height:1.7;}
div[data-testid="stChatInput"] textarea {border-radius:14px;}
div[data-testid="stDataFrame"] {border:1px solid #e3e9f2;border-radius:12px;overflow:hidden;}
.cp-tip-section {margin:1.4rem 0 1rem;padding:.2rem 0 .65rem;border-bottom:1px solid #e2e8f0;}
.cp-tip-heading {display:flex;align-items:center;gap:.65rem;color:#19375e;font-size:1.18rem;font-weight:750;}
.cp-tip-icon {display:inline-flex;align-items:center;justify-content:center;width:2rem;height:2rem;border-radius:10px;background:#eaf1fc;color:#28588f;}
.cp-tip-count {display:inline-block;background:#e8f0fc;color:#275895;border-radius:999px;padding:.18rem .62rem;font-size:.74rem;font-weight:700;}
.cp-tip-card {height:100%;background:#fff;border:1px solid #e2e9f2;border-radius:15px;padding:1rem 1.1rem;margin:.3rem 0 .8rem;box-shadow:0 4px 14px rgba(23,32,51,.035);}
.cp-tip-card:before {content:"“";display:block;color:#7ba4dc;font-family:Georgia,serif;font-size:2rem;line-height:.8;margin-bottom:.45rem;}
.cp-tip-text {color:#263b58;line-height:1.7;font-size:.97rem;}
.cp-tip-meta {border-top:1px solid #edf1f6;margin-top:.8rem;padding-top:.65rem;color:#718096;font-size:.78rem;}
.cp-tip-tag {display:inline-block;background:#edf6f1;color:#28734b;border-radius:999px;padding:.15rem .55rem;margin-right:.4rem;font-weight:650;}

.cp-round-stats {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.55rem;margin:.35rem 0 .85rem;}
.cp-round-stat {min-width:0;background:#f7f9fc;border:1px solid #e8edf4;border-radius:10px;padding:.5rem .7rem;}


.cp-resource-card {background:#fff;border:1px solid #e2e9f2;border-radius:14px;padding:.95rem 1rem;margin:.3rem 0 .55rem;box-shadow:0 3px 12px rgba(23,32,51,.025);}
.cp-resource-title {color:#19375e;font-size:1rem;font-weight:750;line-height:1.4;}
.cp-resource-description {color:#52647d;font-size:.88rem;line-height:1.55;margin-top:.35rem;}
.cp-resource-source {color:#718096;font-size:.76rem;margin-top:.65rem;padding-top:.55rem;border-top:1px solid #edf1f6;}

.cp-top-topic {display:grid;grid-template-columns:2rem minmax(0,1fr) auto;align-items:start;gap:.8rem;background:#fff;border:1px solid #e4eaf2;border-radius:13px;padding:.9rem 1rem;margin:.5rem 0;box-shadow:0 3px 12px rgba(23,32,51,.025);}
.cp-top-topic-rank {display:grid;place-items:center;width:1.8rem;height:1.8rem;border-radius:9px;background:#eaf1fc;color:#28588f;font-size:.82rem;font-weight:750;}
.cp-top-topic-name {color:#203a5d;font-size:1.02rem;font-weight:750;}
.cp-top-topic-count {color:#64748b;font-size:.82rem;font-weight:650;white-space:nowrap;}
.cp-top-topic-subtopics {color:#566982;font-size:.88rem;line-height:1.5;margin-top:.32rem;}
.cp-topic-group {background:#fff;border:1px solid #e2e9f2;border-radius:14px;padding:.9rem 1rem;margin:.45rem 0;box-shadow:0 3px 12px rgba(23,32,51,.025);}
.cp-topic-group-title {font-size:1rem;font-weight:750;color:#19375e;margin-bottom:.55rem;}
.cp-topic-item {display:flex;justify-content:space-between;gap:.6rem;align-items:center;padding:.38rem 0;border-bottom:1px solid #f0f3f7;color:#33445c;font-size:.9rem;}
.cp-topic-item:last-child {border-bottom:0;}
.cp-topic-count {white-space:nowrap;color:#63758e;font-size:.78rem;font-weight:650;}
.cp-round-stat-label {color:#718096;font-size:.74rem;font-weight:650;letter-spacing:.035em;text-transform:uppercase;line-height:1.3;}
.cp-round-stat-value {color:#263b58;font-size:1rem;font-weight:700;line-height:1.35;margin-top:.14rem;overflow-wrap:anywhere;}
@media (max-width:700px) {.cp-round-stats{grid-template-columns:repeat(2,minmax(0,1fr));}}
.cp-chat-intro {display:flex;align-items:flex-start;gap:.55rem;color:#415570;line-height:1.5;}
.cp-chat-intro svg {flex:none;color:#3268a8;margin-top:.1rem;}
.cp-round-summary {color:#263b58;line-height:1.45;}
.cp-round-summary ul {margin:.2rem 0 .35rem;padding-left:1.25rem;}
.cp-round-summary li {margin:.04rem 0;line-height:1.35;padding:0;}


@media (max-width:700px) {.block-container{padding:1rem .8rem 3rem}.cp-title{font-size:1.65rem}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

settings = load_settings()

EXAMPLES = [
    "What topics were asked in the online assessment?",
    "What coding questions were reported?",
    "Summarize the reported interview process.",
    "What preparation tips did selected students share?",
    "What resources were recommended?",
    "Create a seven-day study plan based on these experiences.",
]


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="cp-card"><h4>{html.escape(title)}</h4><p>{html.escape(body)}</p></div>', unsafe_allow_html=True)


def md_safe(text: str) -> str:
    return text.replace("`", "'")

def canonical_role(role: str | None) -> str:
    """Normalize common title variants into role families across every company."""
    raw = (role or "").strip()
    name = re.sub(r"[^a-z0-9]+", " ", raw.lower()).strip()
    if not name or name == "role not mentioned":
        return "Role not mentioned"

    if re.search(r"\b(?:aladdin|alladin)\b", name) and re.search(r"\bdata\b", name):
        return "Aladdin Data"
    if "global marketing" in name:
        return "Global Marketing"

    # Normalize equivalent software development titles, seniority labels, and specializations.
    if re.search(r"\b(?:sde|swe)\b", name) or (
        re.search(r"\bsoftware\b", name)
        and re.search(r"\b(?:engineer|engineering|developer|development)\b", name)
    ):
        return "Software Engineer"
    if re.search(r"\bapplication\b", name) and re.search(r"\b(?:engineer|developer)\b", name):
        return "Application Engineer"
    if re.search(r"\bdigital technology\b", name):
        return "Digital Technology Intern" if re.search(r"\b(?:intern|internship|dti)\b", name) else "Digital Technology"
    if re.search(r"\bassociate product manager\b", name):
        return "Associate Product Manager"
    if re.search(r"\bsummer intern(?:ship)?\b", name) or name in {"intern", "internship"}:
        return "Intern"

    # Drop common level/employment suffixes while retaining the functional title.
    name = re.sub(r"\b(?:internship|intern|graduate|trainee|entry level|junior|senior)\b", " ", name)
    name = re.sub(r"\b(?:level )?\d+[a-z]?\b", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name.title() if name else "Role not mentioned"


def description_points(text: str) -> list[str]:
    """Turn an extracted round narrative into readable sentence-level bullets."""
    cleaned = " ".join((text or "").split())
    return [point.strip() for point in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned) if point.strip()]


def brief_tip(text: str, word_limit: int = 24) -> str:
    """Show one concise, extractive tip preview without changing its meaning."""
    sentences = description_points(text)
    preview = sentences[0] if sentences else " ".join((text or "").split())
    words = preview.split()
    if len(words) > word_limit:
        preview = " ".join(words[:word_limit]).rstrip(" ,;:") + "…"
    return preview


def round_duration(text: str | None) -> str | None:
    """Pull a stated duration from the round account when one is present."""
    if not text:
        return None
    match = re.search(r"\b(\d+(?:\s*(?:[-–]|to)\s*\d+)?\s*(?:minutes?|mins?|hours?|hrs?))\b", text, re.I)
    if not match:
        return None
    duration = re.sub(r"\s*(?:[-–]|to)\s*", "–", match.group(1), flags=re.I)
    duration = re.sub(r"\bmins?\b", "minutes", duration, flags=re.I)
    return re.sub(r"\bhrs?\b", "hours", duration, flags=re.I)


def round_difficulty(round_info) -> str | None:
    """Use an explicitly stored or stated difficulty label, when available."""
    if round_info.difficulty:
        return round_info.difficulty
    for sentence in description_points(round_info.description or ""):
        match = re.search(r"\b(easy\s+to\s+(?:medium|hard)|medium\s+to\s+hard|very\s+easy|easy|medium|hard|difficult|challenging)\b", sentence, re.I)
        if match:
            value = match.group(1).lower()
            value = value[0].upper() + value[1:]
            if re.search(r"\bmcqs?\b", sentence, re.I):
                value += " (MCQs)"
            return value
    return None


def assessment_outcomes(round_info) -> list[str]:
    """Keep candidate-reported attempt or completion details visible for OAs."""
    if not re.search(r"online assessment|\bOA\b|assessment", f"{round_info.name} {round_info.category}", re.I):
        return []
    pattern = re.compile(r"\b(?:solved|completed|attempted|couldn't solve|could not solve|was able to solve|unable to solve)\b", re.I)
    return [sentence for sentence in description_points(round_info.description or "") if pattern.search(sentence)]


def assessment_sections(round_info) -> list[str]:
    """List assessment sections explicitly named in its source account or extracted topics."""
    text = round_info.description or ""
    sections = []
    if re.search(r"\bmcqs?\b|multiple.choice", text, re.I):
        sections.append("MCQs")
    if re.search(r"\bcoding\b", text, re.I):
        sections.append("Coding")
    if re.search(r"\b(?:aptitude|quantitative|logical reasoning|verbal reasoning)\b", text, re.I):
        sections.append("Aptitude / reasoning")
    return sections


def assessment_format(round_info) -> list[str]:
    """Extract source sentences that describe OA timing, sections, and rules."""
    if not re.search(r"online assessment|\bOA\b|assessment", f"{round_info.name} {round_info.category}", re.I):
        return []
    pattern = re.compile(r"\b(?:lasted|duration|minutes?|mins?|hours?|hrs?|there (?:were|was)|consisted of|sections?|submit|revisit|available as a language|time limit)\b", re.I)
    return [sentence for sentence in description_points(round_info.description or "") if pattern.search(sentence)]


@st.cache_resource(show_spinner="Loading embedding model and vector index (first run only)…")
def load_services():
    from backend.services import get_services

    return get_services()


companies = dash.load_companies(settings.data_dir)
manifest = dash.load_manifest(settings.data_dir)

# This deployment is intentionally scoped to the current 2026 records.
companies = [
    dict(c, years=[y for y in c.get("years", []) if y == "2026"],
         name=re.sub(r"\s+2026$", "", c["name"]))
    for c in companies if "2026" in c.get("years", [])
]
st.markdown('<div class="cp-hero"><div class="cp-eyebrow">Campus placement intelligence · 2026</div>'
            f'<div class="cp-title"><span class="cp-title-icon">{svg_icon("school", 32)}</span>CampusPrep AI</div>'
            '<div class="cp-sub">Explore candidate-reported interview rounds, topics, preparation tips and resources.</div></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### About")
    st.caption("Every insight and answer is derived from historical interview-experience PDFs and cites the file and page.")
    docs = manifest.get("documents", {})
    docs = {k: v for k, v in docs.items() if v.get("year") == "2026"}
    indexed = sum(1 for d in docs.values() if d.get("status") == "indexed")
    st.metric("Documents indexed", indexed)

if not companies:
    st.markdown(
        '<div class="cp-card"><h4>No interview experiences indexed yet</h4><p>Put the PDFs in '
        f"<code>{html.escape(settings.pdf_input_dir.name)}/&lt;year&gt;/&lt;company&gt;/</code> and run "
        "<code>python scripts/ingest.py</code>, then restart the app. See the README for details.</p></div>",
        unsafe_allow_html=True,
    )
    st.stop()

by_id = {c["company_id"]: c for c in companies}
st.markdown('<h3 class="cp-section">Explore a company</h3>', unsafe_allow_html=True)
company_id = st.selectbox("Company", options=list(by_id), format_func=lambda i: by_id[i]["name"],
                          key="company_id", help="Type to search", label_visibility="collapsed")
company = by_id[company_id]
company_name = company["name"]
c_docs = [d for d in docs.values() if d.get("company_id") == company_id and d.get("status") == "indexed"]
if c_docs:
    company = dict(company, documents=len(c_docs))
exps = [e for e in dash.load_company_experiences(settings.data_dir, company_id) if e.year == "2026"]

# ------------------------------------------------------------------------------------ overview
st.markdown(f"### {company_name} · 2026 interview experiences")
if not exps:
    st.info(f"{company['documents']} document(s) are indexed for {company_name}, but no structured insights are available "
            "(ingestion ran without an LLM or extraction failed). The chat still works; re-run `python scripts/ingest.py` "
            "to build the dashboard.")
else:
    ov = dash.overview(exps)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Distinct experiences", ov["experiences"])
    c2.metric("Source PDFs", ov["documents"])
    c3.metric("Record year", "2026")
    c4.metric("Roles identified", len(ov["roles"]))
    c5.metric("Topics extracted", ov["topics"])
    card("Reported placement procedure", dash.procedure_summary(exps))
    st.caption("Roles: " + (", ".join(ov["roles"]) if ov["roles"] else "no role was mentioned in the source documents"))

tab_rounds, tab_topics, tab_tips, tab_res, tab_chat = st.tabs(
    ["Rounds", "Topic Intelligence", "Preparation Tips", "Resources", "Ask CampusPrep AI"]
)

# ------------------------------------------------------------------------------------ rounds
with tab_rounds:
    if not exps:
        st.write("No structured rounds available.")
    else:
        st.markdown('<div class="cp-note">Quick view of the rounds and questions candidates reported. Difficulty is shown only when it was stated in the source.</div>', unsafe_allow_html=True)
        experiences = sorted(exps, key=lambda x: (x.year, x.filename), reverse=True)
        role_groups = {}
        for experience in experiences:
            role_groups.setdefault(canonical_role(experience.role), []).append(experience)
        for role_index, role_name in enumerate(sorted(role_groups)):
            role_experiences = role_groups[role_name]
            total_rounds = sum(len(experience.rounds) for experience in role_experiences)
            st.markdown(f"### {html.escape(role_name)}")
            st.caption(f"{len(role_experiences)} candidate experience{'s' if len(role_experiences) != 1 else ''} · {total_rounds} reported rounds")
            for experience_index, e in enumerate(role_experiences):
                label = f"{e.filename} · {len(e.rounds)} round{'s' if len(e.rounds) != 1 else ''}"
                with st.expander(label, expanded=(role_index == 0 and experience_index == 0)):
                    if not e.rounds:
                        st.caption("No interview rounds were extracted from this experience.")
                    for r in e.rounds:
                        with st.container(border=True):
                            st.markdown(f"#### {md_safe(r.name)}")
                            topics = list(dict.fromkeys(t.canonical for t in r.topics))
                            questions = r.questions
                            duration = round_duration(r.description)
                            difficulty = round_difficulty(r)
                            metric_rows = [("Duration", duration or "Not stated"),
                                           ("Difficulty", difficulty or "Not stated"),
                                           ("Focus areas", str(len(topics))),
                                           ("Questions reported", str(len(questions)))]
                            stat_cards = "".join(
                                f'<div class="cp-round-stat"><div class="cp-round-stat-label">{html.escape(label)}</div>'
                                f'<div class="cp-round-stat-value">{html.escape(value)}</div></div>'
                                for label, value in metric_rows
                            )
                            st.markdown(f'<div class="cp-round-stats">{stat_cards}</div>', unsafe_allow_html=True)
                            topic_summary = ", ".join(topics[:8]) if topics else "no specific topics were listed"
                            if len(topics) > 8:
                                topic_summary += f" and {len(topics) - 8} more"
                            duration_text = f"lasted {duration}" if duration else "has no stated duration"
                            st.markdown(f"**Round insight:** This {r.category.lower()} round {duration_text} and covered {md_safe(topic_summary)}. The record includes {len(questions)} reported question{'s' if len(questions) != 1 else ''}.")

                            format_facts = assessment_format(r)
                            sections = assessment_sections(r)
                            if sections:
                                st.markdown("**Assessment sections**  ")
                                st.markdown(" · ".join(f"**{section}**" for section in sections))
                            if format_facts:
                                st.markdown("**Assessment format and rules**")
                                for fact in format_facts:
                                    st.markdown(f"- {md_safe(fact)}")
                            outcomes = assessment_outcomes(r)
                            if outcomes:
                                st.markdown("**Candidate-reported attempt**")
                                for outcome in outcomes:
                                    st.markdown(f"- {md_safe(outcome)}")

                            topic_groups = {}
                            for topic in r.topics:
                                topic_groups.setdefault(topic.category or "Other", [])
                                if topic.canonical not in topic_groups[topic.category or "Other"]:
                                    topic_groups[topic.category or "Other"].append(topic.canonical)
                            if topic_groups:
                                st.markdown("**Topics covered by area**")
                                for area, names in topic_groups.items():
                                    st.markdown(f"- **{md_safe(area)}:** {', '.join(md_safe(name) for name in names)}")

                            kind_counts = {}
                            for question in questions:
                                kind = (question.kind or "other").replace("_", " ").title()
                                kind_counts[kind] = kind_counts.get(kind, 0) + 1
                            if kind_counts:
                                mix = " · ".join(f"{kind}: {count}" for kind, count in kind_counts.items())
                                st.caption(f"Question mix · {mix}")

                            if r.description:
                                with st.expander("Read the candidate's full account"):
                                    st.write(r.description)
                            if questions:
                                st.markdown("**Reported questions**")
                                for q in questions[:5]:
                                    st.markdown(f"- {md_safe(q.text)}")
                                    st.caption(dash.src(e.filename, q.page))
                                if len(questions) > 5:
                                    with st.expander(f"Show {len(questions) - 5} more reported questions"):
                                        for q in questions[5:]:
                                            st.markdown(f"- {md_safe(q.text)}")
                                            st.caption(dash.src(e.filename, q.page))
                            else:
                                st.caption("No specific questions were extracted for this round.")
                            pages = ", ".join(map(str, r.pages)) or "not listed"
                            st.caption(f"Source: {e.filename} · pages {pages}")

# ------------------------------------------------------------------------------------ topics
with tab_topics:
    if not exps:
        st.write("No structured topics available.")
    else:
        st.markdown("### Top 10 topics to prepare")
        st.caption("Ranked by the number of distinct 2026 interview experiences that mentioned each topic. This is historical frequency, not a prediction of what will be asked.")
        top_topics = dash.topic_stats(exps).head(10)
        if top_topics.empty:
            st.info("No topics were extracted for this company yet.")
        else:
            related_by_topic = {}
            for experience in exps:
                for round_info in experience.rounds:
                    round_topics = list(dict.fromkeys(topic.canonical for topic in round_info.topics))
                    for main_topic in round_topics:
                        related = related_by_topic.setdefault(main_topic, {})
                        for related_topic in round_topics:
                            if related_topic != main_topic:
                                related.setdefault(related_topic, set()).add(experience.experience_id)
            for rank, row in enumerate(top_topics.to_dict(orient="records"), start=1):
                topic_name = str(row["Topic"])
                topic = html.escape(topic_name)
                count = int(row["Experiences"])
                related = sorted(related_by_topic.get(topic_name, {}).items(), key=lambda item: (-len(item[1]), item[0]))[:5]
                if related:
                    subtopics = " · ".join(f"{html.escape(name)}" for name, _ in related)
                    subtopic_line = f'<div class="cp-top-topic-subtopics"><b>Related subtopics asked:</b> {subtopics}</div>'
                else:
                    subtopic_line = '<div class="cp-top-topic-subtopics">No related subtopics were extracted from these rounds.</div>'
                st.markdown(
                    f'<div class="cp-top-topic"><div class="cp-top-topic-rank">{rank}</div>'
                    f'<div><div class="cp-top-topic-name">{topic}</div>{subtopic_line}</div>'
                    f'<div class="cp-top-topic-count">{count} experiences</div></div>',
                    unsafe_allow_html=True,
                )

# ------------------------------------------------------------------------------------ tips
with tab_tips:
    grouped = dash.tips_by_category(exps)
    st.markdown('<div class="cp-note">These are preparation tips shared by candidates in the source PDFs. Browse them by interview stage; each card includes its source.</div>', unsafe_allow_html=True)
    if not grouped:
        st.info("No preparation tips were extracted for this company yet.")
    icons = {
        "OA preparation": "clipboard", "Technical preparation": "laptop",
        "Coding preparation": "code", "Project preparation": "folder",
        "HR preparation": "users", "Interview strategy": "target",
    }
    for category, tips in grouped.items():
        st.markdown(
            f'<div class="cp-tip-section"><div class="cp-tip-heading"><span class="cp-tip-icon">{svg_icon(icons.get(category, "lightbulb"))}</span> '
            f'{html.escape(category)} <span class="cp-tip-count">{len(tips)} tip{"s" if len(tips) != 1 else ""}</span></div></div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2, gap="medium")
        for index, tip in enumerate(tips):
            source = html.escape(f'{tip["year"]} · {tip["source"]}')
            preview = html.escape(brief_tip(tip["text"]))
            col = cols[index % 2]
            col.markdown(
                f'<div class="cp-tip-card"><div class="cp-tip-text">{preview}</div>'
                f'<div class="cp-tip-meta"><span class="cp-tip-tag">Candidate-reported</span>{source}</div></div>',
                unsafe_allow_html=True,
            )
            if brief_tip(tip["text"]) != " ".join(tip["text"].split()):
                with col.expander("Read full tip"):
                    st.write(tip["text"])

# ------------------------------------------------------------------------------------ resources
with tab_res:
    res = dash.resource_rows(exps)
    if res.empty:
        st.info("No resources were mentioned in the interview experiences.")
    else:
        st.markdown('<div class="cp-note">Resources explicitly mentioned in the interview experiences, grouped by topic. Links are shown only when verified in the source PDFs.</div>', unsafe_allow_html=True)
        grouped_resources = {}
        for row in res.to_dict(orient="records"):
            raw_topic = row.get("Topic")
            topic = raw_topic.strip() if isinstance(raw_topic, str) and raw_topic.strip() and raw_topic.strip() != "—" else "General resources"
            key = topic.casefold()
            if key not in grouped_resources:
                grouped_resources[key] = {"topic": topic, "items": []}
            grouped_resources[key]["items"].append(row)
        ordered_groups = sorted(grouped_resources.values(), key=lambda group: (group["topic"] == "General resources", group["topic"].casefold()))
        for group_index, group in enumerate(ordered_groups):
            topic = group["topic"]
            items = group["items"]
            with st.expander(f"{topic} · {len(items)} resource{'s' if len(items) != 1 else ''}", expanded=(group_index == 0), icon=":material/menu_book:"):
                cols = st.columns(2, gap="medium")
                for index, item in enumerate(items):
                    col = cols[index % 2]
                    title = item.get("Resource")
                    title = title if isinstance(title, str) and title.strip() else "Untitled resource"
                    description = item.get("Description")
                    description = description if isinstance(description, str) and description.strip() and description != "—" else ""
                    source = item.get("Source")
                    source = source if isinstance(source, str) and source.strip() else "Source not listed"
                    col.markdown(
                        f'<div class="cp-resource-card"><div class="cp-resource-title">{html.escape(title)}</div>'
                        + (f'<div class="cp-resource-description">{html.escape(description)}</div>' if description else "")
                        + f'<div class="cp-resource-source">Mentioned in · {html.escape(source)}</div></div>',
                        unsafe_allow_html=True,
                    )
                    url = item.get("URL")
                    if isinstance(url, str) and url.startswith(("http://", "https://")):
                        col.link_button("Open resource", url, key=f"resource_{company_id}_{group_index}_{index}")
                    else:
                        col.caption("No verified link in the source")

# ------------------------------------------------------------------------------------ chat
with tab_chat:
    histories = st.session_state.setdefault("histories", {})
    history = histories.setdefault(company_id, [])   # one history per company; never shared

    head, btn = st.columns([5, 1])
    head.markdown(f'<div class="cp-chat-intro">{svg_icon("chat", 20)}<span>Ask follow-up questions about <b>{html.escape(company_name)}</b>. Answers are grounded in its 2026 interview PDFs and include supporting citations.</span></div>', unsafe_allow_html=True)
    if btn.button("New chat", key="new_chat", width="stretch"):
        histories[company_id] = []
        st.rerun()

    def ask(q: str) -> None:
        st.session_state["pending_question"] = q

    if not history:
        st.caption("Try one of these questions")
        cols = st.columns(3)
        for i, q in enumerate(EXAMPLES):
            cols[i % 3].button(q, key=f"ex{i}", on_click=ask, args=(q,), width="stretch")

    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("citations"):
                with st.expander("View evidence passages"):
                    for c in m["citations"]:
                        st.markdown(f"**[{c['ref']}] {c['filename']} — {page_label(c)}** ({c['year']})")
                        st.caption(c["excerpt"])

    pending = st.session_state.pop("pending_question", None)
    prompt = st.chat_input(f"Ask about {company_name}…", key=f"chat_input_{company_id}") or pending
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching the interview experiences…"):
                try:
                    from backend.graph import answer_question

                    svc = load_services()
                    result = answer_question(svc.graph, company_id, company_name, prompt, history, years=["2026"])
                except Exception as e:  # missing key, model download failure, index problems, ...
                    result = {"content": f"The assistant is unavailable: {e}", "citations": [], "status": "error"}
            st.markdown(result["content"], unsafe_allow_html=False)
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": result["content"], "citations": result["citations"]})
        st.rerun()
