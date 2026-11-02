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

st.set_page_config(page_title="CampusPrep AI", page_icon="🎓", layout="wide")

CSS = """
<style>
.stApp {background:#f5f7fb;color:#172033;}
.block-container {padding:2rem 2.4rem 4rem;max-width:1380px;}
.cp-hero {padding:1.4rem 1.7rem;border-radius:20px;margin-bottom:1.4rem;background:linear-gradient(120deg,#102a56,#1e4d8d 58%,#3277c8);color:white;box-shadow:0 12px 32px rgba(24,55,102,.16);}
.cp-eyebrow {font-size:.76rem;letter-spacing:.13em;text-transform:uppercase;opacity:.78;font-weight:700;}
.cp-title {font-size:2rem;font-weight:760;color:white;margin:.25rem 0 0;letter-spacing:-.03em;}
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
.cp-tip-count {display:inline-block;background:#e8f0fc;color:#275895;border-radius:999px;padding:.18rem .62rem;font-size:.74rem;font-weight:700;}
.cp-tip-card {height:100%;background:#fff;border:1px solid #e2e9f2;border-radius:15px;padding:1rem 1.1rem;margin:.3rem 0 .8rem;box-shadow:0 4px 14px rgba(23,32,51,.035);}
.cp-tip-card:before {content:"“";display:block;color:#7ba4dc;font-family:Georgia,serif;font-size:2rem;line-height:.8;margin-bottom:.45rem;}
.cp-tip-text {color:#263b58;line-height:1.7;font-size:.97rem;}
.cp-tip-meta {border-top:1px solid #edf1f6;margin-top:.8rem;padding-top:.65rem;color:#718096;font-size:.78rem;}
.cp-tip-tag {display:inline-block;background:#edf6f1;color:#28734b;border-radius:999px;padding:.15rem .55rem;margin-right:.4rem;font-weight:650;}

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

def description_points(text: str) -> list[str]:
    """Turn an extracted round narrative into readable sentence-level bullets."""
    cleaned = " ".join((text or "").split())
    return [point.strip() for point in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned) if point.strip()]


@st.cache_resource(show_spinner="Loading embedding model and vector index (first run only)…")
def load_services():
    from backend.services import get_services

    return get_services()


companies = dash.load_companies(settings.data_dir)
manifest = dash.load_manifest(settings.data_dir)

# This deployment is intentionally scoped to the current 2026 records.
companies = [dict(c, years=[y for y in c.get("years", []) if y == "2026"])
             for c in companies if "2026" in c.get("years", [])]
st.markdown('<div class="cp-hero"><div class="cp-eyebrow">Campus placement intelligence · 2026</div>'
            '<div class="cp-title">🎓 CampusPrep AI</div>'
            '<div class="cp-sub">Explore candidate-reported interview rounds, topics, preparation tips and resources.</div></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### About")
    st.caption("Every insight and answer is derived from historical interview-experience PDFs and cites the file and page.")
    docs = manifest.get("documents", {})
    docs = {k: v for k, v in docs.items() if v.get("year") == "2026"}
    indexed = sum(1 for d in docs.values() if d.get("status") == "indexed")
    failed = {k: v for k, v in docs.items() if v.get("status") == "failed"}
    st.metric("Documents indexed", indexed)
    if failed:
        with st.expander(f"⚠️ {len(failed)} document(s) failed ingestion"):
            for k, v in failed.items():
                st.write(f"**{k}** — {v.get('error', 'unknown error')}")

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
    ["Interview Rounds", "Topic Intelligence", "Preparation Tips", "Resources", "💬 Ask CampusPrep AI"]
)

# ------------------------------------------------------------------------------------ rounds
with tab_rounds:
    if not exps:
        st.write("No structured rounds available.")
    else:
        st.markdown('<div class="cp-note">Quick view of the rounds and questions candidates reported. Difficulty is shown only when it was stated in the source.</div>', unsafe_allow_html=True)
        experiences = sorted(exps, key=lambda x: (x.year, x.filename), reverse=True)
        for idx, e in enumerate(experiences):
            role = e.role or "Role not mentioned"
            label = f"{role} · {e.filename} · {len(e.rounds)} round{'s' if len(e.rounds) != 1 else ''}"
            with st.expander(label, expanded=(idx == 0)):
                if not e.rounds:
                    st.caption("No interview rounds were extracted from this experience.")
                for r in e.rounds:
                    with st.container(border=True):
                        title_col, category_col, difficulty_col = st.columns([3, 2, 2])
                        title_col.markdown(f"#### {md_safe(r.name)}")
                        category_col.markdown(f"**Round type**  \n{md_safe(r.category)}")
                        difficulty_col.markdown(f"**Difficulty**  \n{md_safe(r.difficulty) if r.difficulty else 'Not stated'}")
                        if r.description:
                            st.markdown("**Round summary**")
                            for point in description_points(r.description):
                                st.markdown(f"- {md_safe(point)}")
                        if r.topics:
                            st.markdown("**Key topics:** " + " · ".join(f"{md_safe(t.canonical)}" for t in r.topics))
                        questions = r.questions
                        if questions:
                            st.markdown("**Questions reported**")
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
        st.markdown(f'<div class="cp-note"><b>Metric:</b> {dash.TOPIC_METRIC_LABEL}. {dash.TOPIC_DISCLAIMER}</div>', unsafe_allow_html=True)
        cats = ["All"] + dash.categories_present(exps)
        cat = st.selectbox("Topic category", cats, key="topic_cat")
        stats = dash.topic_stats(exps, cat)
        if stats.empty:
            st.write("No topics in this category.")
        else:
            st.markdown("#### Most frequently mentioned topics")
            st.bar_chart(stats.head(15).set_index("Topic")["Experiences"], horizontal=True, color="#2563EB")
            st.dataframe(stats, width="stretch", hide_index=True)
        st.markdown("#### Topics by round category")
        by_round = dash.topic_by_round(exps)
        if not by_round.empty:
            st.dataframe(by_round, width="stretch")
        st.markdown("#### Sample questions reported")
        kinds = ["All", "coding", "technical", "project", "hr", "aptitude", "other"]
        kind = st.selectbox("Question type", kinds, key="q_kind")
        qs = dash.sample_questions(exps, kind)
        if not qs:
            st.write("No questions of this type were extracted.")
        for q in qs[:25]:
            st.markdown(f"- {md_safe(q['question'])}  \n  <sub>{html.escape(q['category'])} · {html.escape(q['year'])} · {html.escape(q['source'])}</sub>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------ tips
with tab_tips:
    grouped = dash.tips_by_category(exps)
    st.markdown('<div class="cp-note">These are preparation tips shared by candidates in the source PDFs. Browse them by interview stage; each card includes its source.</div>', unsafe_allow_html=True)
    if not grouped:
        st.info("No preparation tips were extracted for this company yet.")
    icons = {
        "OA preparation": "🧩", "Technical preparation": "💻",
        "Coding preparation": "⌨️", "Project preparation": "🛠️",
        "HR preparation": "🤝", "Interview strategy": "🎯",
    }
    for category, tips in grouped.items():
        st.markdown(
            f'<div class="cp-tip-section"><div class="cp-tip-heading">{icons.get(category, "💡")} '
            f'{html.escape(category)} <span class="cp-tip-count">{len(tips)} tip{"s" if len(tips) != 1 else ""}</span></div></div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2, gap="medium")
        for index, tip in enumerate(tips):
            source = html.escape(f'{tip["year"]} · {tip["source"]}')
            body = html.escape(tip["text"])
            cols[index % 2].markdown(
                f'<div class="cp-tip-card"><div class="cp-tip-text">{body}</div>'
                f'<div class="cp-tip-meta"><span class="cp-tip-tag">Candidate-reported</span>{source}</div></div>',
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------------------------------ resources
with tab_res:
    res = dash.resource_rows(exps)
    if res.empty:
        st.write("No resources were mentioned in the interview experiences.")
    else:
        st.caption("Only resources explicitly mentioned in the PDFs. URLs are shown only if they are well-formed and appear verbatim in the source.")
        st.dataframe(res, width="stretch", hide_index=True,
                     column_config={"URL": st.column_config.LinkColumn("URL")})

# ------------------------------------------------------------------------------------ chat
with tab_chat:
    histories = st.session_state.setdefault("histories", {})
    history = histories.setdefault(company_id, [])   # one history per company; never shared

    head, btn = st.columns([5, 1])
    head.markdown(f"Ask follow-up questions about **{company_name}**. Answers are grounded in its 2026 interview PDFs and include supporting citations.")
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
