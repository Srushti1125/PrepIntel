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

import streamlit as st

from backend import dashboard as dash
from backend.citations import page_label
from backend.config import load_settings

st.set_page_config(page_title="CampusPrep AI", page_icon="🎓", layout="wide")

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.cp-title {font-size: 2.1rem; font-weight: 750; color: #172033; margin: 0; letter-spacing: -0.02em;}
.cp-sub {color: #5B6B82; font-size: 1.05rem; margin: 0.1rem 0 1.2rem 0;}
.cp-card {background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
          box-shadow: 0 1px 2px rgba(23,32,51,0.04);}
.cp-card h4 {margin: 0 0 0.4rem 0; color: #172033; font-size: 1rem;}
.cp-card p {margin: 0; color: #334155; line-height: 1.55;}
.cp-badge {display:inline-block; background:#EAF1FF; color:#1D4ED8; border-radius:999px; padding:2px 10px; font-size:0.78rem; margin-right:6px;}
.cp-note {background:#F3F6FA; border-left: 3px solid #2563EB; border-radius: 8px; padding: 0.6rem 0.9rem; color:#334155; font-size:0.9rem;}
div[data-testid="stMetric"] {background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px; padding:0.8rem 1rem;}
button[data-baseweb="tab"] {font-weight: 600;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

settings = load_settings()

EXAMPLES = [
    "What topics were asked in the online assessment?",
    "What coding questions were reported?",
    "Explain the interview process across different years.",
    "What preparation tips did selected students share?",
    "What resources were recommended?",
    "Create a seven-day study plan based on these experiences.",
]


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="cp-card"><h4>{html.escape(title)}</h4><p>{html.escape(body)}</p></div>', unsafe_allow_html=True)


def md_safe(text: str) -> str:
    return text.replace("`", "'")


@st.cache_resource(show_spinner="Loading embedding model and vector index (first run only)…")
def load_services():
    from backend.services import get_services

    return get_services()


companies = dash.load_companies(settings.data_dir)
manifest = dash.load_manifest(settings.data_dir)

# Filter for 2026 records on frontend
has_2026 = any("2026" in c.get("years", []) for c in companies)
if has_2026:
    companies = [
        dict(c, years=[y for y in c.get("years", []) if y == "2026"])
        for c in companies if "2026" in c.get("years", [])
    ]
    st.markdown('<p class="cp-title">🎓 CampusPrep AI</p><p class="cp-sub">Company Placement Intelligence · <b>2026 Records</b></p>', unsafe_allow_html=True)
else:
    st.markdown('<p class="cp-title">🎓 CampusPrep AI</p><p class="cp-sub">Company Placement Intelligence</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### About")
    st.caption("Every insight and answer is derived from historical interview-experience PDFs and cites the file and page.")
    docs = manifest.get("documents", {})
    if has_2026:
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
company_id = st.selectbox("Select a company", options=list(by_id), format_func=lambda i: by_id[i]["name"],
                          key="company_id", help="Type to search")
company = by_id[company_id]
company_name = company["name"]
if has_2026:
    c_docs = [d for d in docs.values() if d.get("company_id") == company_id and d.get("status") == "indexed"]
    if c_docs:
        company = dict(company, documents=len(c_docs))
exps = dash.load_company_experiences(settings.data_dir, company_id)
if has_2026:
    exps = [e for e in exps if e.year == "2026"]

# ------------------------------------------------------------------------------------ overview
st.markdown(f"## {company_name}")
if not exps:
    st.info(f"{company['documents']} document(s) are indexed for {company_name}, but no structured insights are available "
            "(ingestion ran without an LLM or extraction failed). The chat still works; re-run `python scripts/ingest.py` "
            "to build the dashboard.")
else:
    ov = dash.overview(exps)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Distinct experiences", ov["experiences"])
    c2.metric("Source PDFs", ov["documents"])
    c3.metric("Years covered", ", ".join(ov["years"]) if len(ov["years"]) <= 3 else f"{ov['years'][0]}–{ov['years'][-1]}")
    c4.metric("Roles identified", len(ov["roles"]))
    c5.metric("Topics extracted", ov["topics"])
    card("Reported placement procedure", dash.procedure_summary(exps))
    st.caption("Roles: " + (", ".join(ov["roles"]) if ov["roles"] else "no role was mentioned in the source documents")
               + f" · Years: {', '.join(ov['years'])}")

tab_rounds, tab_topics, tab_tips, tab_res, tab_chat = st.tabs(
    ["Interview Rounds", "Topic Intelligence", "Preparation Tips", "Resources", "💬 Ask CampusPrep AI"]
)

# ------------------------------------------------------------------------------------ rounds
with tab_rounds:
    if not exps:
        st.write("No structured rounds available.")
    else:
        st.markdown('<div class="cp-note">Not every candidate went through the same process. Each experience is shown separately, with its source.</div>', unsafe_allow_html=True)
        years = sorted({e.year for e in exps})
        picked = st.multiselect("Filter by year", years, default=years, key="rounds_years")
        shown = [e for e in exps if e.year in picked]
        rows = dash.round_rows(shown)
        if not rows.empty:
            st.dataframe(rows, width="stretch", hide_index=True)
        for e in sorted(shown, key=lambda x: (x.year, x.filename), reverse=True):
            label = f"{e.year} · {e.role or 'role not mentioned'} · {e.filename} ({len(e.rounds)} round{'s' if len(e.rounds) != 1 else ''})"
            with st.expander(label):
                for r in e.rounds:
                    pages = ", ".join(map(str, r.pages)) or "—"
                    st.markdown(f"**{md_safe(r.name)}** · _{r.category}_" + (f" · difficulty stated: {md_safe(r.difficulty)}" if r.difficulty else ""))
                    if r.description:
                        st.write(r.description)
                    if r.topics:
                        st.markdown("Topics: " + " ".join(f"`{md_safe(t.canonical)}`" for t in r.topics))
                    for q in r.questions[:6]:
                        st.markdown(f"- {md_safe(q.text)}  \n  <sub>{html.escape(dash.src(e.filename, q.page))}</sub>", unsafe_allow_html=True)
                    st.caption(f"Source: {e.filename} · pages {pages} · {e.year}")
                    st.divider()

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
        st.markdown("#### Year-wise topic trends")
        by_year, per_year = dash.topic_by_year(exps)
        if not by_year.empty:
            st.dataframe(by_year, width="stretch")
            st.caption("Experiences per year (denominators): " + ", ".join(f"{y}: {n}" for y, n in per_year.items()))
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
    st.markdown('<div class="cp-note">Everything on this page is <b>candidate-reported</b> and copied from the source PDFs. '
                "AI-generated suggestions only appear in the chat, where they are labelled “Inference:”.</div>", unsafe_allow_html=True)
    if not grouped:
        st.write("No preparation tips were extracted for this company.")
    for category, tips in grouped.items():
        st.markdown(f"#### {category}")
        for t in tips:
            st.markdown(f"- {md_safe(t['text'])}  \n  <sub>Candidate-reported · {html.escape(t['year'])} · {html.escape(t['source'])}</sub>", unsafe_allow_html=True)

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
    head.markdown(f"Ask follow-up questions about **{company_name}**. Answers use only this company's interview PDFs.")
    if btn.button("New chat", key="new_chat", width="stretch"):
        histories[company_id] = []
        st.rerun()

    def ask(q: str) -> None:
        st.session_state["pending_question"] = q

    if not history:
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
                    years_filter = ["2026"] if has_2026 else None
                    result = answer_question(svc.graph, company_id, company_name, prompt, history, years=years_filter)
                except Exception as e:  # missing key, model download failure, index problems, ...
                    result = {"content": f"The assistant is unavailable: {e}", "citations": [], "status": "error"}
            st.markdown(result["content"])
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": result["content"], "citations": result["citations"]})
        st.rerun()
