import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from backend.citations import build_citations, format_response, number_refs, verify_answer
from backend.graph import answer_question, build_graph, classify
from backend.prompts import INSUFFICIENT
from ingestion.vector_store import get_vector_store

PASSAGES = {
    "S1": "Round 1 Online Assessment. The online assessment had two coding questions on arrays and dynamic programming.",
    "S2": "Round 2 Technical Interview. The technical interviewer asked about database normalization and SQL joins.",
}


# ------------------------------------------------------------------ verify_answer (pure logic)
def test_supported_claims_pass():
    r = verify_answer("The online assessment had coding questions on arrays and dynamic programming [S1].", PASSAGES)
    assert r.status == "passed" and r.used_ids == ["S1"] and not r.unsupported


def test_missing_citation_is_flagged_and_removed():
    r = verify_answer("The online assessment had coding questions on arrays [S1].\nThe interview also covered quantum computing and blockchain.", PASSAGES)
    assert r.status == "failed" and any("quantum" in u for u in r.unsupported)
    assert "quantum" not in r.body and "arrays" in r.body


def test_citation_must_actually_support_the_claim():
    r = verify_answer("Candidates were asked about quantum computing and blockchain consensus [S1].", PASSAGES)
    assert r.status == "failed" and r.used_ids == []


def test_unknown_source_id_rejected():
    r = verify_answer("The technical interviewer asked about database normalization [S9].", PASSAGES)
    assert r.status == "failed"


def test_headings_do_not_need_citations_and_insufficient_passes_through():
    r = verify_answer("**Technical round:**\n- The technical interviewer asked about database normalization and SQL joins [S2].", PASSAGES)
    assert r.status == "passed"
    assert verify_answer(INSUFFICIENT, PASSAGES).status == "no_evidence"


def test_inference_sentences_still_need_citations():
    r = verify_answer("Inference: students should revise database normalization and SQL joins before interviews.", PASSAGES)
    assert r.status == "failed"


def test_citation_rendering_and_numbering():
    from langchain_core.documents import Document

    docs = {sid: Document(page_content=t, metadata={"company": "Acme", "year": "2024", "filename": "exp_1.pdf",
                                                     "source_path": "2024/Acme/exp_1.pdf", "page": i, "page_end": i})
            for i, (sid, t) in enumerate(PASSAGES.items(), start=1)}
    res = verify_answer("Coding questions on arrays and dynamic programming were asked [S1]. Database normalization and SQL joins came up in the technical interview [S2].", PASSAGES)
    cites = build_citations(res.used_ids, docs)
    out = format_response(number_refs(res.body, cites), cites)
    assert "[1]" in out and "[2]" in out and "[S1]" not in out
    assert "exp_1.pdf — Page 1" in out and "exp_1.pdf — Page 2" in out and out.startswith("**Answer**")


def test_query_classification():
    assert classify("Create a seven-day study plan") == "study_plan"
    assert classify("Show the source for this information") == "source_request"
    assert classify("What resources were recommended?") == "resources"
    assert classify("Explain the interview process across years") == "process"


# ------------------------------------------------------------------ full graph with scripted LLM
def _graph(ingested, responses):
    settings, emb, _ = ingested
    return build_graph(get_vector_store(settings, emb), FakeListChatModel(responses=responses), settings)


def test_grounded_answer_has_real_sources(ingested):
    g = _graph(ingested, ["The online assessment had two coding questions on arrays and dynamic programming [S1][S2][S3]."])
    out = answer_question(g, "acme_corp", "Acme Corp", "What was asked in the online assessment arrays dynamic programming?")
    assert out["status"] == "verified"
    assert "**Sources**" in out["content"] and "exp_1.pdf — Page 1" in out["content"]
    assert all(c["company"] == "Acme_Corp" for c in out["citations"])


def test_unsupported_question_admits_missing_information(ingested):
    g = _graph(ingested, ["should never be called"])
    out = answer_question(g, "acme_corp", "Acme Corp", "Who won the football world cup penalty shootout?")
    assert out["content"] == INSUFFICIENT and out["status"] == "no_evidence" and out["citations"] == []


def test_hallucinated_answer_is_retried_then_withheld(ingested):
    bad = "Acme asked candidates about quantum computing and blockchain consensus protocols [S1]."
    g = _graph(ingested, [bad, bad])
    out = answer_question(g, "acme_corp", "Acme Corp", "online assessment coding arrays")
    assert out["content"] == INSUFFICIENT and out["citations"] == []


def test_retry_recovers_with_supported_answer(ingested):
    bad = "Acme asked candidates about quantum computing and blockchain consensus protocols [S1]."
    good = "The online assessment had two coding questions on arrays and dynamic programming [S1][S2][S3]."
    out = answer_question(_graph(ingested, [bad, good]), "acme_corp", "Acme Corp", "online assessment coding arrays dynamic programming")
    assert out["status"] == "verified" and "quantum" not in out["content"]


def test_partial_answer_keeps_only_supported_sentences(ingested):
    mixed = ("The online assessment had two coding questions on arrays and dynamic programming [S1][S2][S3].\n"
             "Acme also required a blockchain consensus protocol presentation [S1].")
    out = answer_question(_graph(ingested, [mixed, mixed]), "acme_corp", "Acme Corp", "online assessment coding arrays dynamic programming")
    assert out["status"] == "partially_verified" and "blockchain" not in out["content"].split("_Some statements")[0]


def test_company_switching_keeps_evidence_and_history_separate(ingested):
    ans_a = "The online assessment had two coding questions on arrays and dynamic programming [S1]."
    ans_g = "Interviewers asked about operating systems deadlock and process scheduling [S1]."
    g = _graph(ingested, [ans_a, ans_g])
    a = answer_question(g, "acme_corp", "Acme Corp", "online assessment arrays dynamic programming", [])
    b = answer_question(g, "globex", "Globex", "operating systems deadlock process scheduling", [])  # fresh history for the new company
    assert {c["company"] for c in a["citations"]} == {"Acme_Corp"}
    assert {c["company"] for c in b["citations"]} == {"Globex"}
    assert "deadlock" not in a["content"] and "arrays" not in b["content"]


def test_source_request_reuses_previous_citations(ingested):
    g = _graph(ingested, ["unused"])
    prev = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "citations": [
        {"ref": 1, "source_ids": ["S1"], "company": "Acme_Corp", "year": "2024", "filename": "exp_1.pdf", "source_path": "2024/Acme_Corp/exp_1.pdf", "page": 2, "page_end": 2, "excerpt": "..."}]}]
    out = answer_question(g, "acme_corp", "Acme Corp", "Show the source for this information", prev)
    assert out["status"] == "reused" and "exp_1.pdf — Page 2" in out["content"]


def test_llm_failure_is_handled_gracefully(ingested):
    class Boom(FakeListChatModel):
        def _call(self, *a, **k):
            raise RuntimeError("quota exceeded")

    settings, emb, _ = ingested
    g = build_graph(get_vector_store(settings, emb), Boom(responses=["x"]), settings)
    out = answer_question(g, "acme_corp", "Acme Corp", "online assessment arrays")
    assert out["status"] == "error" and "try again" in out["content"].lower()
