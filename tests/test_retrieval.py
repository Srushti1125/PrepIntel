import pytest

from backend.retriever import CompanyRetriever
from ingestion.vector_store import get_vector_store

QUERIES = ["online assessment arrays dynamic programming", "operating systems deadlock", "HR round strengths weaknesses",
           "computer networks TCP OSI model", "database normalization SQL joins", "binary trees coding"]


def _retriever(ingested, company_id, **kw):
    settings, emb, _ = ingested
    return CompanyRetriever(vector_store=get_vector_store(settings, emb), company_id=company_id, top_k=10, max_distance=kw.pop("max_distance", 0.95), **kw)


@pytest.mark.parametrize("company_id,other", [("acme_corp", "globex"), ("globex", "acme_corp")])
def test_retrieval_never_crosses_companies(ingested, company_id, other):
    r = _retriever(ingested, company_id)
    for q in QUERIES:
        for d in r.invoke(q):
            assert d.metadata["company_id"] == company_id != other


def test_company_filter_returns_that_company_even_for_other_companys_topics(ingested):
    docs = _retriever(ingested, "acme_corp", max_distance=None).invoke("operating systems deadlock process scheduling")
    assert docs and all(d.metadata["company_id"] == "acme_corp" for d in docs)
    assert not any("deadlock" in d.page_content for d in docs)


def test_relevant_passage_found_with_source_metadata(ingested):
    docs = _retriever(ingested, "globex").invoke("operating systems deadlock")
    assert docs and "deadlock" in docs[0].page_content
    m = docs[0].metadata
    assert (m["filename"], m["year"], m["page"], m["company"]) == ("exp_1.pdf", "2025", 1, "Globex")


def test_empty_results_for_unrelated_query_and_unknown_company(ingested):
    assert _retriever(ingested, "acme_corp", max_distance=0.9).invoke("football world cup winner penalty") == []
    assert _retriever(ingested, "no_such_company").invoke("arrays") == []


def test_year_filter(ingested):
    docs = _retriever(ingested, "acme_corp", years=["2025"]).invoke("online assessment coding")
    assert docs and {d.metadata["year"] for d in docs} == {"2025"}


def test_retriever_refuses_unscoped_search(ingested):
    with pytest.raises(ValueError):
        _retriever(ingested, "").invoke("arrays")


def test_collection_dimension_matches_embedding(ingested):
    settings, emb, _ = ingested
    coll = get_vector_store(settings, emb)._collection
    got = coll.get(limit=1, include=["embeddings"])["embeddings"][0]
    assert len(got) == len(emb.embed_query("x")) == 256
