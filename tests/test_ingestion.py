import json

import pytest

from ingestion.insight_extractor import is_valid_url
from ingestion.metadata_extractor import MetadataError, extract_metadata
from ingestion.pdf_loader import discover_pdfs, load_pdf
from ingestion.pipeline import load_manifest, run_ingestion
from ingestion.text_splitter import build_chunks, redact_pii
from schemas.taxonomy import normalize_topic
from tests.conftest import ScriptedExtractionLLM, make_pdf


def test_discovers_pdfs_recursively(corpus):
    assert len(discover_pdfs(corpus)) == 3


def test_metadata_comes_from_folder_hierarchy(corpus):
    meta = extract_metadata(corpus / "2024" / "Acme_Corp" / "exp_1.pdf", corpus)
    assert (meta.year, meta.company, meta.company_id, meta.filename) == ("2024", "Acme_Corp", "acme_corp", "exp_1.pdf")
    assert meta.source_path == "2024/Acme_Corp/exp_1.pdf"


def test_metadata_rejects_bad_layout(corpus, tmp_path):
    stray = make_pdf(corpus / "notes.pdf", ["x"])
    with pytest.raises(MetadataError):
        extract_metadata(stray, corpus)
    with pytest.raises(MetadataError):
        extract_metadata(make_pdf(corpus / "misc" / "Acme" / "a.pdf", ["x"]), corpus)


def test_page_numbers_preserved(corpus):
    loaded = load_pdf(corpus / "2024" / "Acme_Corp" / "exp_1.pdf")
    assert [p.page for p in loaded.pages] == [1, 2]
    assert "dynamic programming" in loaded.pages[0].text and "normalization" in loaded.pages[1].text


def test_chunk_metadata_and_pages(corpus):
    path = corpus / "2024" / "Acme_Corp" / "exp_1.pdf"
    meta, loaded = extract_metadata(path, corpus), load_pdf(path)
    chunks = build_chunks(loaded.pages, meta, loaded.sha256, 400, 60)
    assert chunks
    for c in chunks:
        assert {"company", "company_id", "year", "filename", "page", "page_end", "source_path"} <= set(c.metadata)
        assert c.metadata["company_id"] == "acme_corp" and c.metadata["year"] == "2024"
    # a chunk about normalization must be attributed to page 2, the OA one to page 1
    by_text = {("normalization" in c.page_content): c.metadata["page"] for c in chunks}
    assert by_text[True] == 2
    assert any(c.metadata["page"] == 1 and "longest increasing" in c.page_content for c in chunks)


def test_round_heading_keeps_question_with_its_round(corpus):
    path = corpus / "2024" / "Acme_Corp" / "exp_1.pdf"
    chunks = build_chunks(load_pdf(path).pages, extract_metadata(path, corpus), "x" * 64, 1000, 100)
    assert any("Round 2 Technical Interview" in c.page_content and "normalization" in c.page_content for c in chunks)


def test_scanned_pdf_detected_and_reported(corpus, settings, embeddings):
    make_pdf(corpus / "2025" / "Initech" / "scan.pdf", ["", ""], image_only=True)
    loaded = load_pdf(corpus / "2025" / "Initech" / "scan.pdf")
    assert loaded.scanned_pages == [1, 2] and not loaded.has_text
    report = run_ingestion(settings, embeddings, None, progress=lambda m: None)
    assert "2025/Initech/scan.pdf" in report.failed and "scanned" in report.failed["2025/Initech/scan.pdf"]
    assert load_manifest(settings.data_dir)["documents"]["2025/Initech/scan.pdf"]["status"] == "failed"


def test_pii_redaction():
    out = redact_pii("mail me at jane.doe@example.com or call 9876543210 about arrays")
    assert "example.com" not in out and "9876543210" not in out and "arrays" in out


def test_duplicate_ingestion_is_skipped(ingested, corpus):
    settings, emb, first = ingested
    assert len(first.indexed) == 3 and first.chunks > 0
    from ingestion.vector_store import get_vector_store

    n_before = len(get_vector_store(settings, emb).get()["ids"])
    # same content under a new name -> duplicate; unchanged files -> skipped
    make_pdf(corpus / "2025" / "Globex" / "copy.pdf", ["Round 1 Technical Interview\nGlobex interviewers asked about operating systems deadlock and process scheduling. "
                                                        "Computer networks questions covered TCP and the OSI model."])
    second = run_ingestion(settings, emb, ScriptedExtractionLLM(), progress=lambda m: None)
    assert second.indexed == [] and len(second.skipped) == 3
    assert second.duplicates == ["2025/Globex/copy.pdf"]
    assert len(get_vector_store(settings, emb).get()["ids"]) == n_before


def test_changed_file_replaces_old_chunks(ingested, corpus):
    settings, emb, _ = ingested
    make_pdf(corpus / "2025" / "Globex" / "exp_1.pdf", ["Round 1 Technical Interview\nCompletely different text about compilers."])
    run_ingestion(settings, emb, None, skip_insights=True, progress=lambda m: None)
    from ingestion.vector_store import get_vector_store

    store = get_vector_store(settings, emb)
    texts = " ".join(store.get(where={"source_path": "2025/Globex/exp_1.pdf"})["documents"])
    assert "compilers" in texts and "deadlock" not in texts


def test_embedding_model_change_requires_reset(ingested):
    import dataclasses
    from ingestion.pipeline import IngestionError

    settings, emb, _ = ingested
    with pytest.raises(IngestionError):
        run_ingestion(dataclasses.replace(settings, embedding_model="other-model"), emb, None, progress=lambda m: None)


def test_structured_extraction_drops_invented_content(ingested):
    settings, _, report = ingested
    assert report.experiences == 3
    exp = json.loads((settings.data_dir / "experiences" / "acme_corp_2024.json").read_text())["experiences"][0]
    assert exp["company"] == "Acme_Corp" and exp["year"] == "2024"      # from folder names
    assert exp["role"] is None                                          # not in PDF -> not inferred
    topics = {t["canonical"] for r in exp["rounds"] for t in r["topics"]}
    assert {"Dynamic Programming", "Arrays", "Database Management Systems", "SQL"} <= topics   # normalised canonical names
    assert not any("kubernetes" in t.lower() for t in topics)           # invented topic dropped
    questions = [q["text"] for r in exp["rounds"] for q in r["questions"]]
    assert any("longest increasing" in q for q in questions) and not any("LRU" in q for q in questions)
    assert [t["text"] for t in exp["tips"]] == ["Practice dynamic programming on LeetCode before the online assessment."]
    assert [r["url"] for r in exp["resources"]] == ["https://leetcode.com/problemset/"]
    assert exp["rounds"][0]["pages"] == [1] and exp["tips"][0]["page"] == 2   # real page numbers
    originals = {t["original"]: t["canonical"] for r in exp["rounds"] for t in r["topics"]}
    assert originals["dynamic programming"] == "Dynamic Programming"    # original wording preserved
    assert originals["database normalization"] == "Database Management Systems"


def test_companies_index_built_dynamically(ingested):
    settings, _, _ = ingested
    companies = json.loads((settings.data_dir / "companies.json").read_text())
    assert {c["company_id"] for c in companies} == {"acme_corp", "globex"}
    acme = next(c for c in companies if c["company_id"] == "acme_corp")
    assert acme["years"] == ["2024", "2025"] and acme["name"] == "Acme Corp"


def test_topic_normalisation_and_url_validation():
    assert normalize_topic("DSA")[0] == "Data Structures and Algorithms"
    assert normalize_topic("OOPs")[0] == "Object-Oriented Programming"
    assert normalize_topic("DBMS")[0] == "Database Management Systems"
    assert normalize_topic("Quantum Gardening")[1] == "Other"
    assert is_valid_url("https://leetcode.com/x") and not is_valid_url("javascript:alert(1)") and not is_valid_url("http://nodot")
