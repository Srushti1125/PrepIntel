"""Smoke-test the Streamlit UI with the AppTest harness against freshly ingested sample data."""
import dataclasses
import pathlib

from streamlit.testing.v1 import AppTest

from ingestion.pipeline import run_ingestion
from tests.conftest import ACME_2024, GLOBEX_2025, HashingEmbeddings, ScriptedExtractionLLM, make_pdf

APP = pathlib.Path(__file__).resolve().parents[1] / "app.py"


def test_dashboard_renders_and_company_switch(settings, tmp_path, monkeypatch):
    # The app is scoped to the 2026 records, so this test builds its own 2026 sample data.
    root = tmp_path / "Interview_Experiences_2026"
    make_pdf(root / "2026" / "Acme_Corp" / "exp_1.pdf", ACME_2024)
    make_pdf(root / "2026" / "Globex" / "exp_1.pdf", GLOBEX_2025)
    s26 = dataclasses.replace(settings, pdf_input_dir=root, data_dir=tmp_path / "data2026", chroma_dir=tmp_path / "chroma2026")
    run_ingestion(s26, HashingEmbeddings(), ScriptedExtractionLLM(), progress=lambda m: None)

    monkeypatch.setenv("DATA_DIR", str(s26.data_dir))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert len(at.selectbox[0].options) == 2                      # Acme Corp, Globex
    assert {m.label: m.value for m in at.metric}["Distinct experiences"] == "1"
    at.selectbox[0].select("globex").run()
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["Distinct experiences"] == "1"


def test_empty_state_when_nothing_indexed(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "empty"))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("No interview experiences indexed yet" in m.value for m in at.markdown)