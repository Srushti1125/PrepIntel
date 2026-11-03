"""Smoke-test the Streamlit UI with the AppTest harness against freshly ingested sample data."""
import os

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_and_company_switch(ingested, monkeypatch):
    settings, _, _ = ingested
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert set(at.selectbox[0].options) == {"acme_corp", "globex"}
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Distinct experiences"] == "2"
    at.selectbox[0].select("globex").run()
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["Distinct experiences"] == "1"


def test_empty_state_when_nothing_indexed(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "empty"))
    at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("No interview experiences indexed yet" in m.value for m in at.markdown)
