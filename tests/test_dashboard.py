"""Results and observations shown in the dashboard are deterministic aggregates."""

from backend import dashboard


def test_results_and_observations_are_distinct_experience_counts(ingested):
    settings, _, _ = ingested
    exps = dashboard.load_company_experiences(settings.data_dir, "acme_corp")

    result = dashboard.overview(exps)
    assert result == {
        "experiences": 2,
        "years": ["2024", "2025"],
        "roles": [],
        "topics": 7,
        "documents": 2,
    }

    summary = dashboard.procedure_summary(exps)
    assert "Across 2 reported experiences (2024–2025)" in summary
    assert "Online Assessment (2 of 2)" in summary
    assert "Technical Interview (1 of 2)" in summary
    assert "HR Interview (1 of 2)" in summary

    topics = dashboard.topic_stats(exps).set_index("Topic")
    assert topics.loc["Arrays", "Experiences"] == 1
    assert topics.loc["Binary Trees", "Experiences"] == 1
    assert topics.loc["Dynamic Programming", "Share of experiences"] == "50%"
    assert set(topics.loc["Database Management Systems", "Years"].split(", ")) == {"2024"}


def test_topic_year_results_include_experience_denominators(ingested):
    settings, _, _ = ingested
    exps = dashboard.load_company_experiences(settings.data_dir, "acme_corp")
    by_year, denominators = dashboard.topic_by_year(exps)

    assert denominators == {"2024": 1, "2025": 1}
    assert by_year.loc["Dynamic Programming", "2024"] == 1
    assert by_year.loc["Binary Trees", "2025"] == 1

