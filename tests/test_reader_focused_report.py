"""Editorial and structural checks for the reader-focused publication package."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_reader_report_has_requested_narrative_and_navigation():
    tex = (ROOT / "report.tex").read_text(encoding="utf-8")
    required_sections = [
        "Introduction: the problem we want to solve",
        "Our core idea: learning from experiments",
        "How we designed the experiment",
        "How we learned from the experimental results",
        "How we selected candidate models",
        "How we checked whether the findings held up",
        "Comparing DOE with alternative optimization methods",
        "What we learned",
        "Conclusion",
        "Code availability",
    ]
    assert "\\tableofcontents" in tex
    for title in required_sections:
        assert f"\\section{{{title}}}" in tex
    assert tex.index("\\begin{abstract}") < tex.index("\\tableofcontents") < tex.index(required_sections[0])
    assert tex.count("\\lowerbetter") >= 2
    assert "planned before execution and globally randomized" in tex
    assert "hierarchical analyses" in tex
    assert "failure to detect a difference is not evidence of equivalence" in tex


def test_main_report_excludes_development_history_and_internal_ledgers():
    tex = (ROOT / "report.tex").read_text(encoding="utf-8")
    lower = tex.lower()
    forbidden = [
        "work package",
        "codex",
        "antigravity",
        "pull request",
        "pr #",
        "config_sha256",
        "configuration hash",
        "github actions",
        "finalized_selections.json",
        "ai-assistance disclosure",
    ]
    for phrase in forbidden:
        assert phrase not in lower, f"Development-history phrase leaked into main report: {phrase}"
    expected_code_line = (
        "The code and research materials are publicly available at: "
        "\\url{https://github.com/ChristosHussein/xgboost-doe-rsm}"
    )
    assert expected_code_line in tex


def test_reader_figures_and_complete_coverage_inventory_are_present():
    tex = (ROOT / "report.tex").read_text(encoding="utf-8")
    expected_figures = [
        "explainer_research_pipeline.pdf",
        "explainer_design_projection.pdf",
        "explainer_blocking.pdf",
        "explainer_response_surface.pdf",
        "explainer_accuracy_latency.pdf",
        "explainer_method_comparison.pdf",
        "revision_v2_pareto_front.png",
    ]
    for figure in expected_figures:
        assert figure in tex
        assert (ROOT / "figures" / figure).is_file()

    coverage = (ROOT / "docs" / "report_rewrite_coverage.md").read_text(encoding="utf-8")
    for claim_no in range(1, 23):
        assert f"C-{claim_no:02d}" in coverage
    assert "15 pages" in coverage
    assert "27 pages" in coverage
    assert (ROOT / "technical_supplement.tex").is_file()
    assert (ROOT / "technical_supplement.pdf").is_file()

