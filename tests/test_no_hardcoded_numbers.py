"""
tests/test_no_hardcoded_numbers.py - Verifies that report.tex imports macros and tables.
Fails if results sections contain hardcoded result numbers instead of \\input or \\num* macros.
"""

import os
import re
import json
import pandas as pd
import pytest
import pypdf

def test_report_uses_macros_and_inputs():
    """Ensures report.tex uses \\input{results/macros.tex} and \\input{tables/...}."""
    tex_path = "report.tex"
    assert os.path.exists(tex_path), f"Missing {tex_path}"

    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Must include macros.tex
    assert "\\input{results/macros.tex}" in content or "\\input{results/macros}" in content, \
        "report.tex must \\input{results/macros.tex} to source all numbers from code"

    # 2. Must input generated tables
    required_tables = [
        "tab_anova_phase1",
        "tab_anova_ccd_rmse",
        "tab_anova_ccd_latency",
        "tab_lof",
        "tab_confirmation",
        "tab_benchmarks",
        "tab_depth_opt",
        "tab_hypervolume_comparison",
    ]
    for tab in required_tables:
        pattern = rf"\\input{{tables/{tab}(\.tex)?}}"
        assert re.search(pattern, content), f"report.tex must \\input tables/{tab}.tex"

    # 3. Must use \\num* macros in the body text
    macro_matches = re.findall(r"\\num[A-Z][a-zA-Z0-9]*", content)
    assert len(macro_matches) >= 25, f"Expected at least 25 macro usages in text, found {len(macro_matches)}"

    # 4. Must use \\numRev* macros for prospective revision evidence
    rev_macro_matches = re.findall(r"\\numRev[A-Z][a-zA-Z0-9]*", content)
    assert len(rev_macro_matches) >= 10, f"Expected at least 10 revision macro usages in text, found {len(rev_macro_matches)}"


def test_frozen_selections_counts_and_manifest():
    """Verifies 122 selection records, 95 distinct configs, 2440 evaluation rows, and manifest name."""
    manifest_path = os.path.join("results", "revision_v2", "full_run_001", "finalized_selections.json")
    assert os.path.exists(manifest_path), f"Missing {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    selections = manifest_data["configurations"]

    assert len(selections) == 122, f"Expected 122 selection records, found {len(selections)}"
    hashes = {s["config_sha256"] for s in selections}
    assert len(hashes) == 95, f"Expected 95 distinct configuration hashes, found {len(hashes)}"

    evals_path = os.path.join("results", "revision_v2", "full_run_001", "final_evaluations.csv")
    assert os.path.exists(evals_path), f"Missing {evals_path}"
    df_evals = pd.read_csv(evals_path)
    assert len(df_evals) == 2440, f"Expected 2440 evaluation rows, found {len(df_evals)}"


def test_manifest_naming_in_manuscript():
    """Verifies correct finalized_selections.json reference and absence of obsolete frozen_selection_manifest.json."""
    with open("report.tex", "r", encoding="utf-8") as f:
        tex = f.read()
    tex_unescaped = tex.replace(r"\_", "_")
    assert "finalized_selections.json" in tex_unescaped, "report.tex must reference finalized_selections.json"
    assert "frozen_selection_manifest.json" not in tex_unescaped, "report.tex must not reference obsolete frozen_selection_manifest.json"


def test_absence_of_unsupported_equivalence_language():
    """Verifies that no unsupported equivalence language exists in publication manuscripts."""
    banned_phrases = [
        "statistically indistinguishable",
        "equivalent performance",
        "multi-objective equivalence",
        "Multi-Objective Equivalence",
    ]
    for doc in ["report.tex", "REPORT.md"]:
        with open(doc, "r", encoding="utf-8") as f:
            text = f.read()
        for phrase in banned_phrases:
            assert phrase not in text, f"Found banned phrase '{phrase}' in {doc}"


def test_montgomery_edition_and_author_placeholders():
    """Verifies Montgomery 9th edition (2017) and explicit user author placeholders in manuscripts."""
    for doc in ["report.tex", "REPORT.md"]:
        with open(doc, "r", encoding="utf-8") as f:
            text = f.read()
        assert "[Author Names to be Confirmed Prior to Publication]" in text, f"Missing author placeholder in {doc}"
        assert "10th" not in text, f"Found reference to 10th edition in {doc}"
        assert "2017" in text, f"Missing 2017 year for Montgomery 9th edition in {doc}"


def test_compiled_pdf_page_count():
    """Verifies that compiled report.pdf has exactly 22 physical pages."""
    pdf_path = "report.pdf"
    assert os.path.exists(pdf_path), f"Missing {pdf_path}"
    reader = pypdf.PdfReader(pdf_path)
    assert len(reader.pages) == 22, f"Expected exactly 22 pages in {pdf_path}, found {len(reader.pages)}"
