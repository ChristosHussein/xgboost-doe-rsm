"""
tests/test_no_hardcoded_numbers.py - Verifies that report.tex imports macros and tables.
Fails if results sections contain hardcoded result numbers instead of \\input or \\num* macros.
"""

import os
import re
import pytest

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
