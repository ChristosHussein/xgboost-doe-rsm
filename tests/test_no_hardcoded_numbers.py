"""
tests/test_no_hardcoded_numbers.py - Verifies that report.tex imports macros and tables.
Fails if results sections contain hardcoded result numbers instead of \\input or \\num* macros.
"""

import os
import re
import json
import pandas as pd
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


def test_montgomery_edition_and_author_name():
    """Verifies Montgomery 9th edition (2017) and confirmed author name in manuscripts."""
    for doc in ["report.tex", "REPORT.md"]:
        with open(doc, "r", encoding="utf-8") as f:
            text = f.read()
        assert "Christos Chousein Sounios" in text, f"Missing author 'Christos Chousein Sounios' in {doc}"
        assert "[Author Names" not in text, f"Unresolved author placeholder in {doc}"
        assert "[Institutional Affiliations" not in text, f"Unresolved affiliation placeholder in {doc}"
        assert "10th" not in text, f"Found reference to 10th edition in {doc}"
        assert "2017" in text, f"Missing 2017 year for Montgomery 9th edition in {doc}"


def test_report_md_synchronized_with_revised_evidence():
    """Verifies REPORT.md contains updated validation-RMSE metrics and no outdated pre-revision values."""
    with open("REPORT.md", "r", encoding="utf-8") as f:
        md = f.read()

    # Must contain current verified values
    required_current = [
        "Development Validation RMSE",
        "10 confirmation trials",
        "105,432.31",
        "0.6782",
        "40.69%",
    ]
    for token in required_current:
        assert token in md, f"REPORT.md is missing current verified value/label: '{token}'"

    # Must not contain outdated pre-revision numbers
    outdated_tokens = [
        "57,468.86",
        "57468.86",
        "57,469",
        "0.8851",
        "21.60%",
        "0.2160",
        "5 confirmation trials",
        "m = 5",
    ]
    for bad in outdated_tokens:
        assert bad not in md, f"REPORT.md contains outdated pre-revision value: '{bad}'"


def test_holdout_history_and_hypervolume_table_labels():
    """Verifies accurate holdout test-set history phrasing and distinct hypervolume evaluation labels."""
    for doc in ["report.tex", "REPORT.md"]:
        with open(doc, "r", encoding="utf-8") as f:
            text = f.read().lower()
        assert "untouched" not in text, f"Found inaccurate 'untouched' holdout claim in {doc}"
        assert "prevented access" in text or "blocked access" in text, (
            f"Missing explanation of revised holdout access prevention in {doc}"
        )

    hv_tex_path = os.path.join("tables", "tab_hypervolume_comparison.tex")
    assert os.path.exists(hv_tex_path), f"Missing {hv_tex_path}"
    with open(hv_tex_path, "r", encoding="utf-8") as f:
        hv_tex = f.read()

    assert "Candidate Frontiers on Development Split (Split 42, 30-Call Protocol)" not in hv_tex, (
        "Hypervolume table still uses blanket Split 42 / 30-Call heading for all Panel A rows"
    )
    assert "Distinct Evaluation Protocols Noted per Row" in hv_tex
    assert "Single Split 42, 30-Call Search" in hv_tex
    assert "5-Block Means Across Replicates" in hv_tex
    assert "Single Split 42, 27 Configs" in hv_tex


def test_compiled_pdf_page_count():
    """Verifies that compiled report.pdf has exactly 22 physical pages."""
    try:
        import pypdf
    except ImportError:
        pytest.skip("pypdf is not installed; skipping PDF page count check")

    pdf_path = "report.pdf"
    assert os.path.exists(pdf_path), f"Missing {pdf_path}"
    reader = pypdf.PdfReader(pdf_path)
    assert len(reader.pages) == 22, f"Expected exactly 22 pages in {pdf_path}, found {len(reader.pages)}"


def _extract_markdown_table_rows(md_text: str, section_heading: str) -> list[list[str]]:
    """Extracts parsed cells of the first Markdown table following `section_heading`."""
    idx = md_text.find(section_heading)
    assert idx != -1, f"Missing section heading '{section_heading}' in REPORT.md"
    lines = md_text[idx:].splitlines()
    table_lines = []
    in_table = False
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            table_lines.append(stripped)
        elif in_table:
            break
    assert len(table_lines) >= 3, f"No valid Markdown table found under '{section_heading}'"
    rows = []
    for line in table_lines[2:]:  # skip header and separator
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return rows


def _floats_in(text: str) -> list[float]:
    """Extracts all floating-point/integer numbers from a cell after removing LaTeX subscripts/superscripts."""
    cleaned = re.sub(r"[_^]\{[^}]*\}", "", text)
    cleaned = re.sub(r"[_^][0-9a-zA-Z]+", "", cleaned)
    cleaned = cleaned.replace(",", "")
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", cleaned)]


def test_report_md_numerical_tables_match_canonical_artifacts():
    """
    Numerically compares REPORT.md Confirmation, Benchmark, and Hypervolume tables
    (as well as confirmation seeds) directly against canonical experiment artifacts.
    """
    import yaml

    with open("REPORT.md", "r", encoding="utf-8") as f:
        md = f.read()
    with open("report.tex", "r", encoding="utf-8") as f:
        tex = f.read()
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Verify confirmation seeds against config.yaml and results/confirmation_runs.csv
    expected_conf_seeds = cfg["seeds"]["confirmation_seeds"]
    df_conf_runs = pd.read_csv(os.path.join("results", "confirmation_runs.csv"))
    csv_conf_seeds = sorted(df_conf_runs["seed"].unique().tolist())
    assert expected_conf_seeds == [505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414]
    assert csv_conf_seeds == expected_conf_seeds

    seed_match_md = re.search(r"\\mathcal\{S\}_\{\\text\{conf\}\}\s*=\s*\\\{([^}]+)\\\}", md)
    assert seed_match_md, "Missing explicit S_conf seed set in REPORT.md"
    parsed_md_seeds = [int(s.strip()) for s in seed_match_md.group(1).split(",")]
    assert parsed_md_seeds == expected_conf_seeds

    seed_match_tex = re.search(r"\\mathcal\{S\}_\{\\text\{conf\}\}\s*=\s*\\\{([^}]+)\\\}", tex)
    assert seed_match_tex, "Missing explicit S_conf seed set in report.tex"
    parsed_tex_seeds = [int(s.strip()) for s in seed_match_tex.group(1).split(",")]
    assert parsed_tex_seeds == expected_conf_seeds

    # 2. Numerically verify Section 6 Confirmation table against results/confirmation.json
    with open(os.path.join("results", "confirmation.json"), "r", encoding="utf-8") as f:
        conf_data = json.load(f)

    conf_rows = _extract_markdown_table_rows(md, "## 6. Phase 5: Empirical Confirmation Trials")
    # Expected row order:
    # 0: Header row for MO
    # 1: MO Validation RMSE
    # 2: MO Holdout Test RMSE
    # 3: MO Inference Latency
    # 4: Header row for SO
    # 5: SO Validation RMSE
    # 6: SO Holdout Test RMSE
    # 7: SO Inference Latency
    mo_val_nums = _floats_in(" ".join(conf_rows[1][1:4]))
    assert mo_val_nums[0] == pytest.approx(conf_data["Y1_Val_RMSE"]["predicted_mean"], abs=1e-4)
    assert mo_val_nums[1] == pytest.approx(conf_data["Y1_Val_RMSE"]["prediction_interval_95"][0], abs=1e-4)
    assert mo_val_nums[2] == pytest.approx(conf_data["Y1_Val_RMSE"]["prediction_interval_95"][1], abs=1e-4)
    assert mo_val_nums[3] == pytest.approx(conf_data["Y1_Val_RMSE"]["empirical_mean"], abs=1e-4)
    assert mo_val_nums[4] == pytest.approx(conf_data["Y1_Val_RMSE"]["empirical_std"], abs=1e-4)

    mo_test_nums = _floats_in(conf_rows[2][3])
    assert mo_test_nums[0] == pytest.approx(conf_data["Y1_Test_RMSE"]["empirical_mean"], abs=1e-4)
    assert mo_test_nums[1] == pytest.approx(conf_data["Y1_Test_RMSE"]["empirical_std"], abs=1e-4)

    mo_lat_pred = _floats_in(conf_rows[3][1])
    mo_lat_pi = _floats_in(conf_rows[3][2])
    mo_lat_emp = _floats_in(conf_rows[3][3])
    assert mo_lat_pred[0] == pytest.approx(conf_data["Y2_Latency"]["predicted_mean"], abs=0.1)
    assert mo_lat_pi[0] == pytest.approx(conf_data["Y2_Latency"]["prediction_interval_95"][0], abs=0.1)
    assert mo_lat_pi[1] == pytest.approx(conf_data["Y2_Latency"]["prediction_interval_95"][1], abs=0.1)
    assert mo_lat_emp[0] == pytest.approx(conf_data["Y2_Latency"]["empirical_mean"], abs=0.1)
    assert mo_lat_emp[1] == pytest.approx(conf_data["Y2_Latency"]["empirical_std"], abs=0.1)

    so_data = conf_data["Single_Objective_Optimum"]
    so_val_nums = _floats_in(" ".join(conf_rows[5][1:4]))
    assert so_val_nums[0] == pytest.approx(so_data["predicted_val_rmse"], abs=1e-4)
    assert so_val_nums[1] == pytest.approx(so_data["prediction_interval_95_val"][0], abs=1e-4)
    assert so_val_nums[2] == pytest.approx(so_data["prediction_interval_95_val"][1], abs=1e-4)
    assert so_val_nums[3] == pytest.approx(so_data["empirical_val_rmse"], abs=1e-4)
    assert so_val_nums[4] == pytest.approx(so_data["empirical_val_std"], abs=1e-4)

    so_test_nums = _floats_in(conf_rows[6][3])
    assert so_test_nums[0] == pytest.approx(so_data["empirical_test_rmse"], abs=1e-4)
    assert so_test_nums[1] == pytest.approx(so_data["empirical_test_std"], abs=1e-4)

    so_lat_pred = _floats_in(conf_rows[7][1])
    so_lat_pi = _floats_in(conf_rows[7][2])
    so_lat_emp = _floats_in(conf_rows[7][3])
    assert so_lat_pred[0] == pytest.approx(so_data["predicted_latency"], abs=0.1)
    assert so_lat_pi[0] == pytest.approx(so_data["prediction_interval_95_lat"][0], abs=0.1)
    assert so_lat_pi[1] == pytest.approx(so_data["prediction_interval_95_lat"][1], abs=0.1)
    assert so_lat_emp[0] == pytest.approx(so_data["empirical_latency"], abs=0.1)
    assert so_lat_emp[1] == pytest.approx(so_data["empirical_latency_std"], abs=0.1)

    # 3. Numerically verify Section 7.1 Benchmark Table against optimizer_summary.json,
    #    final_summary.csv, final_evaluations.csv, and results/benchmark.csv
    import ast

    rev_dir = os.path.join("results", "revision_v2", "full_run_001")
    with open(os.path.join(rev_dir, "optimizer_summary.json"), "r", encoding="utf-8") as f:
        opt_summary = dict(json.load(f)["optimizers"])
    with open(os.path.join(rev_dir, "doe_selection_summary.json"), "r", encoding="utf-8") as f:
        opt_summary.update(json.load(f)["methods"])
    df_final_sum = pd.read_csv(os.path.join(rev_dir, "final_summary.csv"))
    for col in ["validation_rmse", "test_rmse"]:
        df_final_sum[col + "_mean"] = df_final_sum[col].apply(lambda x: ast.literal_eval(x)["mean"])
    df_final_eval = pd.read_csv(os.path.join(rev_dir, "final_evaluations.csv"))
    df_hist_bench = pd.read_csv(os.path.join("results", "benchmark.csv"))

    bench_rows = _extract_markdown_table_rows(md, "### 7.1 Holdout Generalization and Latency Comparison")
    panel_a_map = {
        "Repeated DOE MO": "repeated_preplanned_doe_multi_objective",
        "Multi-Objective TPE": "multi_objective_tpe",
        "Constrained TPE": "constrained_tpe",
        "Repeated DOE SO": "repeated_preplanned_doe_single_objective",
        "Single-Objective TPE": "single_objective_tpe",
        "Unguided Random Search": "random_search",
    }

    matched_panel_a = 0
    for row in bench_rows:
        label = row[0]
        for prefix, opt_key in panel_a_map.items():
            if prefix in label and "Historical" not in label:
                matched_panel_a += 1
                canon = opt_summary[opt_key]
                sub_sum = df_final_sum[df_final_sum["optimizer"] == opt_key]
                sub_eval = df_final_eval[df_final_eval["optimizer"] == opt_key]
                retrain_sd_canon = float(sub_eval.groupby("selection_id")["test_rmse"].std().mean())
                canon_reps = canon.get("optimizer_replicates", canon.get("selection_replicates"))

                # Column 1: Replicates N=20
                reps_nums = _floats_in(row[1])
                assert int(reps_nums[0]) == canon_reps == 20

                # Column 2: Val RMSE (Mean ± SD)
                val_nums = _floats_in(row[2])
                assert val_nums[0] == pytest.approx(canon["independently_retrained_validation_rmse"]["mean"], abs=1e-4)
                assert val_nums[1] == pytest.approx(canon["independently_retrained_validation_rmse"]["standard_deviation"], abs=1e-4)

                # Column 3: Holdout Test RMSE (Mean ± SD)
                test_nums = _floats_in(row[3])
                assert test_nums[0] == pytest.approx(canon["final_test_rmse"]["mean"], abs=1e-4)
                assert test_nums[1] == pytest.approx(canon["final_test_rmse"]["standard_deviation"], abs=1e-4)

                # Column 4: Holdout Test RMSE [95% CI]
                ci_nums = _floats_in(row[4])
                assert ci_nums[0] == pytest.approx(canon["final_test_rmse"]["confidence_interval_95"][0], abs=1e-4)
                assert ci_nums[1] == pytest.approx(canon["final_test_rmse"]["confidence_interval_95"][1], abs=1e-4)

                # Column 5: Retrain SD
                retrain_nums = _floats_in(row[5])
                assert retrain_nums[0] == pytest.approx(retrain_sd_canon, abs=1e-4)

                # Column 6: predict Latency (Mean ± SD)
                pred_lat_nums = _floats_in(row[6])
                assert pred_lat_nums[0] == pytest.approx(float(sub_sum["predict_latency_us"].mean()), abs=0.1)
                assert pred_lat_nums[1] == pytest.approx(float(sub_sum["predict_latency_us"].std()), abs=0.1)

                # Column 7: inplace_predict Latency (Mean ± SD)
                inpl_lat_nums = _floats_in(row[7])
                assert inpl_lat_nums[0] == pytest.approx(float(sub_sum["inplace_predict_latency_us"].mean()), abs=0.1)
                assert inpl_lat_nums[1] == pytest.approx(float(sub_sum["inplace_predict_latency_us"].std()), abs=0.1)

                # Column 8: Feasible count / 20
                feas_nums = _floats_in(row[8])
                assert int(feas_nums[0]) == int(sub_sum["benchmark_time_feasible"].sum())
                assert int(feas_nums[1]) == canon_reps

    assert matched_panel_a == 6, f"Expected 6 Panel A rows in Section 7.1 table, matched {matched_panel_a}"

    # Verify Panel B historical baseline rows against results/benchmark.csv
    panel_b_map = {
        "Historical DOE MO": "Sequential DOE-CCD (x*, Multi-Objective)",
        "Historical Multi-Obj TPE": "Multi-Objective TPE (Desirability)",
        "Historical Constrained TPE": "Constrained TPE (Latency <= 145 us)",
        "Historical DOE SO": "Sequential DOE-CCD (Single-Objective)",
        "Historical Bayesian TPE": "Bayesian Optimization (Optuna TPE Single-Obj)",
        "Historical Random Search": "Unguided Random Search",
    }
    matched_panel_b = 0
    for row in bench_rows:
        label = row[0]
        for prefix, csv_method in panel_b_map.items():
            if prefix in label:
                matched_panel_b += 1
                hist_row = df_hist_bench[df_hist_bench["method"] == csv_method].iloc[0]
                assert _floats_in(row[2])[0] == pytest.approx(float(hist_row["val_rmse_mean"]), abs=1e-4)
                assert _floats_in(row[3])[0] == pytest.approx(float(hist_row["test_rmse_mean"]), abs=1e-4)
                assert _floats_in(row[6])[0] == pytest.approx(float(hist_row["predict_latency_us_median"]), abs=0.1)
                if pd.notna(hist_row["inplace_latency_us_median"]) and _floats_in(row[7]):
                    assert _floats_in(row[7])[0] == pytest.approx(float(hist_row["inplace_latency_us_median"]), abs=0.1)

    assert matched_panel_b == 6, f"Expected 6 Panel B rows in Section 7.1 table, matched {matched_panel_b}"

    # 4. Numerically verify Section 7.2 Hypervolume Table against hypervolume.json
    with open(os.path.join(rev_dir, "hypervolume.json"), "r", encoding="utf-8") as f:
        hv_data = json.load(f)

    hv_rows = _extract_markdown_table_rows(md, "### 7.2 Multi-Objective Pareto Hypervolume Comparison")
    dev_060 = hv_data["development_domain"]["reference_points"]["[0.6, 250.0]"]
    dev_065 = hv_data["development_domain"]["reference_points"]["[0.65, 275.0]"]
    test_060 = hv_data["external_test_domain"]["reference_points"]["[0.6, 250.0]"]
    test_065 = hv_data["external_test_domain"]["reference_points"]["[0.65, 275.0]"]

    motpe_pts = [len(v["pareto"]["points"]) for v in dev_060["mo_tpe_by_replicate"].values()]
    doe_pts = [len(v["pareto"]["points"]) for v in dev_060["repeated_doe_by_replicate"].values()]

    # Row 1: MO-TPE Candidates (Single Split 42, 30-Call Search)
    r_motpe = [r for r in hv_rows if "Multi-Objective TPE Candidates" in r[0]][0]
    assert _floats_in(r_motpe[2])[0] == pytest.approx(float(pd.Series(motpe_pts).mean()), abs=0.05)
    assert _floats_in(r_motpe[2])[1] == pytest.approx(float(pd.Series(motpe_pts).std(ddof=1)), abs=0.1)
    assert _floats_in(r_motpe[3])[0] == pytest.approx(dev_060["mo_tpe_hypervolume_distribution"]["mean"], abs=1e-4)
    assert _floats_in(r_motpe[3])[1] == pytest.approx(dev_060["mo_tpe_hypervolume_distribution"]["standard_deviation"], abs=1e-4)
    assert _floats_in(r_motpe[4])[0] == pytest.approx(dev_065["mo_tpe_hypervolume_distribution"]["mean"], abs=1e-4)
    assert _floats_in(r_motpe[4])[1] == pytest.approx(dev_065["mo_tpe_hypervolume_distribution"]["standard_deviation"], abs=1e-4)

    # Row 2: Repeated DOE Candidate Fronts (5-Block Means Across Replicates)
    r_rdoe = [r for r in hv_rows if "Repeated DOE Candidate Fronts" in r[0]][0]
    assert _floats_in(r_rdoe[2])[0] == pytest.approx(float(pd.Series(doe_pts).mean()), abs=0.05)
    assert _floats_in(r_rdoe[2])[1] == pytest.approx(float(pd.Series(doe_pts).std(ddof=1)), abs=0.1)
    assert _floats_in(r_rdoe[3])[0] == pytest.approx(dev_060["repeated_doe_hypervolume_distribution"]["mean"], abs=1e-4)
    assert _floats_in(r_rdoe[3])[1] == pytest.approx(dev_060["repeated_doe_hypervolume_distribution"]["standard_deviation"], abs=1e-4)
    assert _floats_in(r_rdoe[4])[0] == pytest.approx(dev_065["repeated_doe_hypervolume_distribution"]["mean"], abs=1e-4)
    assert _floats_in(r_rdoe[4])[1] == pytest.approx(dev_065["repeated_doe_hypervolume_distribution"]["standard_deviation"], abs=1e-4)

    # Row 3: Fixed Full DOE Evaluated Front (Single Split 42, 27 Configs)
    r_fdoe = [r for r in hv_rows if "Fixed Full DOE Evaluated Front" in r[0]][0]
    assert int(_floats_in(r_fdoe[1])[0]) == dev_060["full_doe_evaluated_candidate_front"]["pareto"]["input_count"]
    assert int(_floats_in(r_fdoe[2])[0]) == len(dev_060["full_doe_evaluated_candidate_front"]["pareto"]["points"])
    assert _floats_in(r_fdoe[3])[0] == pytest.approx(dev_060["full_doe_evaluated_candidate_front"]["value"], abs=1e-4)
    assert _floats_in(r_fdoe[4])[0] == pytest.approx(dev_065["full_doe_evaluated_candidate_front"]["value"], abs=1e-4)

    # Row 4: Historical DOE 2-Point Set
    r_hdoe = [r for r in hv_rows if "Historical DOE 2-Point Set" in r[0]][0]
    assert _floats_in(r_hdoe[3])[0] == pytest.approx(dev_060["doe_two_point"]["value"], abs=1e-4)
    assert _floats_in(r_hdoe[4])[0] == pytest.approx(dev_065["doe_two_point"]["value"], abs=1e-4)

    # Row 5: Difference: Fixed Full DOE Front - MO-TPE Mean (Split 42)
    r_diff = [r for r in hv_rows if "Difference:" in r[0]][0]
    diff_060 = dev_060["full_doe_evaluated_candidate_front"]["value"] - dev_060["mo_tpe_hypervolume_distribution"]["mean"]
    diff_065 = dev_065["full_doe_evaluated_candidate_front"]["value"] - dev_065["mo_tpe_hypervolume_distribution"]["mean"]
    assert _floats_in(r_diff[3])[0] == pytest.approx(diff_060, abs=1e-4)
    assert _floats_in(r_diff[4])[0] == pytest.approx(diff_065, abs=1e-4)

    # Panel B Row: Holdout Non-Dominated Set (12 Distinct Configs)
    r_holdout = [r for r in hv_rows if "Non-Dominated Set (12 Distinct Configs)" in r[0]][0]
    assert int(_floats_in(r_holdout[1])[0]) == test_060["all_frozen_selections"]["pareto"]["input_count"] == 122
    assert int(_floats_in(r_holdout[2])[0]) == len(test_060["all_frozen_selections"]["pareto"]["points"]) == 12
    assert _floats_in(r_holdout[3])[0] == pytest.approx(test_060["all_frozen_selections"]["value"], abs=1e-4)
    assert _floats_in(r_holdout[4])[0] == pytest.approx(test_065["all_frozen_selections"]["value"], abs=1e-4)

    # 5. Guard against stale/invented numbers in both REPORT.md and report.tex
    stale_benchmark_numbers = [
        "0.46624",
        "0.46941",
        "0.47121",
        "0.48996",
        "0.48926",
        "0.46960",
        "179.96",
        "165.43",
        "143.65",
        "16.9245",
        "17.1636",
        "31.6531",
        "29.6995",
        "29.5918",
        "29.9328",
        "136.62",
        "150.54",
        "186.19",
    ]
    for doc_name, text in [("REPORT.md", md), ("report.tex", tex)]:
        for stale in stale_benchmark_numbers:
            assert stale not in text, f"Found stale/non-canonical value '{stale}' in {doc_name}"

    # 6. Numerically verify generated LaTeX tables (tab_benchmarks.tex & tab_hypervolume_comparison.tex)
    with open(os.path.join("tables", "tab_benchmarks.tex"), "r", encoding="utf-8") as f:
        tex_bm_lines = [ln.strip() for ln in f if "&" in ln and "\\textbf{Optimization}" not in ln]
    matched_tex_bm = 0
    for ln in tex_bm_lines:
        cells = [c.strip() for c in ln.rstrip("\\").split("&")]
        for prefix, opt_key in panel_a_map.items():
            if prefix in cells[0] and "Historical" not in cells[0]:
                matched_tex_bm += 1
                canon = opt_summary[opt_key]
                sub_sum = df_final_sum[df_final_sum["optimizer"] == opt_key]
                sub_eval = df_final_eval[df_final_eval["optimizer"] == opt_key]
                retrain_sd_canon = float(sub_eval.groupby("selection_id")["test_rmse"].std().mean())
                assert _floats_in(cells[2])[0] == pytest.approx(canon["independently_retrained_validation_rmse"]["mean"], abs=1e-4)
                assert _floats_in(cells[2])[1] == pytest.approx(canon["independently_retrained_validation_rmse"]["standard_deviation"], abs=1e-4)
                assert _floats_in(cells[3])[0] == pytest.approx(canon["final_test_rmse"]["mean"], abs=1e-4)
                assert _floats_in(cells[3])[1] == pytest.approx(canon["final_test_rmse"]["standard_deviation"], abs=1e-4)
                assert _floats_in(cells[4])[0] == pytest.approx(retrain_sd_canon, abs=1e-4)
                assert _floats_in(cells[5])[0] == pytest.approx(float(sub_sum["predict_latency_us"].mean()), abs=0.1)
                assert _floats_in(cells[5])[1] == pytest.approx(float(sub_sum["predict_latency_us"].std()), abs=0.1)
                assert int(_floats_in(cells[6])[0]) == int(sub_sum["benchmark_time_feasible"].sum())
    assert matched_tex_bm == 6

    with open(os.path.join("tables", "tab_hypervolume_comparison.tex"), "r", encoding="utf-8") as f:
        tex_hv_text = f.read()
    assert f"{dev_060['mo_tpe_hypervolume_distribution']['mean']:.4f}" in tex_hv_text
    assert f"{dev_060['repeated_doe_hypervolume_distribution']['mean']:.4f}" in tex_hv_text
    assert f"{dev_060['full_doe_evaluated_candidate_front']['value']:.4f}" in tex_hv_text
    assert f"{test_060['all_frozen_selections']['value']:.4f}" in tex_hv_text

    # 7. Numerically verify Phase 1 ANOVA, Phase 2 CCD ANOVA, and Depth Opt tables in REPORT.md
    p1_md_rows = _extract_markdown_table_rows(md, "### 2.1 Phase 1 ANOVA")
    with open(os.path.join("tables", "tab_anova_phase1.tex"), "r", encoding="utf-8") as f:
        p1_tex_rows = [ln.strip() for ln in f if "&" in ln and "\\textbf{Source" not in ln]
    assert len(p1_md_rows) == len(p1_tex_rows) == 13
    for md_r, tex_ln in zip(p1_md_rows, p1_tex_rows):
        tex_cells = [c.strip() for c in tex_ln.rstrip("\\").split("&")]
        assert _floats_in(md_r[1])[0] == pytest.approx(_floats_in(tex_cells[1])[0], abs=1e-6)
        assert int(_floats_in(md_r[2])[0]) == int(_floats_in(tex_cells[2])[0])
        assert _floats_in(md_r[3])[0] == pytest.approx(_floats_in(tex_cells[3])[0], abs=1e-6)

    p2_md_rows = _extract_markdown_table_rows(md, "### 3.1 Second-Order Response Surface ANOVA")
    with open(os.path.join("tables", "tab_anova_ccd_rmse.tex"), "r", encoding="utf-8") as f:
        p2_tex_rows = [ln.strip() for ln in f if "&" in ln and "\\textbf{Source" not in ln]
    assert len(p2_md_rows) == len(p2_tex_rows) == 17
    for md_r, tex_ln in zip(p2_md_rows, p2_tex_rows):
        tex_cells = [c.strip() for c in tex_ln.rstrip("\\").split("&")]
        assert _floats_in(md_r[1])[0] == pytest.approx(_floats_in(tex_cells[1])[0], abs=1e-6)
        assert int(_floats_in(md_r[2])[0]) == int(_floats_in(tex_cells[2])[0])
        assert _floats_in(md_r[3])[0] == pytest.approx(_floats_in(tex_cells[3])[0], abs=1e-6)

    depth_md_rows = _extract_markdown_table_rows(md, "## 4. Phase 3: Canonical Spectral Analysis")
    df_depth = pd.read_csv(os.path.join("results", "depth_opt_table.csv"))
    assert len(depth_md_rows) == len(df_depth) == 7
    for md_r, (_, d_row) in zip(depth_md_rows, df_depth.iterrows()):
        assert int(_floats_in(md_r[0])[0]) == int(d_row["depth"])
        assert _floats_in(md_r[1])[0] == pytest.approx(float(d_row["x1"]), abs=1e-4)
        assert _floats_in(md_r[6])[0] == pytest.approx(float(d_row["pred_rmse"]), abs=1e-4)
        assert _floats_in(md_r[7])[0] == pytest.approx(float(d_row["se"]), abs=1e-4)




