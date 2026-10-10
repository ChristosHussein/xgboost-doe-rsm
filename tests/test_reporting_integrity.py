"""
tests/test_reporting_integrity.py - Comprehensive verification suite for Work Package D.2:
Single Source of Truth, Automated Scientific Reporting, and Numerical Integrity.

Covers:
1. Canonical artifact loading and fail-closed schema validation.
2. Independent row-level recalculation of source-derived metrics.
3. Exact agreement between REPORT.md, report.tex, results/macros.tex, tables/*.tex,
   and docs/generated/scientific_results_manifest.json.
4. Idempotent / exact reproduction of generated sections (--check mode).
5. Deliberately failing mutation test fixtures (mutated Markdown values, mutated
   confirmation seeds, mutated hypervolume reference points, stale LaTeX outputs,
   missing section markers, untraced outside-block floats, and unsupported equivalence claims).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.generate_report_artifacts import (
    check_report_artifacts,
    render_macros,
    render_tables,
)
from scripts.generate_research_reporting import (
    SECTION_IDS,
    audit_manuscript_integrity,
    check_research_reporting,
    render_report_markdown,
    render_report_sections,
)
from scripts.reporting_data import (
    EXPECTED_BLOCK_SEEDS,
    EXPECTED_CONFIRMATION_SEEDS,
    EXPECTED_FRESH_EVAL_SEEDS,
    EXPECTED_HV_REF_PRIMARY,
    EXPECTED_HV_REF_SECONDARY,
    PROSPECTIVE_OPTIMIZER_ORDER,
    ReportingDataError,
    build_scientific_results_manifest,
    load_reporting_dataset,
)


@pytest.fixture(scope="module")
def canonical_dataset():
    """Load the canonical reporting dataset once per test module."""
    return load_reporting_dataset()


def test_canonical_dataset_loads_and_validates_counts(canonical_dataset):
    """Verifies canonical seed sets, reference points, selection counts, and budget totals."""
    ds = canonical_dataset
    assert ds.block_seeds == EXPECTED_BLOCK_SEEDS == (42, 101, 202, 303, 404)
    assert ds.confirmation_seeds == EXPECTED_CONFIRMATION_SEEDS == (
        505,
        606,
        707,
        808,
        909,
        1010,
        1111,
        1212,
        1313,
        1414,
    )
    assert ds.fresh_eval_seeds == EXPECTED_FRESH_EVAL_SEEDS == tuple(range(2001, 2021))
    assert ds.hv_ref_primary == EXPECTED_HV_REF_PRIMARY == (0.60, 250.0)
    assert ds.hv_ref_secondary == EXPECTED_HV_REF_SECONDARY == (0.65, 275.0)

    # Selection record vs distinct SHA-256 config hash distinction
    assert ds.total_selection_records == 122
    assert ds.distinct_config_hashes == 95
    assert ds.total_evaluation_rows == 2440
    assert ds.total_model_fits == 17077
    assert ds.total_timed_inferences == 1640810


def test_row_level_reconstructions_match_canonical_summaries(canonical_dataset):
    """Verifies key reconstructed statistics from row-level CSVs against saved JSON summaries."""
    ds = canonical_dataset

    # Phase 1 curvature
    assert ds.phase1["f_curvature"] == pytest.approx(105432.3123, abs=1e-2)
    assert ds.phase1["yF_bar"] == pytest.approx(0.63328197, abs=1e-6)
    assert ds.phase1["yC_bar"] == pytest.approx(0.49893490, abs=1e-6)

    # Phase 2 CCD & ICC
    assert ds.ccd_y1_rsq == pytest.approx(0.9951, abs=1e-4)
    assert ds.ccd_y1_adj_rsq == pytest.approx(0.9944, abs=1e-4)
    assert ds.icc["Y1"]["icc_anova"] == pytest.approx(0.40685167, abs=1e-6)
    assert ds.icc["Y1"]["icc_reml"] == pytest.approx(0.40689768, abs=1e-6)
    assert ds.lof_decomp["f_lof_center_y1"] == pytest.approx(323.10, abs=1e-2)

    # Phase 4 Desirability
    assert ds.desirability_star["D"] == pytest.approx(0.6782, abs=1e-4)
    assert ds.desirability_star["d1"] == pytest.approx(0.8783, abs=1e-4)
    assert ds.desirability_star["d2"] == pytest.approx(0.5236, abs=1e-4)

    # Phase 5 Confirmation
    assert ds.confirmation["Y1_Val_RMSE"]["empirical_mean"] == pytest.approx(0.48528634, abs=1e-6)
    assert ds.confirmation["Y1_Test_RMSE"]["empirical_mean"] == pytest.approx(0.48877380, abs=1e-6)
    assert ds.confirmation["Y2_Latency"]["empirical_mean"] == pytest.approx(142.33135, abs=1e-4)

    # Prospective Optimizers (N=20 replicates each)
    doe_mo = ds.prospective_optimizers["repeated_preplanned_doe_multi_objective"]
    mo_tpe = ds.prospective_optimizers["multi_objective_tpe"]
    ctpe = ds.prospective_optimizers["constrained_tpe"]
    doe_so = ds.prospective_optimizers["repeated_preplanned_doe_single_objective"]
    so_tpe = ds.prospective_optimizers["single_objective_tpe"]
    rs = ds.prospective_optimizers["random_search"]

    assert doe_mo.test_rmse_mean == pytest.approx(0.48996859, abs=1e-6)
    assert doe_mo.test_rmse_sd == pytest.approx(0.00731892, abs=1e-6)
    assert doe_mo.predict_latency_mean == pytest.approx(122.29, abs=1e-2)
    assert doe_mo.benchmark_feasible_count == 20

    assert mo_tpe.test_rmse_mean == pytest.approx(0.49255059, abs=1e-6)
    assert mo_tpe.test_rmse_sd == pytest.approx(0.01080407, abs=1e-6)
    assert mo_tpe.predict_latency_mean == pytest.approx(121.46, abs=1e-2)
    assert mo_tpe.benchmark_feasible_count == 20

    assert ctpe.test_rmse_mean == pytest.approx(0.47036167, abs=1e-6)
    assert ctpe.search_feasible_count == 20
    assert ctpe.benchmark_feasible_count == 9
    assert ds.ctpe_depth6_count == 9 and ds.ctpe_depth6_feasible == 9
    assert ds.ctpe_depth7_count == 11 and ds.ctpe_depth7_feasible == 0

    assert doe_so.test_rmse_mean == pytest.approx(0.46859064, abs=1e-6)
    assert doe_so.test_rmse_sd == pytest.approx(0.0, abs=1e-12)
    assert so_tpe.test_rmse_mean == pytest.approx(0.46691285, abs=1e-6)
    assert rs.test_rmse_mean == pytest.approx(0.46953210, abs=1e-6)

    # Welch and paired t-tests
    assert ds.doe_mo_vs_motpe_diff == pytest.approx(-0.00258200, abs=1e-6)
    assert ds.doe_mo_vs_motpe_welch_t == pytest.approx(-0.8849, abs=1e-4)
    assert ds.doe_mo_vs_motpe_welch_p == pytest.approx(0.3826, abs=1e-4)
    assert ds.doe_mo_vs_motpe_paired_t == pytest.approx(-0.9658, abs=1e-4)
    assert ds.doe_mo_vs_motpe_paired_p == pytest.approx(0.3463, abs=1e-4)

    # Hypervolumes & Holdout non-dominated breakdown
    hvs = ds.hv_summary
    assert hvs["motpe_dev_060_mean"] == pytest.approx(18.3834, abs=1e-4)
    assert hvs["rdoe_dev_060_mean"] == pytest.approx(15.6128, abs=1e-4)
    assert hvs["fdoe_dev_060"] == pytest.approx(16.8585, abs=1e-4)
    assert hvs["holdout_frozen_060"] == pytest.approx(17.6714, abs=1e-4)
    assert hvs["holdout_nd"] == 12
    assert ds.holdout_nondominated_by_optimizer == {
        "repeated_preplanned_doe_multi_objective": 3,
        "multi_objective_tpe": 3,
        "constrained_tpe": 5,
        "repeated_preplanned_doe_single_objective": 0,
        "single_objective_tpe": 1,
        "random_search": 0,
    }


def test_check_modes_and_manuscript_audit_pass_on_clean_repo(canonical_dataset):
    """Verifies that both --check modes and manuscript auditing pass with zero errors."""
    ok_tex, msg_tex = check_report_artifacts(canonical_dataset)
    assert ok_tex, f"check_report_artifacts failed: {msg_tex}"

    ok_rep, msg_rep = check_research_reporting(canonical_dataset)
    assert ok_rep, f"check_research_reporting failed: {msg_rep}"

    audit_errors = audit_manuscript_integrity(canonical_dataset)
    assert audit_errors == [], f"audit_manuscript_integrity returned errors: {audit_errors}"


def test_manifest_schema_and_cross_document_agreement(canonical_dataset):
    """Verifies that docs/generated/scientific_results_manifest.json is complete and valid."""
    manifest_path = canonical_dataset.root_dir / "docs" / "generated" / "scientific_results_manifest.json"
    assert manifest_path.is_file(), "Missing docs/generated/scientific_results_manifest.json"

    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = build_scientific_results_manifest(canonical_dataset)
    assert on_disk == expected

    required_fields = {
        "metric_id",
        "value",
        "formatted_value",
        "precision",
        "input_artifact_path",
        "source_field_or_reconstruction",
        "evaluation_domain",
        "protocol",
        "n_independent_units",
        "uncertainty_definition",
        "verification_status",
        "target_documents",
    }
    assert len(on_disk["metrics"]) >= 50
    for item in on_disk["metrics"]:
        assert required_fields.issubset(item.keys()), f"Metric {item.get('metric_id')} missing fields"
        assert item["verification_status"] in {"RECONSTRUCTED", "SOURCE-TRACED"}
        artifact_file = canonical_dataset.root_dir / item["input_artifact_path"]
        assert artifact_file.is_file(), f"Manifest input_artifact_path does not exist: {artifact_file}"


def _create_isolated_workspace(tmp_path: Path, root_dir: Path) -> Path:
    """Create a lightweight isolated workspace in tmp_path for negative/mutation testing."""
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)

    # Copy top-level files
    for fname in ["config.yaml", "REPORT.md", "report.tex"]:
        shutil.copy2(root_dir / fname, ws / fname)

    # Copy docs/generated
    (ws / "docs" / "generated").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        root_dir / "docs" / "generated" / "scientific_results_manifest.json",
        ws / "docs" / "generated" / "scientific_results_manifest.json",
    )

    # Copy tables
    shutil.copytree(root_dir / "tables", ws / "tables")

    # Copy results top-level files and revision_v2/full_run_001
    (ws / "results" / "revision_v2").mkdir(parents=True, exist_ok=True)
    for item in (root_dir / "results").iterdir():
        if item.is_file():
            shutil.copy2(item, ws / "results" / item.name)
    shutil.copytree(
        root_dir / "results" / "revision_v2" / "full_run_001",
        ws / "results" / "revision_v2" / "full_run_001",
    )
    return ws


def test_mutation_corrupted_markdown_benchmark_value_fails_check(tmp_path, canonical_dataset):
    """Negative test: mutating a benchmark number in REPORT.md must fail check_research_reporting."""
    ws = _create_isolated_workspace(tmp_path, canonical_dataset.root_dir)
    ds_ws = load_reporting_dataset(ws)

    # Baseline passes
    ok, _ = check_research_reporting(ds_ws)
    assert ok

    # Corrupt Repeated DOE MO holdout test RMSE 0.48997 -> 0.48123 in REPORT.md
    report_md = ws / "REPORT.md"
    original_text = report_md.read_text(encoding="utf-8")
    assert "0.48997" in original_text
    report_md.write_text(original_text.replace("0.48997", "0.48123", 1), encoding="utf-8")

    ok_mut, msg_mut = check_research_reporting(ds_ws)
    assert not ok_mut
    assert "REPORT.md" in msg_mut
    assert "0.48123" in msg_mut


def test_mutation_corrupted_confirmation_seeds_fails_closed(tmp_path, canonical_dataset):
    """Negative test: mutating confirmation seeds in config.yaml, CSV, or REPORT.md must fail."""
    ws = _create_isolated_workspace(tmp_path, canonical_dataset.root_dir)

    # 1. Corrupt confirmation seed 505 -> 555 in REPORT.md
    report_md = ws / "REPORT.md"
    orig_md = report_md.read_text(encoding="utf-8")
    report_md.write_text(orig_md.replace("505, 606", "555, 606"), encoding="utf-8")
    ds_ws = load_reporting_dataset(ws)
    ok_md, msg_md = check_research_reporting(ds_ws)
    assert not ok_md
    assert "Confirmation seeds" in msg_md or "REPORT.md" in msg_md
    report_md.write_text(orig_md, encoding="utf-8")

    # 2. Corrupt confirmation seed 505 -> 555 in results/confirmation_runs.csv
    conf_csv = ws / "results" / "confirmation_runs.csv"
    orig_csv = conf_csv.read_text(encoding="utf-8")
    conf_csv.write_text(orig_csv.replace("505,", "555,", 1), encoding="utf-8")
    with pytest.raises(ReportingDataError, match="confirmation_runs.csv seeds"):
        load_reporting_dataset(ws)
    conf_csv.write_text(orig_csv, encoding="utf-8")

    # 3. Corrupt confirmation seed in config.yaml
    cfg_path = ws / "config.yaml"
    orig_cfg = cfg_path.read_text(encoding="utf-8")
    cfg_path.write_text(orig_cfg.replace("505, 606", "105, 606"), encoding="utf-8")
    with pytest.raises(ReportingDataError, match="confirmation_seeds"):
        load_reporting_dataset(ws)


def test_mutation_corrupted_hypervolume_reference_point_fails_closed(tmp_path, canonical_dataset):
    """Negative test: mutating hypervolume reference point in REPORT.md or config.yaml must fail."""
    ws = _create_isolated_workspace(tmp_path, canonical_dataset.root_dir)

    # 1. Corrupt reference point [0.60, 250] -> [0.55, 240] in REPORT.md
    report_md = ws / "REPORT.md"
    orig_md = report_md.read_text(encoding="utf-8")
    report_md.write_text(
        orig_md.replace("[0.60, 250]", "[0.55, 240]").replace("[0.60, 250.0]", "[0.55, 240.0]"),
        encoding="utf-8",
    )
    ds_ws = load_reporting_dataset(ws)
    ok_md, msg_md = check_research_reporting(ds_ws)
    assert not ok_md
    assert "Primary hypervolume reference point" in msg_md
    report_md.write_text(orig_md, encoding="utf-8")

    # 2. Corrupt reference point in config.yaml
    cfg_path = ws / "config.yaml"
    orig_cfg = cfg_path.read_text(encoding="utf-8")
    cfg_path.write_text(orig_cfg.replace("[0.60, 250.0]", "[0.55, 240.0]"), encoding="utf-8")
    with pytest.raises(ReportingDataError, match="Primary hypervolume reference point"):
        load_reporting_dataset(ws)


def test_mutation_stale_latex_or_missing_marker_or_untraced_float_fails(tmp_path, canonical_dataset):
    """Negative test: stale LaTeX macro/table, missing marker, untraced outside float, or TOST claim fails."""
    ws = _create_isolated_workspace(tmp_path, canonical_dataset.root_dir)
    ds_ws = load_reporting_dataset(ws)

    # 1. Stale LaTeX table
    bm_tex = ws / "tables" / "tab_benchmarks.tex"
    orig_bm = bm_tex.read_text(encoding="utf-8")
    bm_tex.write_text(orig_bm.replace("0.4900", "0.4999", 1), encoding="utf-8")
    ok_tex, msg_tex = check_report_artifacts(ds_ws)
    assert not ok_tex
    assert "tab_benchmarks.tex" in msg_tex
    bm_tex.write_text(orig_bm, encoding="utf-8")

    # 2. Missing auto-generated section marker in REPORT.md
    report_md = ws / "REPORT.md"
    orig_md = report_md.read_text(encoding="utf-8")
    report_md.write_text(
        orig_md.replace("<!-- BEGIN AUTO-GENERATED: EXEC_SUMMARY -->", ""),
        encoding="utf-8",
    )
    errors_marker = audit_manuscript_integrity(ds_ws)
    assert any("EXEC_SUMMARY" in e for e in errors_marker)

    # 3. Untraced floating-point number injected outside auto-generated blocks in REPORT.md
    report_md.write_text(orig_md + "\nUnverified extra result: 0.41234 RMSE.\n", encoding="utf-8")
    errors_float = audit_manuscript_integrity(ds_ws)
    assert any("Untraced floating-point numbers" in e for e in errors_float)

    # 4. Unsupported equivalence claim injected into report.tex while rmse_equivalence_margin is null
    report_md.write_text(orig_md, encoding="utf-8")
    report_tex = ws / "report.tex"
    orig_tex = report_tex.read_text(encoding="utf-8")
    report_tex.write_text(
        orig_tex + "\n% The two methods are statistically equivalent under TOST.\n",
        encoding="utf-8",
    )
    errors_equiv = audit_manuscript_integrity(ds_ws)
    assert any("Unsupported equivalence claim" in e for e in errors_equiv)
