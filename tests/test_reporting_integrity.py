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
    build_changelog_claim_specifications,
    build_scientific_artifact_registry,
    check_research_reporting,
    clear_custom_scientific_artifacts,
    discover_scientific_and_documentation_files,
    register_custom_scientific_artifact,
    render_changelog,
    render_report_markdown,
    render_report_sections,
    validate_changelog_claims,
    verify_scientific_artifact_registry,
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
    verify_canonical_evidence_hashes,
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

    # Copy top-level files discovered by the repository-wide artifact registry
    for fname in [
        "config.yaml",
        "REPORT.md",
        "README.md",
        "EXPERIMENT_PROTOCOL.md",
        "report.tex",
        "report.pdf",
        "technical_supplement.tex",
        "technical_supplement.pdf",
        "CHANGELOG.md",
        "plots.py",
    ]:
        shutil.copy2(root_dir / fname, ws / fname)

    # Copy full docs/ directory (generated manifest, provenance, milestone reports, and F/F.1 audit docs)
    shutil.copytree(root_dir / "docs", ws / "docs")

    # Copy tables
    shutil.copytree(root_dir / "tables", ws / "tables")

    # Copy registered publication figures and scripts cited by generated documents.
    shutil.copytree(root_dir / "figures", ws / "figures")
    shutil.copytree(root_dir / "scripts", ws / "scripts")
    for source in root_dir.glob("*.py"):
        shutil.copy2(source, ws / source.name)

    # Copy results top-level files, revision_v2/full_run_001, and revision_v2/scientific_audit_c1
    (ws / "results" / "revision_v2").mkdir(parents=True, exist_ok=True)
    for item in (root_dir / "results").iterdir():
        if item.is_file():
            shutil.copy2(item, ws / "results" / item.name)
    shutil.copytree(
        root_dir / "results" / "revision_v2" / "full_run_001",
        ws / "results" / "revision_v2" / "full_run_001",
    )
    shutil.copytree(
        root_dir / "results" / "revision_v2" / "scientific_audit_c1",
        ws / "results" / "revision_v2" / "scientific_audit_c1",
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
    with pytest.raises(ReportingDataError, match="Cryptographic evidence hash mismatch|confirmation_runs.csv seeds"):
        load_reporting_dataset(ws)
    conf_csv.write_text(orig_csv, encoding="utf-8")

    # 3. Corrupt confirmation seed in config.yaml
    cfg_path = ws / "config.yaml"
    orig_cfg = cfg_path.read_text(encoding="utf-8")
    cfg_path.write_text(orig_cfg.replace("505, 606", "105, 606"), encoding="utf-8")
    with pytest.raises(ReportingDataError, match="Cryptographic evidence hash mismatch|confirmation_seeds"):
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
    with pytest.raises(ReportingDataError, match="Cryptographic evidence hash mismatch|Primary hypervolume reference point"):
        load_reporting_dataset(ws)


def test_mutation_stale_latex_or_missing_marker_or_untraced_float_fails(tmp_path, canonical_dataset):
    """Negative test: stale LaTeX macro/table, missing marker, untraced outside float, hardcoded tex float, or TOST claim fails."""
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
    report_md.write_text(orig_md, encoding="utf-8")

    # 4. Hardcoded empirical result float injected into report.tex prose (e.g., 18.3834 or 0.0217)
    report_tex = ws / "report.tex"
    orig_tex = report_tex.read_text(encoding="utf-8")
    report_tex.write_text(
        orig_tex.replace(r"\numRevHvMoTpeDevMean", "18.3834", 1),
        encoding="utf-8",
    )
    errors_tex_float = audit_manuscript_integrity(ds_ws)
    assert any("Hardcoded empirical floating-point numbers found in report.tex prose" in e and "18.3834" in e for e in errors_tex_float)
    report_tex.write_text(orig_tex, encoding="utf-8")

    # 5. Unsupported equivalence claim injected into report.tex while rmse_equivalence_margin is null
    report_tex.write_text(
        orig_tex + "\n% The two methods are statistically equivalent under TOST.\n",
        encoding="utf-8",
    )
    errors_equiv = audit_manuscript_integrity(ds_ws)
    assert any("Unsupported equivalence claim" in e for e in errors_equiv)


def test_mutation_corrupted_audit_docs_or_nonexistent_paths_fail_closed(tmp_path, canonical_dataset):
    """Negative test: mutating empirical numbers or injecting nonexistent file paths in Work Package F audit docs must fail closed."""
    ws = _create_isolated_workspace(tmp_path, canonical_dataset.root_dir)
    ds_ws = load_reporting_dataset(ws)

    # 1. Corrupt Curvature SS 0.288786 -> 2.6602 in docs/scientific_claims_audit.md
    claims_path = ws / "docs" / "scientific_claims_audit.md"
    orig_claims = claims_path.read_text(encoding="utf-8")
    assert "0.288786" in orig_claims
    claims_path.write_text(orig_claims.replace("0.288786", "2.6602", 1), encoding="utf-8")
    ok_c, msg_c = check_research_reporting(ds_ws)
    assert not ok_c
    assert "scientific_claims_audit.md" in msg_c
    assert "2.6602" in msg_c
    claims_path.write_text(orig_claims, encoding="utf-8")

    # 2. Corrupt DOE MO test RMSE 0.48997 -> 0.48373 in docs/f_scientific_review_validation.md
    val_path = ws / "docs" / "f_scientific_review_validation.md"
    orig_val = val_path.read_text(encoding="utf-8")
    assert "0.48997" in orig_val
    val_path.write_text(orig_val.replace("0.48997", "0.48373", 1), encoding="utf-8")
    ok_v, msg_v = check_research_reporting(ds_ws)
    assert not ok_v
    assert "f_scientific_review_validation.md" in msg_v
    assert "0.48373" in msg_v
    val_path.write_text(orig_val, encoding="utf-8")

    # 3. Corrupt Constrained TPE test RMSE 0.47036 -> 0.46111 in docs/proposed_followup_experiments.md
    followup_path = ws / "docs" / "proposed_followup_experiments.md"
    orig_fup = followup_path.read_text(encoding="utf-8")
    assert "0.47036" in orig_fup
    followup_path.write_text(orig_fup.replace("0.47036", "0.46111", 1), encoding="utf-8")
    ok_f, msg_f = check_research_reporting(ds_ws)
    assert not ok_f
    assert "proposed_followup_experiments.md" in msg_f
    assert "0.46111" in msg_f
    followup_path.write_text(orig_fup, encoding="utf-8")

    # 4. Inject a nonexistent backticked path (`docs/canonical_evidence_hashes.json`) into docs/scientific_claims_audit.md
    claims_path.write_text(orig_claims + "\nSee `docs/canonical_evidence_hashes.json`.\n", encoding="utf-8")
    errs_path = audit_manuscript_integrity(ds_ws)
    assert any("docs/canonical_evidence_hashes.json" in e for e in errs_path)
    claims_path.write_text(orig_claims, encoding="utf-8")


def test_cryptographic_evidence_hash_protection_fails_on_tampered_artifact(tmp_path, canonical_dataset):
    """Negative test: tampering with any frozen canonical evidence file fails SHA-256 verification."""
    ws = _create_isolated_workspace(tmp_path, canonical_dataset.root_dir)
    verify_canonical_evidence_hashes(ws)

    # Tamper with a top-level v1.0.0 artifact
    p1_json = ws / "results" / "phase1.json"
    orig_p1 = p1_json.read_text(encoding="utf-8")
    p1_json.write_text(orig_p1 + "\n", encoding="utf-8")
    with pytest.raises(ReportingDataError, match="Cryptographic evidence hash mismatch"):
        verify_canonical_evidence_hashes(ws)
    p1_json.write_text(orig_p1, encoding="utf-8")

    # Tamper with a prospective full_run_001 artifact tracked in provenance.json
    hv_json = ws / "results" / "revision_v2" / "full_run_001" / "hypervolume.json"
    orig_hv = hv_json.read_text(encoding="utf-8")
    hv_json.write_text(orig_hv + "\n", encoding="utf-8")
    with pytest.raises(ReportingDataError, match="Cryptographic evidence hash mismatch.*provenance\\.json"):
        verify_canonical_evidence_hashes(ws)


def test_work_package_f_scientific_audit_reconstructions_and_deliverables(canonical_dataset):
    """Verifies Work Package F/F.1 scientific reconstructions, sensitivity analyses, and deliverables."""
    ds = canonical_dataset
    root = ds.root_dir

    # Task 2 & 5: Chronology, 1,323-point grid provenance, and low-eta boundary budget accounting
    assert ds.doe_axial_before_run100 == 30
    assert ds.doe_low_eta_run_count == 45
    assert ds.doe_low_eta_run_pct == pytest.approx(32.142857, abs=1e-4)
    assert ds.doe_low_eta_mean_rmse == pytest.approx(0.7545, abs=1e-3)
    assert ds.doe_high_eta_mean_rmse == pytest.approx(0.5055, abs=1e-3)
    assert ds.doe_grid_size == 1323
    assert ds.doe_so_rep_grid_index == 1052
    assert ds.doe_mo_rep_unique_hashes == 12

    # Task 3: Curvature sensitivity across 7 error denominators
    cs = ds.curvature_sensitivity
    assert cs["f_pe_center"] == pytest.approx(105432.31, abs=0.05)
    assert cs["f_block_adj_pooled"] == pytest.approx(20235.41, abs=0.05)
    assert cs["f_txb_p1"] == pytest.approx(17013.24, abs=0.05)
    assert cs["f_block_paired"] == pytest.approx(7342.69, abs=0.05)
    assert cs["t_block_paired"] == pytest.approx(85.6895, abs=1e-3)
    assert cs["f_pe_coord_unadj"] == pytest.approx(4140.06, abs=0.05)
    assert cs["f_2fi_curv_resid"] == pytest.approx(3251.19, abs=0.05)
    assert cs["f_2fi_resid"] == pytest.approx(82.86, abs=0.05)
    assert cs["ci95_block_paired_low"] == pytest.approx(0.1300, abs=1e-3)
    assert cs["ci95_block_paired_high"] == pytest.approx(0.1387, abs=1e-3)

    # Task 6: Latency interface overheads across all stages
    assert ds.hist_runs_interface_overhead_mean == pytest.approx(40.88, abs=0.05)
    assert ds.hist_conf_mo_interface_overhead_mean == pytest.approx(38.18, abs=0.05)
    assert ds.hist_conf_so_interface_overhead_mean == pytest.approx(36.85, abs=0.05)
    assert ds.hist_bench_interface_overhead_mean == pytest.approx(32.07, abs=0.05)
    assert ds.interface_overhead_mean == pytest.approx(32.19, abs=0.05)

    # Task 9: Computational budget accounting
    assert ds.total_warmup_inferences == 201270
    assert ds.total_all_inferences == 1842080
    assert "repeated_doe" in ds.method_timing_summary
    assert ds.method_timing_summary["repeated_doe"]["wall_s_sum"] == pytest.approx(729.76, abs=0.05)
    assert ds.method_timing_summary["repeated_doe"]["wall_s_mean"] == pytest.approx(36.49, abs=0.05)

    # Task 10: Holdout Pareto bootstrap
    hpb = ds.holdout_pareto_bootstrap
    assert hpb["n_boot"] == 2000
    assert hpb["seed_only_060_mean"] == pytest.approx(17.6923, abs=1e-3)
    assert hpb["seed_only_060_ci_low"] < 17.6714 < hpb["seed_only_060_ci_high"]
    assert hpb["seed_only_065_mean"] == pytest.approx(29.0170, abs=1e-3)
    assert hpb["seed_only_065_ci_low"] < 28.9900 < hpb["seed_only_065_ci_high"]
    assert hpb["by_optimizer"]["repeated_preplanned_doe_multi_objective"]["present_pct"] == pytest.approx(100.0)
    assert hpb["by_optimizer"]["multi_objective_tpe"]["present_pct"] == pytest.approx(100.0)
    assert hpb["by_optimizer"]["constrained_tpe"]["present_pct"] == pytest.approx(100.0)
    assert hpb["by_optimizer"]["single_objective_tpe"]["present_pct"] == pytest.approx(91.5, abs=0.1)

    # Deliverables existence & content verification
    claims_doc = (root / "docs" / "scientific_claims_audit.md").read_text(encoding="utf-8")
    assert "SUPPORTED" in claims_doc and "NEEDS QUALIFICATION" in claims_doc and "UNSUPPORTED" in claims_doc
    assert "Claim C-22" in claims_doc
    assert "0.288786" in claims_doc and "0.48997" in claims_doc and "17.6714" in claims_doc

    followup_doc = (root / "docs" / "proposed_followup_experiments.md").read_text(encoding="utf-8")
    assert "PROPOSED FUTURE WORK" in followup_doc
    for proto in ["Protocol A", "Protocol B", "Protocol C", "Protocol D", "Protocol E"]:
        assert proto in followup_doc

    val_doc = (root / "docs" / "f_scientific_review_validation.md").read_text(encoding="utf-8")
    assert val_doc.strip().endswith("**READY FOR INDEPENDENT SCIENTIFIC REVIEW**")

    f1_val_doc = (root / "docs" / "f1_final_audit_validation.md").read_text(encoding="utf-8")
    assert f1_val_doc.strip().endswith("READY FOR FINAL MERGE REVIEW")
    for canonical_token in [
        "0.28878615275038744",
        "1.4271325866e-05",
        "20,235.41",
        "0.48997 ± 0.00732",
        "0.0073189235",
        "0.49255 ± 0.01080",
        "0.0108040718",
        "17.6923 ± 0.0564",
        "136.70 ± 0.82",
        "150.45 ± 0.67",
        "0.23244253",
        "0.82952703",
        "0.15150460",
    ]:
        assert canonical_token in f1_val_doc, f"Missing canonical token {canonical_token} in f1_final_audit_validation.md"

    assert (root / "tables" / "tab_computational_budget.tex").is_file()

    # Title correction check
    tex_text = (root / "report.tex").read_text(encoding="utf-8")
    md_text = (root / "REPORT.md").read_text(encoding="utf-8")
    assert "Learning and Optimizing XGBoost with Planned Experiments" in tex_text
    assert md_text.startswith("# Pre-Planned Response Surface Methodology")


def test_scientific_artifact_registry_and_negative_coverage(tmp_path, canonical_dataset):
    """Negative test suite proving repository-wide scientific document coverage and fail-closed verification."""
    ws = _create_isolated_workspace(tmp_path, canonical_dataset.root_dir)
    ds_ws = load_reporting_dataset(ws)

    # Require complete coverage, including the new publication and preserved archives.
    discovered = set(discover_scientific_and_documentation_files(ws))
    registry = build_scientific_artifact_registry(ds_ws)
    assert discovered == set(registry.keys())
    assert 'docs/generated/publication_claim_bindings.json' in registry
    assert 'docs/archive/catalog.json' in registry
    assert verify_scientific_artifact_registry(ds_ws) == []

    # 1. Introduce a new docs/new_scientific_audit.md containing empirical results but no verification registration -> MUST FAIL
    new_doc = ws / "docs" / "new_scientific_audit.md"
    valid_new_content = (
        "# New Scientific Audit Summary\n\n"
        "- Repeated DOE MO holdout test RMSE: `0.48997` ± `0.00732`\n"
        "- MO-TPE holdout test RMSE: `0.49255` ± `0.01080`\n"
        "- Block-adjusted curvature F-statistic: `20,235.41` (SS = `0.288786`)\n"
    )
    new_doc.write_text(valid_new_content, encoding="utf-8")

    reg_errs_unreg = verify_scientific_artifact_registry(ds_ws)
    assert any(
        "Unregistered scientific or documentation artifact discovered: 'docs/new_scientific_audit.md'" in e
        for e in reg_errs_unreg
    )
    ok_unreg, msg_unreg = check_research_reporting(ds_ws)
    assert not ok_unreg
    assert "docs/new_scientific_audit.md" in msg_unreg

    # 2. Register docs/new_scientific_audit.md, verify it passes when valid, then corrupt an empirical value -> MUST FAIL
    from scripts.generate_research_reporting import render_f1_final_audit_validation
    inventory = ws / 'docs/f1_final_audit_validation.md'
    baseline_inventory = inventory.read_text(encoding='utf-8')
    try:
        register_custom_scientific_artifact(
            "docs/new_scientific_audit.md",
            mechanism="claim_registry_verified",
            source_artifacts=[
                "results/phase1.json",
                "results/revision_v2/full_run_001/final_summary.csv",
            ],
            required_claims=["0.48997", "0.00732", "0.49255", "0.01080", "20,235.41", "0.288786"],
            allowed_context_floats=(),
            description="Custom registered scientific audit document for negative testing.",
        )
        inventory.write_text(render_f1_final_audit_validation(ds_ws), encoding='utf-8')
        assert verify_scientific_artifact_registry(ds_ws) == []
        ok_reg, msg_reg = check_research_reporting(ds_ws)
        assert ok_reg, f"Expected registered valid document to pass, got: {msg_reg}"

        # Now corrupt an empirical value in the registered document (0.00732 -> 0.01325)
        new_doc.write_text(valid_new_content.replace("0.00732", "0.01325"), encoding="utf-8")
        reg_errs_corrupt = verify_scientific_artifact_registry(ds_ws)
        assert any("docs/new_scientific_audit.md" in e and "0.01325" in e for e in reg_errs_corrupt)
        ok_corrupt, msg_corrupt = check_research_reporting(ds_ws)
        assert not ok_corrupt
        assert "docs/new_scientific_audit.md" in msg_corrupt
        assert "0.01325" in msg_corrupt
    finally:
        clear_custom_scientific_artifacts()
        if new_doc.exists():
            new_doc.unlink()
        inventory.write_text(baseline_inventory, encoding='utf-8')

    # 3. Corrupt an empirical value in docs/f1_final_audit_validation.md -> MUST FAIL
    f1_doc = ws / "docs" / "f1_final_audit_validation.md"
    orig_f1 = f1_doc.read_text(encoding="utf-8")
    assert "0.48997 ± 0.00732" in orig_f1
    f1_doc.write_text(orig_f1.replace("0.48997 ± 0.00732", "0.48997 ± 0.01325", 1), encoding="utf-8")
    ok_f1, msg_f1 = check_research_reporting(ds_ws)
    assert not ok_f1
    assert "docs/f1_final_audit_validation.md" in msg_f1
    assert "0.01325" in msg_f1
    f1_doc.write_text(orig_f1, encoding="utf-8")

    # Also corrupt an empirical value in a claim-registry verified milestone report (docs/c1_final_assessment.md) -> MUST FAIL
    c1_doc = ws / "docs" / "c1_final_assessment.md"
    orig_c1 = c1_doc.read_text(encoding="utf-8")
    c1_doc.write_text(orig_c1.replace("0.48997", "0.48373", 1), encoding="utf-8")
    ok_c1, msg_c1 = check_research_reporting(ds_ws)
    assert not ok_c1
    assert "docs/c1_final_assessment.md" in msg_c1
    assert "0.48373" in msg_c1
    c1_doc.write_text(orig_c1, encoding="utf-8")

    # 4. Corrupt a source artifact -> Existing cryptographic and reconstruction checks MUST FAIL
    conf_json = ws / "results" / "confirmation.json"
    orig_conf = conf_json.read_text(encoding="utf-8")
    conf_json.write_text(orig_conf.replace("0.23244253", "0.14490000", 1), encoding="utf-8")
    with pytest.raises(ReportingDataError, match="Cryptographic evidence hash mismatch"):
        load_reporting_dataset(ws)
    conf_json.write_text(orig_conf, encoding="utf-8")


def test_changelog_context_bound_claims_and_negative_mutations(tmp_path, canonical_dataset):
    """Negative test suite proving context-bound, campaign-specific verification of CHANGELOG.md."""
    ws = _create_isolated_workspace(tmp_path, canonical_dataset.root_dir)
    ds_ws = load_reporting_dataset(ws)

    changelog_path = ws / "CHANGELOG.md"
    orig_changelog = changelog_path.read_text(encoding="utf-8")

    # Baseline verification: deterministic rendering and 16-row claim specification
    specs = build_changelog_claim_specifications(ds_ws)
    assert len(specs) == 16
    assert orig_changelog == render_changelog(ds_ws)
    assert validate_changelog_claims(ds_ws, orig_changelog) == []
    ok_base, msg_base = check_research_reporting(ds_ws)
    assert ok_base, f"Expected baseline workspace to pass, got: {msg_base}"

    # Verify required prospective Single-Objective comparison values are present in CHANGELOG.md
    assert "Single-Objective TPE: test RMSE $0.46691 \\pm 0.00116$, `predict` latency $188.93 \\pm 12.07\\,\\mu\\text{s}$" in orig_changelog
    assert "Repeated DOE Single-Objective: test RMSE $0.46859 \\pm 0.00000$ (conditional retraining SD $0.00304$), `predict` latency $150.74 \\pm 0.72\\,\\mu\\text{s}$" in orig_changelog

    # Negative Test 1: Replace SO-TPE RMSE 0.46691 with obsolete 0.4673 in the prospective Changelog row -> MUST FAIL
    mut_obsolete_sotpe = orig_changelog.replace(
        "Single-Objective TPE: test RMSE $0.46691 \\pm 0.00116$",
        "Single-Objective TPE: test RMSE $0.4673 \\pm 0.00116$",
        1,
    )
    assert mut_obsolete_sotpe != orig_changelog
    changelog_path.write_text(mut_obsolete_sotpe, encoding="utf-8")
    claim_errs_1 = validate_changelog_claims(ds_ws)
    assert any("prospective_sotpe_summary" in e or "0.4673" in e for e in claim_errs_1)
    ok_1, msg_1 = check_research_reporting(ds_ws)
    assert not ok_1
    assert "CHANGELOG.md" in msg_1
    assert "0.4673" in msg_1 or "prospective_sotpe_summary" in msg_1
    changelog_path.write_text(orig_changelog, encoding="utf-8")

    # Negative Test 2: Swap DOE SO (150.74) and SO-TPE (188.93) latency values -> MUST FAIL even though both values are canonical
    mut_swapped_lat = (
        orig_changelog.replace("188.93 \\pm 12.07", "__TMP_LAT__", 1)
        .replace("150.74 \\pm 0.72", "188.93 \\pm 12.07", 1)
        .replace("__TMP_LAT__", "150.74 \\pm 0.72", 1)
    )
    assert mut_swapped_lat != orig_changelog
    changelog_path.write_text(mut_swapped_lat, encoding="utf-8")
    claim_errs_2 = validate_changelog_claims(ds_ws)
    assert any("prospective_sotpe_summary" in e for e in claim_errs_2)
    assert any("prospective_doe_so_summary" in e for e in claim_errs_2)
    ok_2, msg_2 = check_research_reporting(ds_ws)
    assert not ok_2
    assert "CHANGELOG.md" in msg_2
    assert "prospective_sotpe_summary" in msg_2
    changelog_path.write_text(orig_changelog, encoding="utf-8")

    # Negative Test 3: Corrupt a historical benchmark value while preserving its historical label -> MUST FAIL
    assert "[HISTORICAL SPLIT-42 BASELINE: results/benchmark.csv]" in orig_changelog
    mut_hist_corrupt = orig_changelog.replace(
        "DOE $\\mathbf{x}^*_{\\text{SO}}$ (Depth 7) Test RMSE $0.4690$, median `predict` latency $147.84\\,\\mu\\text{s}$",
        "DOE $\\mathbf{x}^*_{\\text{SO}}$ (Depth 7) Test RMSE $0.4668$, median `predict` latency $147.84\\,\\mu\\text{s}$",
        1,
    )
    assert mut_hist_corrupt != orig_changelog
    assert "[HISTORICAL SPLIT-42 BASELINE: results/benchmark.csv]" in mut_hist_corrupt
    changelog_path.write_text(mut_hist_corrupt, encoding="utf-8")
    claim_errs_3 = validate_changelog_claims(ds_ws)
    assert any("historical_split42_so_comparison" in e or "0.4668" in e for e in claim_errs_3)
    ok_3, msg_3 = check_research_reporting(ds_ws)
    assert not ok_3
    assert "CHANGELOG.md" in msg_3
    assert "historical_split42_so_comparison" in msg_3 or "0.4668" in msg_3
    changelog_path.write_text(orig_changelog, encoding="utf-8")

    # Negative Test 4: Add a new numerical claim to the Changelog without a source mapping -> MUST FAIL
    # Case 4a: Unmapped numerical claim appended inside an existing row
    mut_unmapped_in_row = orig_changelog.replace(
        "Revision-v2 enforced programmatic holdout isolation during optimization and candidate selection.",
        "Revision-v2 enforced programmatic holdout isolation during optimization and candidate selection (auxiliary holdout drift 0.00412).",
        1,
    )
    changelog_path.write_text(mut_unmapped_in_row, encoding="utf-8")
    claim_errs_4a = validate_changelog_claims(ds_ws)
    assert any("unmapped numerical claim" in e and "0.00412" in e for e in claim_errs_4a)
    ok_4a, msg_4a = check_research_reporting(ds_ws)
    assert not ok_4a
    assert "0.00412" in msg_4a

    # Case 4b: Unmapped extra table row or trailing numerical note added to CHANGELOG.md
    mut_unmapped_row = (
        orig_changelog
        + "| **Unmapped Extra Benchmark** | Initial 0.5123 | **Final 0.4789** | Unmapped row |\n"
    )
    changelog_path.write_text(mut_unmapped_row, encoding="utf-8")
    claim_errs_4b = validate_changelog_claims(ds_ws)
    assert any("Unmapped numerical claim row" in e for e in claim_errs_4b)
    ok_4b, msg_4b = check_research_reporting(ds_ws)
    assert not ok_4b
    assert "CHANGELOG.md" in msg_4b
    changelog_path.write_text(orig_changelog, encoding="utf-8")




@pytest.mark.parametrize('mutation', ['swap', 'duplicate', 'unit', 'caveat', 'bare_number'])
def test_reader_publication_rejects_context_mutations(tmp_path, canonical_dataset, mutation):
    from scripts.publication import check_publication_source, bindings
    ws = _create_isolated_workspace(tmp_path, canonical_dataset.root_dir)
    ds = load_reporting_dataset(ws)
    manuscript = ws / 'report.tex'
    original = manuscript.read_text(encoding='utf-8')
    if mutation == 'swap':
        changed = original.replace(r'\numPubOnlineCalls', r'\numPubPrimaryCalls', 1)
    elif mutation == 'duplicate':
        changed = original.replace(r'\numPubOnlineCalls', r'\numPubOnlineCalls{} \numPubOnlineCalls', 1)
    elif mutation == 'unit':
        changed = original.replace('microseconds', 'milliseconds', 1)
    elif mutation == 'caveat':
        changed = original.replace('not', 'always', 1)
    else:
        template = ws / 'scripts/templates/report.tex'
        # Insert inside the scientific body, before the bibliography exclusion.
        template.write_text(original.replace(r'\begin{thebibliography}', 'Unmapped result: 0.12345\n'+r'\begin{thebibliography}'), encoding='utf-8')
        with pytest.raises(ValueError, match='Unbound numeric literals'):
            bindings(ds)
        return
    assert changed != original
    manuscript.write_text(changed, encoding='utf-8')
    assert check_publication_source(ds)


def test_reader_publication_rejects_stale_pdf_inputs(tmp_path, canonical_dataset):
    from scripts.publication import check_pdf
    ws = _create_isolated_workspace(tmp_path, canonical_dataset.root_dir)
    ds = load_reporting_dataset(ws)
    assert check_pdf(ds) == []
    source = ws / 'report.tex'
    source.write_text(source.read_text(encoding='utf-8')+'\n% Changed input\n', encoding='utf-8')
    assert any('stale' in error for error in check_pdf(ds))
