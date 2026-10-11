import json
from pathlib import Path
import shutil
import pytest
import pandas as pd

from scripts.audit_scientific_consistency import (
    ScientificConsistencyError,
    audit_scientific_consistency,
    audit_test_isolation_and_integrity,
    audit_hypervolume_consistency,
    audit_computational_budget,
)

RUN_DIR = Path("results/revision_v2/full_run_001")
BUDGET_FILE = Path("results/revision_v2/scientific_audit_c1/budget_reconciliation.json")


def test_audit_scientific_consistency_full_run_passes():
    """Verify that the full scientific benchmark run passes all consistency checks."""
    if not RUN_DIR.exists():
        pytest.skip("Full benchmark run directory not present")

    results = audit_scientific_consistency(run_dir=RUN_DIR, budget_file=BUDGET_FILE, verbose=False)

    # 1. Budget
    assert results["computational_budget"]["total_model_fits"] == 17077
    assert results["computational_budget"]["optimizer_search_fits"] == 11200
    assert results["computational_budget"]["doe_selection_runs_fits"] == 2800
    assert results["computational_budget"]["final_evaluation_fits"] == 2440

    # 2. Test isolation
    assert results["test_isolation"]["holdout_leakage"] is False
    assert results["test_isolation"]["hash_mismatches"] == 0

    # 3. Hypervolumes
    hv = results["hypervolume"]
    assert hv["mo_tpe_mean_060"] == pytest.approx(18.3834, abs=1e-4)
    assert hv["full_doe_candidate_060"] == pytest.approx(16.8585, abs=1e-4)
    assert hv["difference_full_doe_minus_mo_tpe"] == pytest.approx(-1.5249, abs=1e-4)

    # 4. Selection stability
    assert results["selection_stability"]["doe_so_unique_configs"] == 1
    assert results["selection_stability"]["between_search_sd"] == pytest.approx(0.0)
    assert results["selection_stability"]["conditional_retraining_sd"] > 0.0025

    # 5. Constrained TPE
    assert results["constrained_tpe"]["search_feasible"] == "20/20"
    assert results["constrained_tpe"]["benchmark_feasible"] == "9/20"
    assert results["constrained_tpe"]["depth_breakdown"] == {7: 11, 6: 9}

    # 6. Statistical significance
    assert results["statistical_significance"]["statistically_significant"] is False
    assert results["statistical_significance"]["welch_p_value"] > 0.30
    assert results["statistical_significance"]["mean_difference"] == pytest.approx(-0.00258, abs=1e-4)

    # 7. Factor domain
    assert results["factor_hypercube"]["factor_ranges_valid"] is True


def test_audit_detects_test_leakage(tmp_path):
    """Verify that audit raises ScientificConsistencyError if test metrics appear in search trials."""
    if not RUN_DIR.exists():
        pytest.skip("Full benchmark run directory not present")

    # Create dummy dir with leaked test column in trials
    mock_dir = tmp_path / "mock_run"
    mock_dir.mkdir()
    trials = pd.read_csv(RUN_DIR / "optimizer_trials.csv", nrows=10)
    trials["test_rmse"] = 0.50  # Leaked test column
    trials.to_csv(mock_dir / "optimizer_trials.csv", index=False)

    shutil.copy(RUN_DIR / "finalized_selections.json", mock_dir / "finalized_selections.json")
    shutil.copy(RUN_DIR / "final_evaluations.csv", mock_dir / "final_evaluations.csv")

    with pytest.raises(ScientificConsistencyError, match="Holdout leakage detected"):
        audit_test_isolation_and_integrity(mock_dir)


def test_audit_detects_hash_mismatch(tmp_path):
    """Verify that audit raises ScientificConsistencyError if selection hashes are corrupted."""
    if not RUN_DIR.exists():
        pytest.skip("Full benchmark run directory not present")

    mock_dir = tmp_path / "mock_run"
    mock_dir.mkdir()
    shutil.copy(RUN_DIR / "optimizer_trials.csv", mock_dir / "optimizer_trials.csv")
    shutil.copy(RUN_DIR / "finalized_selections.json", mock_dir / "finalized_selections.json")

    # Corrupt hash in final evaluations
    evals = pd.read_csv(RUN_DIR / "final_evaluations.csv")
    evals.loc[0, "config_sha256"] = "corrupted_hash_00000000000000000000"
    evals.to_csv(mock_dir / "final_evaluations.csv", index=False)

    with pytest.raises(ScientificConsistencyError, match="Configuration hash mismatch"):
        audit_test_isolation_and_integrity(mock_dir)


def test_audit_detects_hypervolume_discrepancy(tmp_path):
    """Verify that audit raises ScientificConsistencyError if hypervolume numbers diverge."""
    if not RUN_DIR.exists():
        pytest.skip("Full benchmark run directory not present")

    mock_dir = tmp_path / "mock_run"
    mock_dir.mkdir()
    with open(RUN_DIR / "hypervolume.json", encoding="utf-8") as f:
        hv_data = json.load(f)

    # Artificially alter MO-TPE mean
    hv_data["development_domain"]["reference_points"]["[0.6, 250.0]"][
        "mo_tpe_hypervolume_distribution"
    ]["mean"] = 25.0000

    with open(mock_dir / "hypervolume.json", "w", encoding="utf-8") as f:
        json.dump(hv_data, f)
    shutil.copy(RUN_DIR / "doe_matched_candidate_front.csv", mock_dir / "doe_matched_candidate_front.csv")

    with pytest.raises(ScientificConsistencyError, match="MO-TPE mean hypervolume.*mismatch"):
        audit_hypervolume_consistency(mock_dir)
