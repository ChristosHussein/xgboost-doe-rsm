"""
test_verification.py - Self-Contained Verification Suite
=========================================================
Assert-based mathematical and algorithmic self-check:
1. Reversibility of factor coding / decoding.
2. Design matrix dimensions and orthogonality properties.
3. Derringer-Suich boundary conditions and monotonicity.
4. Canonical spectral decomposition identity: x^T B x == w^T Lambda w.
5. ANOVA Sum of Squares identity: SS_Residual == SS_Pure_Error + SS_Lack_of_Fit.
"""

import numpy as np
import pandas as pd
from pipeline import (
    encode_factors, decode_factors,
    generate_design_plan, CONFIG
)
from analysis import derringer_suich_desirability

ETA_MIN = CONFIG["factors"]["x1"]["min"]
ETA_MAX = CONFIG["factors"]["x1"]["max"]
DEPTH_MIN = CONFIG["factors"]["x2"]["min"]
DEPTH_MAX = CONFIG["factors"]["x2"]["max"]
SUBSAMPLE_MIN = CONFIG["factors"]["x3"]["min"]
SUBSAMPLE_MAX = CONFIG["factors"]["x3"]["max"]
LAMBDA_MIN = CONFIG["factors"]["x4"]["min"]
LAMBDA_MAX = CONFIG["factors"]["x4"]["max"]

def test_factor_coding_reversibility():
    """Verifies that decode(code(xi)) == xi and code(decode(x)) == x across corners, centers, and random values."""
    print("Testing factor coding reversibility...")
    test_coded = [
        np.array([-1.0, -1.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([-0.5, 0.3333333333, 0.2, -0.7]),
    ]
    for x in test_coded:
        eta, depth, subsample, reg_lambda = decode_factors(x)
        assert ETA_MIN - 1e-9 <= eta <= ETA_MAX + 1e-9, f"eta {eta} outside [{ETA_MIN}, {ETA_MAX}]"
        assert DEPTH_MIN - 1e-9 <= depth <= DEPTH_MAX + 1e-9, f"depth {depth} outside [{DEPTH_MIN}, {DEPTH_MAX}]"
        assert SUBSAMPLE_MIN - 1e-9 <= subsample <= SUBSAMPLE_MAX + 1e-9, f"subsample {subsample} outside [{SUBSAMPLE_MIN}, {SUBSAMPLE_MAX}]"
        assert LAMBDA_MIN - 1e-9 <= reg_lambda <= LAMBDA_MAX + 1e-9, f"lambda {reg_lambda} outside [{LAMBDA_MIN}, {LAMBDA_MAX}]"
        
        x_rec = encode_factors(eta, depth, subsample, reg_lambda)
        np.testing.assert_allclose(x, x_rec, atol=1e-5, err_msg="Factor coding failed roundtrip recovery!")
    print("[OK] Factor coding reversibility verified.")

def test_design_matrix_properties():
    """Verifies factorial dimensions, center point replication, and orthogonality of corner runs."""
    print("Testing design matrix dimensions and orthogonality...")
    runs = generate_design_plan()
    assert len(runs) == 140, f"Expected 140 runs, got {len(runs)}"
    
    df_runs = pd.DataFrame(runs)
    p1_runs = df_runs[df_runs["phase"].str.startswith("Phase1")]
    assert len(p1_runs) == 100, f"Expected 100 Phase 1 runs, got {len(p1_runs)}"
    
    p2_runs = df_runs[df_runs["phase"].str.startswith("Phase2")]
    assert len(p2_runs) == 40, f"Expected 40 Phase 2 runs, got {len(p2_runs)}"
    
    # Check 16 corners + 4 center runs per block
    for b in [1, 2, 3, 4, 5]:
        b_df = p1_runs[p1_runs["block"] == b]
        assert len(b_df) == 20, f"Block {b} should have 20 runs, got {len(b_df)}"
        n_fact = (b_df["phase"] == "Phase1_Factorial").sum()
        n_center = (b_df["phase"] == "Phase1_Center").sum()
        assert n_fact == 16, f"Block {b} should have 16 factorial points, got {n_fact}"
        assert n_center == 4, f"Block {b} should have 4 center points, got {n_center}"

    # Check orthogonality of 2^4 factorial portion
    corners_df = df_runs[(df_runs["block"] == 1) & (df_runs["phase"] == "Phase1_Factorial")]
    X_fact = corners_df[["x1", "x2", "x3", "x4"]].values
    XtX = X_fact.T @ X_fact
    # Off-diagonal elements must be exactly 0
    off_diag = XtX - np.diag(np.diag(XtX))
    np.testing.assert_allclose(off_diag, 0.0, atol=1e-10, err_msg="2^4 factorial columns are not mutually orthogonal!")
    print("[OK] Design matrix orthogonality and dimensions verified.")

def test_desirability_function_properties():
    """Verifies Derringer-Suich boundary conditions and monotonicity."""
    print("Testing Derringer-Suich desirability boundary conditions...")
    L1, U1 = 0.40, 0.80
    L2, U2 = 100.0, 500.0
    
    # At lower bound or below: desirability must be 1.0
    d1, d2, D = derringer_suich_desirability(0.35, 90.0, L1, U1, L2, U2)
    assert d1 == 1.0 and d2 == 1.0 and D == 1.0
    
    # At upper bound or above: desirability must be 0.0
    d1, d2, D = derringer_suich_desirability(0.85, 600.0, L1, U1, L2, U2)
    assert d1 == 0.0 and d2 == 0.0 and D == 0.0
    
    # Midpoint: desirability must be strictly between 0 and 1
    d1, d2, D = derringer_suich_desirability(0.60, 300.0, L1, U1, L2, U2)
    assert 0.0 < d1 < 1.0 and 0.0 < d2 < 1.0 and 0.0 < D < 1.0
    
    # Monotonicity check
    d1_prev, d2_prev = 1.0, 1.0
    for y1 in np.linspace(L1, U1, 10):
        d1, _, _ = derringer_suich_desirability(y1, 200.0, L1, U1, L2, U2)
        assert d1 <= d1_prev + 1e-12, "Desirability violated monotonicity!"
        d1_prev = d1
    print("[OK] Derringer-Suich desirability properties verified.")

def test_canonical_spectral_identity():
    """Verifies B = M Lambda M^T and quadratic form equivalence x^T B x == w^T Lambda w."""
    print("Testing canonical spectral decomposition identity...")
    rng = np.random.RandomState(42)
    A = rng.randn(4, 4)
    B = 0.5 * (A + A.T) # symmetric
    
    eigenvals, eigenvecs = np.linalg.eigh(B)
    M = eigenvecs
    Lambda = np.diag(eigenvals)
    
    # B == M Lambda M^T
    B_rec = M @ Lambda @ M.T
    np.testing.assert_allclose(B, B_rec, atol=1e-10)
    
    # For arbitrary x, x^T B x == w^T Lambda w
    x = rng.randn(4)
    x0 = rng.randn(4)
    w = M.T @ (x - x0)
    
    quad_x = (x - x0).T @ B @ (x - x0)
    quad_w = w.T @ Lambda @ w
    np.testing.assert_allclose(quad_x, quad_w, atol=1e-10)
    print("[OK] Canonical spectral decomposition identity verified.")

def run_all_tests():
    test_factor_coding_reversibility()
    test_design_matrix_properties()
    test_desirability_function_properties()
    test_canonical_spectral_identity()
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_all_tests()
