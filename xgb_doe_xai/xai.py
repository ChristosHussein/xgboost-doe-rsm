"""xgb_doe_xai.xai - Methodological framework bridging 2nd-order ANOVA and SHAP Interactions.

Addresses Idea 1:
Connects ANOVA (main effects, interaction terms, curvature) with SHAP
(TreeExplainer / KernelExplainer attributions and interaction values)
at the surrogate model level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from math import factorial


@dataclass(frozen=True)
class AnovaAttributions:
    factor_names: List[str]
    main_effect_ss: Dict[str, float]
    main_effect_ranking: List[str]
    interaction_matrix_ss: np.ndarray  # k x k symmetric matrix
    curvature_ss: Dict[str, float]


@dataclass(frozen=True)
class ShapAttributions:
    factor_names: List[str]
    mean_abs_shap: Dict[str, float]
    shap_ranking: List[str]
    interaction_matrix_shap: np.ndarray  # k x k symmetric matrix
    raw_shap_values: np.ndarray
    raw_interaction_values: Optional[np.ndarray] = None


@dataclass(frozen=True)
class AnovaShapComparisonResult:
    spearman_rho: Optional[float]
    kendall_tau: Optional[float]
    interaction_correlation: float
    rank_agreement: bool
    top_factor_anova: str
    top_factor_shap: str
    top_interaction_anova: Tuple[str, str]
    top_interaction_shap: Tuple[str, str]
    discrepancy_scores: Dict[str, float]


def extract_anova_attributions(
    rsm_surrogate: Any,
    factor_names: Optional[List[str]] = None,
) -> AnovaAttributions:
    """Extract factor-specific main effect SS, interaction matrix, and curvature SS from ANOVA."""
    if factor_names is None:
        factor_names = ["x1", "x2", "x3", "x4"]
    k = len(factor_names)

    # 1. Main effects: beta_i^2 / Var(beta_i) proxy or individual factor SS
    main_ss: Dict[str, float] = {}
    p_names = rsm_surrogate.feature_names
    beta = rsm_surrogate.beta_

    for i, f in enumerate(factor_names):
        if f in p_names:
            idx = p_names.index(f)
            # Semi-partial sum of squares
            c_ii = rsm_surrogate.xtx_inv_[idx, idx]
            ss_f = float((beta[idx] ** 2) / max(c_ii, 1e-12))
            main_ss[f] = ss_f
        else:
            main_ss[f] = 0.0

    # 2. Curvature SS: beta_ii^2 / Var(beta_ii)
    curv_ss: Dict[str, float] = {}
    for f in factor_names:
        name_q = f"{f}^2"
        if name_q in p_names:
            idx = p_names.index(name_q)
            c_ii = rsm_surrogate.xtx_inv_[idx, idx]
            curv_ss[f] = float((beta[idx] ** 2) / max(c_ii, 1e-12))
        else:
            curv_ss[f] = 0.0

    # 3. 2-factor interaction matrix (k x k)
    inter_mat = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            pair_name = f"{factor_names[i]}:{factor_names[j]}"
            if pair_name in p_names:
                idx = p_names.index(pair_name)
                c_ii = rsm_surrogate.xtx_inv_[idx, idx]
                ss_ij = float((beta[idx] ** 2) / max(c_ii, 1e-12))
                inter_mat[i, j] = ss_ij
                inter_mat[j, i] = ss_ij

    # Rank factors by main effect SS descending
    ranking = sorted(factor_names, key=lambda f: main_ss[f], reverse=True)

    return AnovaAttributions(
        factor_names=factor_names,
        main_effect_ss=main_ss,
        main_effect_ranking=ranking,
        interaction_matrix_ss=inter_mat,
        curvature_ss=curv_ss,
    )


def compute_surrogate_shap_attributions(
    surrogate_model: Any,
    background_data: np.ndarray,
    eval_data: np.ndarray,
    factor_names: Optional[List[str]] = None,
) -> ShapAttributions:
    """Compute SHAP values and SHAP interaction matrices on the surrogate model predicting response."""
    if factor_names is None:
        factor_names = ["x1", "x2", "x3", "x4"]
    k = len(factor_names)

    # Exact interventional coalition expectations over the declared empirical
    # background. Four factors require only 16 coalitions per explained point.
    background_data = np.asarray(background_data, dtype=float)
    eval_data = np.asarray(eval_data, dtype=float)
    if (background_data.ndim != 2 or eval_data.ndim != 2
            or background_data.shape[1] != k or eval_data.shape[1] != k
            or len(background_data) == 0 or len(eval_data) == 0
            or not np.isfinite(background_data).all() or not np.isfinite(eval_data).all()):
        raise ValueError('SHAP requires finite nonempty background/evaluation matrices')
    values = np.empty((len(eval_data), 1 << k))
    for row, point in enumerate(eval_data):
        for mask in range(1 << k):
            hybrid = background_data.copy()
            present = [i for i in range(k) if mask & (1 << i)]
            hybrid[:, present] = point[present]
            prediction = np.asarray(surrogate_model.predict(hybrid)).reshape(-1)
            if prediction.shape != (len(hybrid),) or not np.isfinite(prediction).all():
                raise ValueError('Surrogate must return one finite response per row')
            values[row, mask] = prediction.mean()
    shap_arr = np.zeros((len(eval_data), k))
    interactions = np.zeros((len(eval_data), k, k))
    for i in range(k):
        for mask in range(1 << k):
            if mask & (1 << i):
                continue
            size = mask.bit_count()
            weight = factorial(size) * factorial(k-size-1) / factorial(k)
            shap_arr[:, i] += weight * (values[:, mask | (1 << i)] - values[:, mask])
        for j in range(i + 1, k):
            for mask in range(1 << k):
                if mask & ((1 << i) | (1 << j)):
                    continue
                size = mask.bit_count()
                weight = factorial(size) * factorial(k-size-2) / (2*factorial(k-1))
                delta = (values[:, mask | (1 << i) | (1 << j)]
                         - values[:, mask | (1 << i)]
                         - values[:, mask | (1 << j)] + values[:, mask])
                interactions[:, i, j] += weight * delta
            interactions[:, j, i] = interactions[:, i, j]
    # Diagonals retain each feature's main effect after removing pair effects.
    for i in range(k):
        interactions[:, i, i] = 0
        interactions[:, i, i] = shap_arr[:, i] - interactions[:, i, :].sum(axis=1)

    # Mean absolute SHAP per factor
    mean_abs: Dict[str, float] = {}
    for i, f in enumerate(factor_names):
        mean_abs[f] = float(np.mean(np.abs(shap_arr[:, i])))

    ranking = sorted(factor_names, key=lambda f: mean_abs[f], reverse=True)

    inter_mat = np.mean(np.abs(interactions), axis=0)

    return ShapAttributions(
        factor_names=factor_names,
        mean_abs_shap=mean_abs,
        shap_ranking=ranking,
        interaction_matrix_shap=inter_mat,
        raw_shap_values=shap_arr,
        raw_interaction_values=interactions,
    )


def compare_anova_vs_shap(
    anova: AnovaAttributions,
    shap_attr: ShapAttributions,
) -> AnovaShapComparisonResult:
    """Rigorous statistical comparison between ANOVA and SHAP representations."""
    factors = anova.factor_names
    k = len(factors)

    # 1. Main effects rank correlation
    anova_scores = np.array([anova.main_effect_ss[f] for f in factors])
    shap_scores = np.array([shap_attr.mean_abs_shap[f] for f in factors])

    rho, _ = stats.spearmanr(anova_scores, shap_scores)
    tau, _ = stats.kendalltau(anova_scores, shap_scores)

    # 2. Top factors
    top_f_anova = anova.main_effect_ranking[0]
    top_f_shap = shap_attr.shap_ranking[0]
    rank_agree = (anova.main_effect_ranking == shap_attr.shap_ranking)

    # 3. Off-diagonal interaction correlation
    triu_idx = np.triu_indices(k, k=1)
    anova_inter_vec = anova.interaction_matrix_ss[triu_idx]
    shap_inter_vec = shap_attr.interaction_matrix_shap[triu_idx]

    if np.std(anova_inter_vec) > 1e-12 and np.std(shap_inter_vec) > 1e-12:
        inter_corr, _ = stats.pearsonr(anova_inter_vec, shap_inter_vec)
    else:
        inter_corr = 1.0 if np.allclose(anova_inter_vec, shap_inter_vec) else 0.0

    # Top interaction pair
    max_anova_idx = int(np.argmax(anova_inter_vec))
    i_a, j_a = triu_idx[0][max_anova_idx], triu_idx[1][max_anova_idx]
    top_pair_anova = (factors[i_a], factors[j_a])

    max_shap_idx = int(np.argmax(shap_inter_vec))
    i_s, j_s = triu_idx[0][max_shap_idx], triu_idx[1][max_shap_idx]
    top_pair_shap = (factors[i_s], factors[j_s])

    # 4. Discrepancy score per factor: absolute difference between normalized ranks
    anova_norm = anova_scores / max(np.sum(anova_scores), 1e-12)
    shap_norm = shap_scores / max(np.sum(shap_scores), 1e-12)
    discrepancy = {f: float(abs(a - s)) for f, a, s in zip(factors, anova_norm, shap_norm)}

    return AnovaShapComparisonResult(
        spearman_rho=float(rho) if np.isfinite(rho) else None,
        kendall_tau=float(tau) if np.isfinite(tau) else None,
        interaction_correlation=float(inter_corr),
        rank_agreement=rank_agree,
        top_factor_anova=top_f_anova,
        top_factor_shap=top_f_shap,
        top_interaction_anova=top_pair_anova,
        top_interaction_shap=top_pair_shap,
        discrepancy_scores=discrepancy,
    )
