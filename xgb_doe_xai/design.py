"""xgb_doe_xai.design - Experimental design matrix generation (FCCD) and adaptive infill augmentation.

Implements Face-Centered Central Composite Designs (FCCD) and sequential
adaptive sampling criteria (uncertainty, Pareto improvement, and surrogate discrepancy).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from xgb_doe_xai.config import canonical_configuration


@dataclass(frozen=True)
class DesignPoint:
    point_id: str
    point_type: str  # 'factorial', 'axial', 'center', 'adaptive'
    coded: Dict[str, float]
    block_id: int

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.coded["x1"], self.coded["x2"], self.coded["x3"], self.coded["x4"]])


def generate_fccd_points(
    factors: List[str] = ("x1", "x2", "x3", "x4"),
    n_center: int = 4,
    block_id: int = 0,
) -> List[DesignPoint]:
    """Generate Face-Centered Central Composite Design (FCCD, alpha=1.0) points for k factors."""
    k = len(factors)
    points: List[DesignPoint] = []
    pt_idx = 1

    # 1. 2^k Factorial corner points (+1, -1)
    for combo in itertools.product([-1.0, 1.0], repeat=k):
        coded_dict = {f: float(val) for f, val in zip(factors, combo)}
        points.append(
            DesignPoint(
                point_id=f"B{block_id}_F{pt_idx:02d}",
                point_type="factorial",
                coded=coded_dict,
                block_id=block_id,
            )
        )
        pt_idx += 1

    # 2. 2*k Axial / star points (alpha = 1.0 for Face-Centered)
    for i, factor in enumerate(factors):
        for sign in [-1.0, 1.0]:
            coded_dict = {f: 0.0 for f in factors}
            coded_dict[factor] = sign
            points.append(
                DesignPoint(
                    point_id=f"B{block_id}_A{pt_idx:02d}",
                    point_type="axial",
                    coded=coded_dict,
                    block_id=block_id,
                )
            )
            pt_idx += 1

    # 3. n_center center points (0, 0, ..., 0)
    for c in range(n_center):
        coded_dict = {f: 0.0 for f in factors}
        points.append(
            DesignPoint(
                point_id=f"B{block_id}_C{c+1:02d}",
                point_type="center",
                coded=coded_dict,
                block_id=block_id,
            )
        )

    return points


def generate_multiblock_fccd(
    n_blocks: int = 5,
    factors: List[str] = ("x1", "x2", "x3", "x4"),
    n_center_per_block: int = 4,
) -> List[DesignPoint]:
    """Generate blocked FCCD points replicated across n_blocks nuisance partitions."""
    all_points: List[DesignPoint] = []
    for b in range(n_blocks):
        b_pts = generate_fccd_points(factors=factors, n_center=n_center_per_block, block_id=b)
        all_points.extend(b_pts)
    return all_points


def select_adaptive_infill_points(
    existing_points: List[DesignPoint],
    surrogate_uncertainty_fn: Callable[[np.ndarray], np.ndarray],
    n_infill: int = 4,
    n_candidates: int = 2000,
    random_seed: int = 42,
    criterion: str = "uncertainty",
    surrogate_discrepancy_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    surrogate_mean_fn=None,
    best_observed=None,
    cycle_id: int = 1,
) -> List[DesignPoint]:
    """Select new evaluation coordinates using active learning / sequential DoE criteria.

    Parameters:
    - criterion: 'uncertainty' (max surrogate variance),
                 'discrepancy' (max divergence between polynomial and non-parametric model),
                 'hybrid' (combination of uncertainty and space-filling).
    """
    if criterion not in ('uncertainty', 'discrepancy', 'hybrid', 'expected_improvement'):
        raise ValueError('Unknown acquisition criterion')
    if n_infill < 1 or n_candidates < n_infill or not existing_points:
        raise ValueError('Invalid infill budget or empty existing design')
    rng = np.random.RandomState(random_seed)
    candidates = rng.uniform(-1.0, 1.0, size=(n_candidates, 4))
    candidates = np.array([list(canonical_configuration(dict(zip(('x1','x2','x3','x4'), p))).values())
                           for p in candidates])
    candidates = np.unique(candidates, axis=0)

    # Compute Euclidean distance to nearest existing point to avoid trivial duplication
    existing_mat = np.array([pt.vector for pt in existing_points])

    if criterion == "discrepancy":
        if surrogate_discrepancy_fn is None:
            raise ValueError('Discrepancy criterion needs discrepancy predictions')
        scores = surrogate_discrepancy_fn(candidates)
    elif criterion == 'expected_improvement':
        from scipy.stats import norm
        if surrogate_mean_fn is None or best_observed is None:
            raise ValueError('Expected improvement needs means and best_observed')
        sd = np.asarray(surrogate_uncertainty_fn(candidates))
        improvement = best_observed - surrogate_mean_fn(candidates)
        z = improvement / np.maximum(sd, 1e-15)
        scores = np.where(sd > 1e-15, improvement*norm.cdf(z)+sd*norm.pdf(z),
                          np.maximum(improvement, 0))
    else:
        scores = surrogate_uncertainty_fn(candidates)
    scores = np.asarray(scores)
    if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
        raise ValueError('Acquisition scores must be finite and match candidates')

    # Penalize candidates that are too close to existing points (minimum distance threshold)
    selected_indices: List[int] = []
    pool = list(range(len(candidates)))

    for _ in range(n_infill):
        best_idx = -1
        best_score = -1e9
        for idx in pool:
            cand = candidates[idx]
            dists = np.linalg.norm(existing_mat - cand, axis=1)
            min_dist = np.min(dists)
            if min_dist < 0.15:  # skip immediate neighbors
                continue
            combined_score = scores[idx] * (min_dist ** 0.5) if criterion == 'hybrid' else scores[idx]
            if combined_score > best_score:
                best_score = combined_score
                best_idx = idx

        if best_idx == -1:
            raise ValueError('Not enough distinct feasible acquisition candidates')

        selected_indices.append(best_idx)
        pool.remove(best_idx)
        existing_mat = np.vstack([existing_mat, candidates[best_idx]])

    factors = ["x1", "x2", "x3", "x4"]
    infill_points: List[DesignPoint] = []
    for k, idx in enumerate(selected_indices, start=1):
        cand = candidates[idx]
        coded_dict = {f: float(cand[i]) for i, f in enumerate(factors)}
        infill_points.append(
            DesignPoint(
                point_id=f"C{cycle_id}_ADAPT_{k:02d}",
                point_type="adaptive",
                coded=coded_dict,
                block_id=0,
            )
        )

    return infill_points
