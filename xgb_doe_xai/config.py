"""xgb_doe_xai.config - Hyperparameter domain, coding transformations, and benchmark constraints.

Addresses Remark 1:
Learning rate excludes the historical eta=0.01 setting. Convergence on a new
dataset must still be calibrated using development observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import numpy as np


@dataclass(frozen=True)
class FactorSpec:
    name: str
    code_name: str
    low: float
    center: float
    high: float
    is_integer: bool = False
    log_scale: bool = False

    def to_coded(self, natural_value: float) -> float:
        """Transform natural coordinate to [-1, +1] coded space."""
        low, center, high = (np.log([self.low, self.center, self.high])
                             if self.log_scale else (self.low, self.center, self.high))
        value = np.log(natural_value) if self.log_scale else natural_value
        return (value - center) / ((high - low) / 2.0)

    def to_natural(self, coded_value: float) -> float | int:
        """Transform [-1, +1] coded coordinate to natural space."""
        low, center, high = (np.log([self.low, self.center, self.high])
                             if self.log_scale else (self.low, self.center, self.high))
        nat = center + coded_value * (high - low) / 2.0
        if self.log_scale:
            nat = np.exp(nat)
        nat = np.clip(nat, self.low, self.high)
        if self.is_integer:
            return int(round(nat))
        return float(nat)


# Canonical 4-factor domain for XGBoost Sentiment Analysis
DEFAULT_FACTORS: Dict[str, FactorSpec] = {
    "x1": FactorSpec(
        name="learning_rate",
        code_name="x1",
        low=0.05,
        center=0.175,
        high=0.30,
        is_integer=False,
    ),
    "x2": FactorSpec(
        name="max_depth",
        code_name="x2",
        low=3.0,
        center=6.0,
        high=9.0,
        is_integer=True,
    ),
    "x3": FactorSpec(
        name="subsample",
        code_name="x3",
        low=0.50,
        center=0.75,
        high=1.00,
        is_integer=False,
    ),
    "x4": FactorSpec(
        name="reg_lambda",
        code_name="x4",
        low=0.10,
        center=1.00,
        high=10.00,
        is_integer=False,
        log_scale=True,
    ),
}

FACTOR_NAMES: List[str] = ["x1", "x2", "x3", "x4"]
FACTOR_LABELS: Dict[str, str] = {
    "x1": "Learning Rate (eta)",
    "x2": "Max Depth",
    "x3": "Subsample Ratio",
    "x4": "Leaf-weight Regularization (lambda)",
}


@dataclass
class BenchmarkConfig:
    latency_threshold_us: float = 145.0
    warmup_inferences: int = 100
    timed_inferences: int = 500
    n_trees: int = 100
    holdout_ratio: float = 0.20
    n_nuisance_blocks: int = 5
    hv_ref_loss: float = 0.50
    hv_ref_latency_us: float = 250.0
    random_seed: int = 42
    center_replicates: int = 4
    adaptive_cycles: int = 2
    infill_per_cycle: int = 10
    confirmation_points: int = 12
    search_replicates: int = 3
    acquisition: str = 'hybrid'
    mode: str = 'research'
    include_static_baseline: bool = False

    def __post_init__(self):
        if self.mode not in ('research', 'smoke'):
            raise ValueError('mode must be research or smoke')
        if self.mode == 'research' and (self.warmup_inferences, self.timed_inferences) != (100, 500):
            raise ValueError('Research latency protocol requires 100 warmup and 500 timed calls')
        if min(self.n_trees, self.n_nuisance_blocks, self.search_replicates,
               self.center_replicates, self.confirmation_points, self.infill_per_cycle) < 1:
            raise ValueError('Budgets must be positive')
        if self.adaptive_cycles < 1:
            raise ValueError('Sequential research requires at least one adaptive cycle')
        if self.n_nuisance_blocks < 2 or self.center_replicates < 2:
            raise ValueError('Blocked studies need at least two blocks and two center repeats')
        if min(self.latency_threshold_us, self.hv_ref_loss, self.hv_ref_latency_us) <= 0:
            raise ValueError('Latency limits and hypervolume reference coordinates must be positive')
        if self.acquisition not in ('uncertainty', 'discrepancy', 'hybrid', 'expected_improvement'):
            raise ValueError('Unknown acquisition criterion')


def code_configuration(natural_params: Dict[str, float]) -> Dict[str, float]:
    """Transform dictionary of natural hyperparameter values to coded coordinates."""
    coded: Dict[str, float] = {}
    for code_k, factor in DEFAULT_FACTORS.items():
        nat_val = natural_params.get(factor.name, natural_params.get(code_k))
        if nat_val is None:
            raise KeyError(f"Missing parameter '{factor.name}' (or '{code_k}')")
        coded[code_k] = factor.to_coded(float(nat_val))
    return coded


def uncode_configuration(coded_params: Dict[str, float]) -> Dict[str, float | int]:
    """Transform dictionary of coded coordinates to natural hyperparameter values."""
    natural: Dict[str, float | int] = {}
    for code_k, factor in DEFAULT_FACTORS.items():
        if code_k not in coded_params:
            raise KeyError(f"Missing coded factor '{code_k}'")
        natural[factor.name] = factor.to_natural(coded_params[code_k])
    return natural


def canonical_configuration(coded_params):
    """Coordinates of the configuration actually trained, including integer depth."""
    values = np.array([coded_params[name] for name in FACTOR_NAMES], dtype=float)
    if not np.isfinite(values).all() or np.any(np.abs(values) > 1 + 1e-12):
        raise ValueError('Coded coordinates must lie in [-1, 1]')
    return {name: round(value, 12) for name, value in
            code_configuration(uncode_configuration(coded_params)).items()}
