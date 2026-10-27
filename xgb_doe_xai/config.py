"""xgb_doe_xai.config - Hyperparameter domain, coding transformations, and benchmark constraints.

Addresses Remark 1:
Learning rate range is calibrated to [0.05, 0.30] (avoiding the 32% underfitting
at eta=0.01 observed in previous iterations).
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

    def to_coded(self, natural_value: float) -> float:
        """Transform natural coordinate to [-1, +1] coded space."""
        half_range = (self.high - self.low) / 2.0
        return (natural_value - self.center) / half_range

    def to_natural(self, coded_value: float) -> float | int:
        """Transform [-1, +1] coded coordinate to natural space."""
        half_range = (self.high - self.low) / 2.0
        nat = self.center + coded_value * half_range
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
        name="colsample_bytree",
        code_name="x4",
        low=0.30,
        center=0.65,
        high=1.00,
        is_integer=False,
    ),
}

FACTOR_NAMES: List[str] = ["x1", "x2", "x3", "x4"]
FACTOR_LABELS: Dict[str, str] = {
    "x1": "Learning Rate (eta)",
    "x2": "Max Depth",
    "x3": "Subsample Ratio",
    "x4": "Feature Subsample (colsample_bytree)",
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
