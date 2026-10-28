"""Deterministic provenance helpers for desirability-based selections."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def historical_stated_grid() -> np.ndarray:
    """Return the 21 x 7 x 2 x 2 grid implemented by the v1 sensitivity code."""
    points = []
    for x1 in np.linspace(-1.0, 1.0, 21):
        for x3 in (0.0, 1.0):
            for x4 in (0.0, 1.0):
                for depth in range(3, 10):
                    x2 = (depth - 6.0) / 3.0
                    points.append((x1, x2, x3, x4))
    return np.asarray(points, dtype=float)


def coordinate_membership(
    coordinate: Sequence[float],
    candidates: Iterable[Sequence[float]],
    *,
    absolute_tolerance: float = 1e-9,
) -> dict:
    selected = np.asarray(coordinate, dtype=float)
    grid = np.asarray(list(candidates), dtype=float)
    if selected.shape != (4,) or grid.ndim != 2 or grid.shape[1] != 4:
        raise ValueError("coordinate and candidate grid must have four coded factors")
    distances = np.max(np.abs(grid - selected), axis=1)
    nearest_index = int(np.argmin(distances))
    return {
        "is_member": bool(distances[nearest_index] <= absolute_tolerance),
        "nearest_candidate_index": nearest_index,
        "nearest_candidate": grid[nearest_index].tolist(),
        "maximum_absolute_coordinate_difference": float(distances[nearest_index]),
        "absolute_tolerance": float(absolute_tolerance),
    }


def selection_audit(
    coordinate: Sequence[float],
    candidates: Iterable[Sequence[float]],
    desirabilities: Sequence[float],
) -> dict:
    grid = np.asarray(list(candidates), dtype=float)
    values = np.asarray(desirabilities, dtype=float)
    if len(grid) != len(values) or len(grid) == 0:
        raise ValueError("one desirability value is required for every candidate")
    if not np.isfinite(values).all():
        raise ValueError("desirabilities must be finite")
    membership = coordinate_membership(coordinate, grid)
    maximum = float(np.max(values))
    maximizers = np.flatnonzero(np.isclose(values, maximum, rtol=0.0, atol=1e-15))
    winner_index = int(maximizers[0])
    return {
        "candidate_count": int(len(grid)),
        "membership": membership,
        "mathematical_grid_maximum": {
            "candidate_index": winner_index,
            "coordinate": grid[winner_index].tolist(),
            "desirability": maximum,
            "tie_count": int(len(maximizers)),
            "tie_breaking_rule": "lowest deterministic enumeration index",
        },
        "selected_coordinate_classification": (
            "grid_candidate"
            if membership["is_member"]
            else "post_search_coordinate_not_generated_by_stated_grid"
        ),
    }
