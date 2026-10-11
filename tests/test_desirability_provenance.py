import numpy as np


def test_historical_desirability_grid_has_exact_stated_size_and_levels():
    from desirability_provenance import historical_stated_grid

    grid = historical_stated_grid()
    assert grid.shape == (588, 4)
    assert len(np.unique(grid[:, 0])) == 21
    assert len(np.unique(grid[:, 1])) == 7
    assert set(grid[:, 2]) == {0.0, 1.0}
    assert set(grid[:, 3]) == {0.0, 1.0}


def test_historical_selected_coordinate_is_not_on_stated_grid():
    from desirability_provenance import coordinate_membership, historical_stated_grid

    selected = [0.8499708, -0.66666667, 1.0, -0.08116946]
    result = coordinate_membership(selected, historical_stated_grid())
    assert result["is_member"] is False
    assert result["maximum_absolute_coordinate_difference"] > 0.08


def test_selection_audit_returns_actual_grid_maximum_with_tie_rule():
    from desirability_provenance import selection_audit

    candidates = [[-1, -1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]
    result = selection_audit([0, 0, 0, 0], candidates, [0.2, 0.8, 0.8])
    assert result["membership"]["is_member"] is True
    assert result["mathematical_grid_maximum"]["candidate_index"] == 1
    assert result["mathematical_grid_maximum"]["tie_count"] == 2
    assert result["selected_coordinate_classification"] == "grid_candidate"
