import numpy as np
import pytest

from scientific_stats import (
    blocked_lack_of_fit_decomposition,
    hypervolume_2d_min,
    paired_difference_summary,
    paired_tost,
    pareto_front_2d_min,
)


def test_hypervolume_empty_and_single_point():
    empty = hypervolume_2d_min([], (3.0, 5.0))
    single = hypervolume_2d_min([(1.0, 2.0)], (3.0, 5.0))

    assert empty.value == 0.0
    assert empty.pareto.points == ()
    assert single.value == pytest.approx(6.0)


def test_hypervolume_two_points_is_union_of_rectangles():
    result = hypervolume_2d_min([(1.0, 4.0), (2.0, 2.0)], (3.0, 5.0))

    # Areas 2 and 3 overlap by 1, so the union is 4 rather than 5.
    assert result.value == pytest.approx(4.0)
    assert result.pareto.points == ((1.0, 4.0), (2.0, 2.0))


def test_pareto_filter_reports_duplicates_dominated_and_outside_points():
    points = [
        (2.5, 4.5),  # dominated by (1, 4)
        (2.0, 2.0),
        (1.0, 4.0),
        (1.0, 4.0),  # duplicate
        (4.0, 1.0),  # outside reference in objective 1
        (1.0, 6.0),  # outside reference in objective 2
    ]

    result = pareto_front_2d_min(points, reference=(3.0, 5.0))

    assert result.points == ((1.0, 4.0), (2.0, 2.0))
    assert result.input_count == 6
    assert result.unique_count == 5
    assert result.duplicate_count == 1
    assert result.outside_reference_count == 2
    assert result.dominated_count == 1


def test_hypervolume_is_input_order_invariant_and_reference_sensitive():
    points = [(2.0, 2.0), (1.0, 4.0)]

    assert hypervolume_2d_min(points, (3.0, 5.0)).value == pytest.approx(4.0)
    assert hypervolume_2d_min(list(reversed(points)), (3.0, 5.0)).value == pytest.approx(4.0)
    assert hypervolume_2d_min(points, (4.0, 6.0)).value == pytest.approx(10.0)


@pytest.mark.parametrize(
    "points",
    [[(np.nan, 1.0)], [(1.0, np.inf)], [(1.0,)]],
)
def test_pareto_filter_rejects_invalid_points(points):
    with pytest.raises(ValueError):
        pareto_front_2d_min(points)


def test_paired_difference_summary_has_explicit_a_minus_b_sign_and_t_interval():
    # Differences are [-1, 0, 1], giving mean=0, sample SD=1, df=2.
    result = paired_difference_summary([0.0, 2.0, 4.0], [1.0, 2.0, 3.0])

    assert result["difference"] == "a_minus_b"
    assert result["n_pairs"] == 3
    assert result["mean_difference"] == pytest.approx(0.0)
    assert result["sd_difference"] == pytest.approx(1.0)
    assert result["ci_low"] == pytest.approx(-2.484137712, rel=1e-9)
    assert result["ci_high"] == pytest.approx(2.484137712, rel=1e-9)
    assert result["paired_t_pvalue"] == pytest.approx(1.0)
    assert result["cohens_dz"] == pytest.approx(0.0)


def test_paired_difference_sign_reverses_with_operand_order():
    ab = paired_difference_summary([2.0, 4.0, 7.0], [1.0, 3.0, 5.0])
    ba = paired_difference_summary([1.0, 3.0, 5.0], [2.0, 4.0, 7.0])

    assert ab["mean_difference"] == pytest.approx(-ba["mean_difference"])
    assert ab["ci_low"] == pytest.approx(-ba["ci_high"])
    assert ab["ci_high"] == pytest.approx(-ba["ci_low"])


def test_paired_difference_all_zero_is_stable():
    result = paired_difference_summary([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0
    assert result["paired_t_statistic"] == 0.0
    assert result["paired_t_pvalue"] == 1.0
    assert result["wilcoxon_statistic"] == 0.0
    assert result["wilcoxon_pvalue"] == 1.0


@pytest.mark.parametrize(
    ("a", "b"),
    [([1.0], [1.0]), ([1.0, 2.0], [1.0]), ([1.0, np.nan], [1.0, 2.0])],
)
def test_paired_difference_rejects_invalid_pairs(a, b):
    with pytest.raises(ValueError):
        paired_difference_summary(a, b)


def test_tost_establishes_equivalence_only_inside_predeclared_margin():
    equivalent = paired_tost(
        [-0.01, 0.0, 0.01],
        [0.0, 0.0, 0.0],
        margin=0.05,
    )
    boundary = paired_tost(
        [0.04, 0.05, 0.06],
        [0.0, 0.0, 0.0],
        margin=0.05,
    )

    assert equivalent["performed"] is True
    assert equivalent["equivalent"] is True
    assert equivalent["ci_level"] == pytest.approx(0.90)
    assert boundary["equivalent"] is False
    assert boundary["upper_pvalue"] == pytest.approx(0.5)


def test_tost_without_margin_is_explicitly_not_performed():
    result = paired_tost([1.0, 2.0], [1.0, 2.0], margin=None)

    assert result == {
        "performed": False,
        "equivalent": None,
        "reason": "equivalence_margin_not_predeclared",
    }


@pytest.mark.parametrize("margin", [0.0, -0.1, np.inf])
def test_tost_rejects_invalid_margin(margin):
    with pytest.raises(ValueError):
        paired_tost([1.0, 2.0], [1.0, 2.0], margin=margin)


def test_blocked_lack_of_fit_decomposes_expected_historical_degrees_of_freedom():
    n = 140
    basis = np.eye(n)
    reduced = basis[:, :19]
    additive = basis[:, :29]
    cell_means = basis[:, :125]
    y = np.linspace(-1.0, 1.0, n)

    result = blocked_lack_of_fit_decomposition(y, reduced, additive, cell_means)

    assert result["components"]["structural_lack_of_fit"]["df"] == 10
    assert result["components"]["treatment_by_block"]["df"] == 96
    assert result["components"]["center_pure_error"]["df"] == 15
    assert result["models"]["reduced"]["residual_df"] == 121
    assert result["identity"]["df_components_sum"] == 121
    assert result["identity"]["ss_components_sum"] == pytest.approx(
        result["models"]["reduced"]["sse"]
    )


def test_blocked_lack_of_fit_reports_restricted_design_rank_deficiency():
    n = 95
    basis = np.eye(n)
    reduced = np.column_stack([basis[:, :18], basis[:, 1]])
    additive = basis[:, :20]
    cell_means = basis[:, :80]
    y = np.linspace(-1.0, 1.0, n)

    result = blocked_lack_of_fit_decomposition(y, reduced, additive, cell_means)

    assert result["models"]["reduced"]["column_count"] == 19
    assert result["models"]["reduced"]["rank"] == 18
    assert result["models"]["reduced"]["rank_deficiency"] == 1
    assert result["components"]["structural_lack_of_fit"]["df"] == 2
    assert result["components"]["treatment_by_block"]["df"] == 60
    assert result["components"]["center_pure_error"]["df"] == 15


def test_blocked_lack_of_fit_rejects_non_nested_models():
    y = np.arange(6.0)
    reduced = np.eye(6)[:, [0]]
    additive = np.eye(6)[:, [1, 2]]
    cell_means = np.eye(6)[:, [1, 2, 3, 4]]

    with pytest.raises(ValueError, match="nested"):
        blocked_lack_of_fit_decomposition(y, reduced, additive, cell_means)


def test_blocked_lack_of_fit_marks_zero_denominator_tests_unavailable():
    basis = np.eye(8)
    reduced = basis[:, :1]
    additive = basis[:, :2]
    cell_means = basis[:, :4]
    y = basis[:, 0]

    result = blocked_lack_of_fit_decomposition(y, reduced, additive, cell_means)

    assert result["tests"]["structural_vs_pooled_additive"]["available"] is False
    assert result["tests"]["interaction_vs_center_pure_error"]["available"] is False
