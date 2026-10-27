"""Counterexamples for the scientific contracts of the new framework."""
import itertools
import numpy as np
import pytest

from xgb_doe_xai import (
    PolynomialRSMSurrogate, compute_surrogate_shap_attributions,
    create_strictly_isolated_partitions, generate_multiblock_fccd,
)
from xgb_doe_xai.xai import AnovaAttributions, ShapAttributions, compare_anova_vs_shap


def test_exact_interactions_vanish_for_additive_response():
    class Additive:
        def predict(self, x):
            return x[:, 0] + 2 * x[:, 1]
    x = np.array(list(itertools.product([-1., 1.], repeat=4)))
    result = compute_surrogate_shap_attributions(Additive(), x, x)
    tensor = result.raw_interaction_values
    assert tensor is not None
    assert np.allclose(tensor[:, 0, 1], 0)
    assert np.allclose(tensor.sum(axis=2), result.raw_shap_values)


def test_validation_vocabulary_is_not_used_to_fit_fold_preprocessing():
    texts = [f'common shared unique{i:03}' for i in range(100)]
    parts = create_strictly_isolated_partitions(texts, np.arange(100) % 2)
    for (train, _, valid, _), vectorizer in zip(parts.blocks, parts.vectorizers):
        only_valid = set(' '.join(valid).split()) - set(' '.join(train).split())
        assert only_valid.isdisjoint(vectorizer.vocabulary_)
    assert not hasattr(parts, 'holdout_labels')
    assert not hasattr(parts, 'holdout_features')


def test_fixed_block_predictions_preserve_model_metadata_and_reject_unseen_block():
    points = generate_multiblock_fccd(n_blocks=2, n_center_per_block=4)
    x = np.array([p.vector for p in points])
    blocks = np.array([p.block_id for p in points])
    y = x[:, 0] + 5 * blocks + np.random.default_rng(2).normal(0, .01, len(x))
    model = PolynomialRSMSurrogate().fit(x, y, blocks)
    names = list(model.feature_names)
    a = model.predict(x[:1], block_ids=np.array([0]))
    b = model.predict(x[:1], block_ids=np.array([1]))
    assert b[0] - a[0] == pytest.approx(5, abs=.02)
    assert model.feature_names == names
    with pytest.raises(ValueError, match='unseen'):
        model.predict_interval(x[:1], block_ids=np.array([2]))


def test_rank_deficient_surface_is_rejected():
    with pytest.raises(ValueError, match='rank'):
        PolynomialRSMSurrogate(False).fit(np.zeros((30, 4)), np.arange(30.))


def test_missing_git_is_reported_without_preventing_execution(monkeypatch):
    from xgb_doe_xai import study
    def unavailable(*args, **kwargs):
        raise FileNotFoundError('git unavailable')
    monkeypatch.setattr(study.subprocess, 'run', unavailable)
    assert study.git_provenance() == {'git_revision': 'UNAVAILABLE', 'git_worktree_dirty': None}


def test_single_class_evaluation_retains_binary_logloss_protocol():
    from xgb_doe_xai import WarmedLatencyProfiler, train_and_evaluate_config
    x = np.arange(80, dtype=np.float32).reshape(20, 4)
    result = train_and_evaluate_config(dict.fromkeys(['x1', 'x2', 'x3', 'x4'], 0.),
        x, np.arange(20) % 2, x[:4], np.zeros(4, dtype=int), WarmedLatencyProfiler(1, 2), n_trees=2)
    from sklearn.metrics import log_loss
    probabilities = np.asarray(result.validation_probabilities, dtype=result.validation_probability_dtype)
    assert result.val_logloss == pytest.approx(log_loss(np.zeros(4), probabilities, labels=[0, 1]))


def test_known_pair_interaction_and_undefined_ranks():
    class Pair:
        def predict(self, x):
            return 3*x[:, 0]*x[:, 1] + x[:, 2]**2
    x = np.array(list(itertools.product([-1., 0., 1.], repeat=4)))
    explanation = compute_surrogate_shap_attributions(Pair(), x, x)
    tensor = explanation.raw_interaction_values
    assert np.allclose(tensor[:, 0, 1], 1.5*x[:, 0]*x[:, 1])
    assert np.allclose(tensor, tensor.transpose(0, 2, 1))
    assert np.allclose(tensor.sum((1, 2)), Pair().predict(x)-Pair().predict(x).mean())
    factors = ['x1', 'x2', 'x3', 'x4']
    anova = AnovaAttributions(factors, dict(zip(factors, [1, 2, 3, 4])),
                             factors[::-1], np.zeros((4, 4)), {})
    constant = ShapAttributions(factors, dict.fromkeys(factors, 1.), factors,
                                np.zeros((4, 4)), np.ones((1, 4)))
    result = compare_anova_vs_shap(anova, constant)
    assert result.spearman_rho is None and result.kendall_tau is None


def test_final_evaluator_requires_freeze_and_rejects_modified_contract(tmp_path):
    parts = create_strictly_isolated_partitions(
        [f'common unique{i}' for i in range(100)], np.arange(100) % 2)
    gate = parts.final_evaluator
    contract, output = tmp_path/'selection.json', tmp_path/'final.json'
    called = []
    def callback(*args):
        called.append(True)
        return {'score': .5}
    with pytest.raises(ValueError, match='frozen'):
        gate.evaluate(contract, output, callback)
    gate.freeze(contract, [{'max_depth': 6}], 'synthetic-fixture', {'n_trees': 5})
    original = contract.read_text()
    contract.write_text(original.replace('"max_depth": 6', '"max_depth": 7'))
    with pytest.raises(ValueError, match='integrity'):
        gate.evaluate(contract, output, callback)
    assert not called
    contract.write_text(original)
    gate.evaluate(contract, output, callback)
    assert called == [True] and output.exists()
    with pytest.raises(ValueError, match='once'):
        gate.evaluate(contract, output, callback)


def test_integer_acquisition_and_training_budget_match(monkeypatch):
    from xgb_doe_xai.config import canonical_configuration
    from xgb_doe_xai.design import select_adaptive_infill_points
    from xgb_doe_xai import generate_fccd_points
    import xgb_doe_xai.benchmarks as benchmark
    from types import SimpleNamespace
    points = select_adaptive_infill_points(
        generate_fccd_points(), lambda x: np.ones(len(x)), n_infill=3)
    for point in points:
        assert point.coded == canonical_configuration(point.coded)
    with pytest.raises(ValueError, match='criterion'):
        select_adaptive_infill_points(points, lambda x: np.ones(len(x)), criterion='typo')
    seen = []
    def evaluate(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(val_loss=.1, latency=SimpleNamespace(predict_median_us=100))
    monkeypatch.setattr(benchmark, 'train_and_evaluate_config', evaluate)
    benchmark.run_matched_motpe_benchmark(2, None, None, None, None, None, n_trees=23)
    assert [x['n_trees'] for x in seen] == [23, 23]
    assert all(x['coded_params'] == canonical_configuration(x['coded_params']) for x in seen)


def test_profiler_rejects_missing_affinity_and_records_uncached_observations(monkeypatch):
    import xgb_doe_xai.profiler as profiler
    monkeypatch.setattr(profiler, 'pin_cpu_affinity', lambda _: False)
    with pytest.raises(RuntimeError, match='affinity'):
        profiler.WarmedLatencyProfiler(1, 2).profile_booster(None, np.ones((1, 4)))
    monkeypatch.setattr(profiler, 'pin_cpu_affinity', lambda _: True)
    matrices = []
    monkeypatch.setattr(profiler.xgb, 'DMatrix', lambda *args, **kwargs: object())
    class Booster:
        def set_param(self, params):
            assert params['nthread'] == 1
        def predict(self, matrix):
            matrices.append(matrix)
            return np.array([.5])
        def inplace_predict(self, sample):
            return np.array([.5])
    result = profiler.WarmedLatencyProfiler(3, 5).profile_booster(Booster(), np.ones((1, 4)))
    assert len({id(m) for m in matrices}) == 8
    assert len(result.predict_raw_us) == len(result.inplace_raw_us) == 5
    assert result.affinity_applied


def test_independent_observation_coverage_for_known_fixed_blocks():
    rng = np.random.default_rng(301)
    points = generate_multiblock_fccd(n_blocks=3)
    x = np.array([p.vector for p in points])
    blocks = np.array([p.block_id for p in points])
    def response(x, b):
        return .4 + .03*x[:, 0] + .02*x[:, 1]**2 + .01*x[:, 0]*x[:, 2] + .1*b
    model = PolynomialRSMSurrogate().fit(x, response(x, blocks)+rng.normal(0, .01, len(x)), blocks)
    independent = rng.uniform(-1, 1, (1000, 4))
    new_blocks = rng.integers(0, 3, 1000)
    truth = response(independent, new_blocks)+rng.normal(0, .01, 1000)
    _, low, high = model.predict_interval(independent, block_ids=new_blocks)
    assert .89 < np.mean((low <= truth) & (truth <= high)) < .995


def test_sequential_runner_has_disjoint_confirmation_and_equal_logged_fit_budgets(tmp_path, monkeypatch):
    from xgb_doe_xai.study import run_study, check_study
    from xgb_doe_xai import BenchmarkConfig
    from xgb_doe_xai.benchmarks import EvaluationResult
    from xgb_doe_xai.profiler import LatencyProfileResult
    cfg = BenchmarkConfig(n_nuisance_blocks=2, adaptive_cycles=1, infill_per_cycle=2,
                          confirmation_points=3, search_replicates=1, mode='smoke',
                          acquisition='expected_improvement')
    import xgb_doe_xai.study as study
    acquire = study.select_adaptive_infill_points
    acquisition_calls = []
    def record_acquisition(*args, **kwargs):
        acquisition_calls.append(kwargs['best_observed'])
        return acquire(*args, **kwargs)
    monkeypatch.setattr(study, 'select_adaptive_infill_points', record_acquisition)
    seen = []
    def evaluator(point, block, seed, config):
        seen.append((point.point_id, block))
        x = point.vector
        latency = LatencyProfileResult(100+10*x[1], 100., 100., 1., 102., 80., 80., 80.,
                                       100, 500, 1)
        return EvaluationResult(point.point_id, point.coded, {},
                                .8, .2+.02*x[0]+.01*x[0]*x[1]+.005*block,
                                .3, latency, block_id=block, n_trees=config.n_trees, seed=seed)
    output = tmp_path/'study'
    summary = run_study(cfg, evaluator, output, dataset_id='synthetic',
                        dataset_sha256='fixture-digest', evidence_kind='synthetic_fixture')
    assert summary['search_fits']['doe'] == summary['search_fits']['tpe'] == 60
    assert summary['confirmation_fits'] == 12
    assert summary['total_fits'] == len(seen) == 132
    assert summary['final_evaluation_status'] == 'NOT RUN'
    check_study(output)
    import json
    rows = [json.loads(line) for line in (output/'evaluations.jsonl').read_text().splitlines()]
    initial_means = study.group_search([r for r in rows if r['stage'] == 'initial'])
    expected_best = min(r['loss']/cfg.hv_ref_loss + r['latency']/cfg.hv_ref_latency_us for r in initial_means)
    assert acquisition_calls == pytest.approx([expected_best])
    search = {tuple(r['coded_params'].values()) for r in rows if r['stage'] != 'confirmation'}
    confirmation = {tuple(r['coded_params'].values()) for r in rows if r['stage'] == 'confirmation'}
    assert search.isdisjoint(confirmation)
    original_summary = (output/'summary.json').read_text()
    (output/'summary.json').write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        check_study(output)
    (output/'summary.json').write_text(original_summary)
    # Updating a checksum cannot bless a report contradicting its bound evidence.
    import hashlib
    report = output/'report.md'
    report.write_text(report.read_text(encoding='utf-8')+'\nUntraced result: 0.999\n', encoding='utf-8')
    registry_path = output/'artifact_registry.json'
    registry = json.loads(registry_path.read_text())
    registry['files']['report.md'] = hashlib.sha256(report.read_bytes()).hexdigest()
    registry_path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match='Report differs'):
        check_study(output)


def test_official_test_split_and_modified_holdout_are_enforced(tmp_path):
    labels = np.arange(100) % 2
    parts = create_strictly_isolated_partitions(
        [f'common token{i}' for i in range(100)], labels,
        official_holdout_mask=np.arange(100) >= 80)
    assert parts.holdout_count == 20
    assert all('token99' not in ' '.join(b[0]+b[2]) for b in parts.blocks)
    gate = parts.final_evaluator
    gate.freeze(tmp_path/'freeze.json', [{'max_depth': 6}], 'fixture', {})
    gate._labels = tuple(1-np.array(gate._labels))
    with pytest.raises(ValueError, match='integrity'):
        gate.evaluate(tmp_path/'freeze.json', tmp_path/'final.json', lambda *args: None)
