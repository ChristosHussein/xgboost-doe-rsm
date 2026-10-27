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
