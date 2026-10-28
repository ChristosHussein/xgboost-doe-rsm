"""Regression tests for requirements from the author's research notes."""
import numpy as np
import pytest

from xgb_doe_xai.config import DEFAULT_FACTORS, BenchmarkConfig, uncode_configuration


def test_original_four_factors_and_log_regularization():
    assert {f.name for f in DEFAULT_FACTORS.values()} == {
        'learning_rate', 'max_depth', 'subsample', 'reg_lambda'}
    center = uncode_configuration(dict.fromkeys(DEFAULT_FACTORS, 0.))
    assert center['reg_lambda'] == pytest.approx(1.)
    for level, expected in [(-1., .1), (1., 10.)]:
        point = dict.fromkeys(DEFAULT_FACTORS, 0.)
        point['x4'] = level
        assert uncode_configuration(point)['reg_lambda'] == pytest.approx(expected)


def test_original_fit_budget_includes_adaptive_evaluations():
    from xgb_doe_xai.design import search_budget
    cfg = BenchmarkConfig(center_replicates=2, adaptive_cycles=2, infill_per_cycle=1)
    budget = search_budget(cfg)
    assert budget == dict(initial_coordinates=26, adaptive_coordinates=2,
                         total_coordinates=28, fits_per_method_search=140)


def test_local_word_rsm_recovers_interaction_and_checks_unseen_perturbations():
    from xgb_doe_xai.local_explanations import explain_word_response
    def response(weights):
        return .2 + .1*weights[:, 0] + .3*weights[:, 0]*weights[:, 1]
    evidence = explain_word_response(response, ['good', 'acting', 'slow', 'ending'], seed=91)
    assert evidence['confirmation']['rmse'] < 1e-10
    assert evidence['rsm']['interaction_matrix_ss'][0][1] > 0
    assert abs(evidence['shap']['raw_interaction_values'][0][0][1]) > 0
    assert len(evidence['confirmation']['weights']) > 0
    assert evidence['scope'] == 'conditional feature-weight perturbations; not causal word effects'


def test_local_explanation_requires_distinct_named_words():
    from xgb_doe_xai.local_explanations import explain_word_response
    with pytest.raises(ValueError):
        explain_word_response(lambda x: x[:, 0], ['same']*4)


def test_document_explanation_uses_real_classifier_probabilities():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from xgb_doe_xai.local_explanations import explain_document
    texts=['good acting charming ending', 'bad acting boring ending',
           'great acting moving ending', 'poor acting slow ending']
    vectorizer=TfidfVectorizer()
    features=vectorizer.fit_transform(texts)
    model=LogisticRegression().fit(features,[1,0,1,0])
    evidence=explain_document(model,vectorizer,texts[0])
    assert set(evidence['words'])=={'good','acting','charming','ending'}
    assert all(0<=p<=1 for p in evidence['confirmation']['observed'])


def test_local_word_evidence_reconstructs_from_saved_model(tmp_path):
    from xgb_doe_xai import generate_synthetic_sentiment_corpus,create_strictly_isolated_partitions
    from xgb_doe_xai.word_study import run_word_study,check_word_study
    texts,labels=generate_synthetic_sentiment_corpus(100,42)
    parts=create_strictly_isolated_partitions(texts,labels,n_blocks=2,max_features=100)
    cfg=BenchmarkConfig(n_trees=2)
    output=tmp_path/'words'
    run_word_study(parts,cfg,{'coded_params':dict.fromkeys(DEFAULT_FACTORS,0.)},output,documents=2)
    check_word_study(output)


def test_word_reconstruction_tolerates_only_roundoff():
    from xgb_doe_xai.word_study import equivalent_reconstruction
    assert equivalent_reconstruction({'values':[.25+1e-15]},{'values':[.25]})
    assert not equivalent_reconstruction({'values':[.251]},{'values':[.25]})
    assert not equivalent_reconstruction({'word':'good'},{'word':'bad'})


def test_constant_local_response_has_no_invented_ranking():
    from xgb_doe_xai.local_explanations import explain_word_response
    evidence=explain_word_response(lambda x:np.repeat(.5,len(x)),['a','b','c','d'])
    assert evidence['rsm']['rank_defined'] is False
    assert evidence['rsm']['main_effect_ranking']==[]
