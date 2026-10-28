"""Synthetic publication fixtures test bindings, never stand in for experiment evidence."""
import pytest
import numpy as np
from scripts import research_publication as publication


def test_incomplete_research_cannot_be_published(tmp_path):
    with pytest.raises(ValueError,match='incomplete'):
        publication.verified_campaign(tmp_path)


def test_new_metrics_are_bound_to_the_correct_method_and_response(monkeypatch):
    bundle={'contract':dict(configs={},imdb_cohort=dict(development_rows=200),source_provenance={},
        analysis=dict(efficiency_target_fraction=.95))}
    for dataset in publication.DATASETS:
        cfg=dict(n_trees=100,n_nuisance_blocks=5,search_replicates=3,warmup_inferences=100,
                 timed_inferences=500,training_threads=4,latency_threshold_us=100.)
        bundle['contract']['configs'][dataset]=cfg
        bundle['contract']['source_provenance'][dataset]=dict(split_counts=dict(train=200,test=50))
        analysis=dict(estimates={},agreement={},confirmation={},gp_confirmation={},curvature_agreement=[],adaptive_change={},
            progress={'0':{'shared_target':{'fits':dict.fromkeys(publication.METHODS,130)}}})
        for i,method in enumerate(publication.METHODS):
            analysis['estimates'][method]={k:dict(mean=.1+i,sd=.01) for k in
                ('test_macro_f1','test_logloss','test_probability_rmse','test_latency_us','hypervolume',
                 'observed_reference_regret','feasibility_rate','overhead_seconds')}
            for response in ('val_loss','latency_us'):
                key=method+':'+response
                analysis['agreement'][key]={k:dict(mean=.2,sd=None) for k in ('spearman_rho','interaction_correlation')}
                analysis['agreement'][key]['per_search']=[dict(top_factor_anova='x1',top_factor_shap='x2',
                    top_interaction_anova=['x1','x2'],top_interaction_shap=None)]
                analysis['curvature_agreement'].append(dict(method=method,response=response,spearman_rho=.4))
                analysis['confirmation'][key]=dict(rmse=.1,observed_minus_predicted_bias=-.01,pi_coverage=.5)
                analysis['gp_confirmation'][key]=dict(rmse=.2,observed_minus_predicted_bias=.01,coverage=.25)
        bundle[dataset]=dict(analysis=analysis,summary={'test_results':[]},measurement=dict(search_fits_per_method=420),
            words=dict(documents=[dict(confirmation=dict(rmse=.04),rsm=dict(rank_defined=False))]))
    monkeypatch.setattr(publication,'verified_campaign',lambda root:bundle)
    monkeypatch.setattr(publication,'common_mean_confirmation',lambda *args:tuple(np.array([.1,.2]) for _ in range(3)))
    records=publication.metric_records()
    assert records['numTextImdbAdaptiveFOneMean']['value']==1.1
    assert records['numTextImdbAdaptiveFOneMean']['selector']=='estimates.doe.test_macro_f1.mean'
    assert records['numTextImdbAdaptiveLossOlsCoverage']['value']==50.
    assert records['numTextImdbAdaptiveLossGpCoverage']['value']==25.
    assert records['numTextYelpFits']['value']==140
    assert records['numTextYelpAdaptiveLossFactorRhoSd']['formatted']==r'\text{undefined}'
    assert r'\numTextImdbAdaptiveFOneMean' in publication.render_benchmark_table()
