import numpy as np
from xgb_doe_xai.research_analysis import bootstrap_mean, progress_to_target


def test_bootstrap_mean_preserves_undefined_and_constant_samples():
    assert bootstrap_mean([None,None])['mean'] is None
    assert bootstrap_mean([2.,2.,2.])['interval'] == [2.,2.]
    assert bootstrap_mean([2.])['interval'] is None


def test_efficiency_target_uses_shared_reference_and_actual_fit_axis():
    curves={'a':[(5,1.),(10,3.)], 'b':[(5,2.),(10,4.)]}
    result=progress_to_target(curves, fraction=.75)
    assert result['target']==3.
    assert result['fits']=={'a':10,'b':10}
    assert progress_to_target({'a':[(5,0.)]})['fits']=={'a':None}


def test_progress_never_compares_incomplete_or_unequally_weighted_blocks():
    from xgb_doe_xai.research_analysis import completed_block_progress
    cfg=dict(n_nuisance_blocks=2,hv_ref_loss=1.,hv_ref_latency_us=10.)
    def row(block,loss):
        return dict(coded_params=dict.fromkeys(['x1','x2','x3','x4'],0.),
                    block_id=block,val_loss=loss,latency_us=5.)
    result=completed_block_progress([row(0,.2),row(0,.2),row(1,.8)],cfg)
    assert result[:2]==[(1,0.),(2,0.)]
    assert result[-1][1]==2.5
