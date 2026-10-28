import numpy as np
import xgboost as xgb


def test_profiling_restores_training_affinity():
    from xgb_doe_xai.profiler import WarmedLatencyProfiler, get_process_affinity
    x=np.arange(80,dtype=np.float32).reshape(20,4)
    model=xgb.XGBClassifier(n_estimators=2,n_jobs=1).fit(x,np.arange(20)%2)
    before=get_process_affinity()
    WarmedLatencyProfiler(1,2).profile_booster(model.get_booster(),x[:1])
    assert get_process_affinity()==before
