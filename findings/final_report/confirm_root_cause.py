"""Confirm sparse-index trigger on the original development fold; no test scoring."""
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score,log_loss
root=Path(__file__).resolve().parents[2]
df=pd.read_csv(root/'phase_b_sequential_xai/data/imdb.csv')
df=df[df.split=='train'].reset_index(drop=True)
tr,va=next(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(df.text,df.label))
vec=TfidfVectorizer(max_features=10000,ngram_range=(1,2),sublinear_tf=True)
a=vec.fit_transform(df.text.iloc[tr]); b=vec.transform(df.text.iloc[va])
sa=a.copy();sa.sort_indices();sb=b.copy();sb.sort_indices()
assert (a-sa).nnz==0 and (b-sb).nnz==0
out=Path(__file__).parent/'diagnostic'
report=dict(scope='development fold zero only; controlled diagnostic, not replacement campaign',
    xgboost=xgb.__version__,original_train_sorted=a.has_sorted_indices,original_validation_sorted=b.has_sorted_indices,
    feature_values_identical_after_sort=True,train_shape=a.shape,validation_shape=b.shape,results=[])
params=dict(max_depth=3,learning_rate=.05,subsample=1.,reg_lambda=.1,n_estimators=100,n_jobs=4,
            random_state=42,objective='binary:logistic',tree_method='hist')
for mode in ['cuda_original','cuda_sorted','cpu_sorted']:
    x=a if mode=='cuda_original' else sa
    m=xgb.XGBClassifier(device='cuda' if mode.startswith('cuda') else 'cpu',**params)
    started=time.perf_counter();m.fit(x,df.label.iloc[tr]);secs=time.perf_counter()-started
    t=[json.loads(z) for z in m.get_booster().get_dump(dump_format='json',with_stats=True)]
    predictions={}
    for predictor in ['cpu','cuda']:
        m.get_booster().set_param({'device':predictor})
        for order,z in [('original',b),('sorted',sb)]:
            p=m.predict_proba(z)[:,1]
            predictions[predictor+'_'+order]=p
    p=predictions['cpu_sorted']
    r=dict(mode=mode,training_seconds=secs,f1=float(f1_score(df.label.iloc[va],p>=.5,average='macro')),
        logloss=float(log_loss(df.label.iloc[va],p)),leaf_only=sum('leaf' in z for z in t),
        max_prediction_diff_from_cpu_sorted={k:float(np.max(np.abs(v-p))) for k,v in predictions.items()},
        first_tree=t[0],pmin=float(p.min()),pmax=float(p.max()))
    if mode=='cuda_original':
        saved=xgb.Booster();saved.load_model(root/'results/phase_b_run/imdb_words/classifier.ubj')
        r['tree_dump_matches_saved_word_model']=saved.get_dump(dump_format='json')==m.get_booster().get_dump(dump_format='json')
    report['results'].append(r)
    m.save_model(out/(mode+'_fullfold.ubj'))
    np.save(out/(mode+'_fullfold_probabilities.npy'),p)
    (out/'root_cause.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in r.items() if k!='first_tree'}),flush=True)
