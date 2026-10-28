"""Small development-only device/input checks; never evaluates official test rows."""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss, f1_score

root = Path(__file__).resolve().parents[2]
out = Path(__file__).parent / 'diagnostic'
out.mkdir(exist_ok=True)
df = pd.read_csv(root/'phase_b_sequential_xai/data/imdb.csv')
df = df[df.split == 'train']
small, _ = train_test_split(df, train_size=2000, stratify=df.label, random_state=239)
train, valid = train_test_split(small, test_size=.2, stratify=small.label, random_state=42)
v = TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)
xt = v.fit_transform(train.text); xv = v.transform(valid.text)
results=[]
for device in ['cpu','cuda']:
    for kind in ['original','sorted','float32_sorted','dense_missing_nan']:
        a,b=xt.copy(),xv.copy()
        if kind != 'original':
            a.sort_indices(); b.sort_indices()
        if kind == 'float32_sorted':
            a=a.astype(np.float32); b=b.astype(np.float32)
        if kind == 'dense_missing_nan':
            a=a.toarray().astype(np.float32); b=b.toarray().astype(np.float32)
            a[a==0]=np.nan; b[b==0]=np.nan
        m=xgb.XGBClassifier(device=device,tree_method='hist',n_estimators=20,
            max_depth=3,learning_rate=.05,subsample=1,reg_lambda=.1,
            n_jobs=4,random_state=42,objective='binary:logistic',eval_metric='logloss')
        t=time.perf_counter()
        m.fit(a,train.label,eval_set=[(b,valid.label)],verbose=False)
        p=m.predict_proba(b)[:,1]
        trees=[json.loads(x) for x in m.get_booster().get_dump(dump_format='json',with_stats=True)]
        r=dict(device=device,input=kind,shape=a.shape,dtype=str(a.dtype),
            f1=float(f1_score(valid.label,p>=.5,average='macro')),logloss=float(log_loss(valid.label,p)),
            first_loss=m.evals_result()['validation_0']['logloss'][0],last_loss=m.evals_result()['validation_0']['logloss'][-1],
            leaf_only=sum('leaf' in x for x in trees),pmin=float(p.min()),pmax=float(p.max()),
            seconds=time.perf_counter()-t)
        results.append(r); print(json.dumps(r),flush=True)
        m.save_model(out/f'{device}_{kind}.ubj')
        (out/'results.json').write_text(json.dumps(dict(version=xgb.__version__,scope='development-only diagnostic, not campaign results',results=results),indent=2))
