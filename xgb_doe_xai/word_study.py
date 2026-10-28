"""Replayable local explanations of a selected classifier on development documents."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
import xgboost as xgb

from .config import uncode_configuration
from .local_explanations import explain_feature_vector
from .study import write_json


def run_word_study(partitions,config,selection,output,*,documents=10):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    contract=dict(selection=selection,config=asdict(config),training_fits=1,
        document_rule='first eligible documents in development validation block zero; labels not used to select',
        document_limit=documents,confirmation_points=64,
        perturbation='attenuate four highest-TFIDF present unigrams; all other feature weights fixed; no renormalization',
        scope='conditional feature-weight perturbations; not causal word effects')
    write_json(output/'contract.json',contract)
    x_train,y_train,x_valid,_=partitions.block_features[0]
    params=uncode_configuration(selection['coded_params'])
    model=xgb.XGBClassifier(**params,n_estimators=config.n_trees,
        n_jobs=config.training_threads,random_state=config.random_seed,objective='binary:logistic')
    started=time.perf_counter()
    model.fit(x_train,y_train)
    training_seconds=time.perf_counter()-started
    model.save_model(output/'classifier.ubj')
    names=partitions.vectorizers[0].get_feature_names_out()
    results,indices=[],[]
    for i in range(len(partitions.blocks[0][2])):
        row=x_valid[i]
        if sum(' ' not in names[j] for j in row.indices)<4:
            continue
        result=explain_feature_vector(model,row,names,seed=config.random_seed+len(results))
        result['validation_document_index']=i
        results.append(result);indices.append(i)
        if len(results)==documents:
            break
    if not results:
        raise ValueError('No eligible development documents for local word study')
    sparse.save_npz(output/'inputs.npz',x_valid[indices])
    write_json(output/'feature_names.json',names.tolist())
    write_json(output/'observations.json',dict(training_seconds=training_seconds,documents=results))
    write_json(output/'registry.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.iterdir() if p.is_file()})
    check_word_study(output)


def check_word_study(output):
    output=Path(output)
    expected=json.loads((output/'registry.json').read_text())
    actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()
            if p.is_file() and p.name!='registry.json'}
    if actual!=expected:
        raise ValueError('Local explanation evidence was altered')
    booster=xgb.Booster()
    booster.load_model(output/'classifier.ubj')
    booster.set_param({'nthread':1})
    class Oracle:
        def predict_proba(self,x):
            p=booster.predict(xgb.DMatrix(x,nthread=1))
            return np.column_stack([1-p,p])
    vectors=sparse.load_npz(output/'inputs.npz')
    names=json.loads((output/'feature_names.json').read_text())
    observations=json.loads((output/'observations.json').read_text())['documents']
    for row,original in zip(vectors,observations):
        replay=explain_feature_vector(Oracle(),row,names,seed=original['seed'],
                                     confirmation_points=len(original['confirmation']['weights']))
        if replay!={k:v for k,v in original.items() if k!='validation_document_index'}:
            raise ValueError('Local explanations do not reconstruct from saved classifier and vectors')
