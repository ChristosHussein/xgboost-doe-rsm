"""Literal word deletion with revectorization, exact SHAP, and local RSM checks."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import sparse
import xgboost as xgb
from xgb_doe_xai.local_explanations import explain_word_response
from xgb_doe_xai.word_study import equivalent_reconstruction
from xgb_doe_xai.study import write_json,check_study

SCOPE='literal word deletion and revectorization; RSM approximates expected response under independent word retention; not causal effects'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deleted_texts(text,words):
    if len(words)!=4 or len(set(words))!=4:
        raise ValueError('Specify four distinct words')
    variants=[]
    for mask in range(16):
        edited=text
        for i,word in enumerate(words):
            if not mask&(1<<i):
                edited=re.sub(r'(?<!\w)'+re.escape(word)+r'(?!\w)',' ',edited,flags=re.IGNORECASE)
        variants.append(' '.join(edited.split()))
    return variants


def expected_response(probabilities,retention):
    probabilities=np.asarray(probabilities,dtype=float)
    retention=np.asarray(retention,dtype=float)
    if probabilities.shape!=(16,) or retention.ndim!=2 or retention.shape[1]!=4:
        raise ValueError('Expected a four-word coalition game and retention matrix')
    if (not np.isfinite(probabilities).all() or not np.isfinite(retention).all()
            or np.any((retention<0)|(retention>1))):
        raise ValueError('Retention probabilities must be finite and in the unit interval')
    present=np.array([[bool(mask&(1<<i)) for i in range(4)] for mask in range(16)])
    weights=np.where(present[None,:,:],retention[:,None,:],1-retention[:,None,:]).prod(axis=2)
    return weights@probabilities


def declared_protocol():
    return dict(declared_at_utc=datetime.now(timezone.utc).isoformat(),
        scope=SCOPE,word_count=4,document_rule='same first eligible development documents as the frozen feature-weight study',
        coalition_rule='all literal word-presence subsets; delete all whole-token occurrences, then revectorize including bigrams',
        response='positive-class probability; RSM response is its exact expectation under independent word-retention probabilities',
        confirmation='held-out retention probabilities, not independent natural reviews',
        metrics=['conditional SHAP main and pair contributions','partial ANOVA interaction and curvature SS',
                 'held-out retention-distribution approximation RMSE','constant-response undefined rankings'],
        additional_classifier_training_fits=0,causal_claim=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest())


def assert_primary_registry(campaign):
    expected=json.loads((campaign/'programme_registry.json').read_text())
    actual={p.relative_to(campaign).as_posix():sha(p) for p in campaign.rglob('*')
            if p.is_file() and p!=campaign/'programme_registry.json'}
    if actual!=expected:
        raise ValueError('Primary programme is incomplete or changed')


def run(campaign,output,protocol_path,data_dir=None):
    assert_primary_registry(campaign)
    protocol=json.loads(protocol_path.read_text())
    if protocol['source_sha256']!=declared_protocol()['source_sha256']:
        raise ValueError('Word-masking implementation changed after declaration')
    contract=json.loads((campaign/'programme_contract.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'protocol.json',protocol)
    started=time.perf_counter()
    from xgb_doe_xai.config import BenchmarkConfig
    from xgb_doe_xai.data import create_strictly_isolated_partitions
    from xgb_doe_xai.execution import load_sentiment_csv
    for dataset in ('imdb','yelp'):
        check_study(campaign/dataset)
        path=Path(contract['dataset_paths'][dataset])
        if data_dir:path=data_dir/path.name
        if sha(path)!=contract['dataset_sha256'][dataset]:
            raise ValueError('Word-stage dataset differs from the frozen application')
        cfg=BenchmarkConfig(**contract['configs'][dataset])
        texts,labels,mask,_=load_sentiment_csv(path)
        parts=create_strictly_isolated_partitions(texts,labels,cfg.holdout_ratio,cfg.n_nuisance_blocks,
                        cfg.random_seed,max_features=cfg.max_features,official_holdout_mask=mask)
        model_path=campaign/(dataset+'_words')/'classifier.ubj'
        booster=xgb.Booster();booster.load_model(model_path);booster.set_param({'nthread':1})
        originals=json.loads((campaign/(dataset+'_words')/'observations.json').read_text())['documents']
        documents=[]
        for ordinal,original in enumerate(originals):
            index=original['validation_document_index']
            text=parts.blocks[0][2][index]
            variants=deleted_texts(text,original['words'])
            vectors=parts.vectorizers[0].transform(variants)
            probabilities=booster.predict(xgb.DMatrix(vectors,nthread=1)).astype(float)
            evidence=explain_word_response(lambda p:expected_response(probabilities,p),original['words'],seed=original['seed'])
            evidence['scope']=SCOPE
            evidence.update(validation_document_index=index,original_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                            coalition_probabilities=probabilities.tolist(),queried_text_variants=len(variants))
            sparse.save_npz(output/f'{dataset}_{ordinal}_vectors.npz',vectors)
            documents.append(evidence)
        write_json(output/(dataset+'_observations.json'),dict(model_sha256=sha(model_path),documents=documents))
    write_json(output/'completion.json',dict(status='COMPLETED',wall_seconds=time.perf_counter()-started))
    write_json(output/'registry.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()})
    check(campaign,output)


def check(campaign,output):
    assert_primary_registry(campaign)
    registry=json.loads((output/'registry.json').read_text())
    if {p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='registry.json'}!=registry:
        raise ValueError('Word-masking evidence changed')
    for dataset in ('imdb','yelp'):
        evidence=json.loads((output/(dataset+'_observations.json')).read_text())
        model_path=campaign/(dataset+'_words')/'classifier.ubj'
        if sha(model_path)!=evidence['model_sha256']:
            raise ValueError('Word-study classifier differs from its frozen source')
        booster=xgb.Booster();booster.load_model(model_path);booster.set_param({'nthread':1})
        for ordinal,original in enumerate(evidence['documents']):
            vectors=sparse.load_npz(output/f'{dataset}_{ordinal}_vectors.npz')
            probabilities=booster.predict(xgb.DMatrix(vectors,nthread=1)).astype(float)
            if not np.allclose(probabilities,original['coalition_probabilities'],rtol=1e-12,atol=1e-12):
                raise ValueError('Literal text-variant predictions do not reconstruct')
            replay=explain_word_response(lambda p:expected_response(probabilities,p),original['words'],seed=original['seed'])
            replay['scope']=SCOPE
            if not equivalent_reconstruction(replay,{k:original[k] for k in replay}):
                raise ValueError('Word RSM and SHAP results do not reconstruct')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,default=ROOT/'results/sentiment_revision_v1/confirmed_campaign')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--declare',type=Path)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--data-dir',type=Path)
    args=parser.parse_args()
    if args.declare:
        if args.declare.exists():raise FileExistsError(args.declare)
        write_json(args.declare,declared_protocol())
    elif args.check:
        if not args.output:parser.error('--output required')
        check(args.campaign,args.output)
    else:
        if not args.output or not args.protocol:parser.error('--output and --protocol required')
        run(args.campaign,args.output,args.protocol,args.data_dir)
