"""Prepare and calibrate approved public-data studies without touching housing evidence."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.model_selection import train_test_split
from xgb_doe_xai.config import BenchmarkConfig
from xgb_doe_xai.data import create_strictly_isolated_partitions
from xgb_doe_xai.execution import load_sentiment_csv, make_development_evaluator
from xgb_doe_xai.design import DesignPoint, search_budget
from xgb_doe_xai.study import write_json, source_hashes

SOURCES = {
    'imdb': 'https://huggingface.co/datasets/stanfordnlp/imdb/tree/e6281661ce1c48d982bc483cf8a173c1bbeb5d31/plain_text',
    'yelp': 'https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip'}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        manifest = data_dir/(name+'.json')
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if file_hash(data_dir/(name+'.csv')) != saved['csv_sha256']:
                raise ValueError('Prepared dataset changed')
            continue
        rows = []
        source_files = {}
        if name == 'imdb':
            import pandas as pd
            for split in ('train','test'):
                path = data_dir/('imdb_'+split+'.parquet')
                source_url=url.replace('/tree/','/resolve/')+'/'+split+'-00000-of-00001.parquet'
                download(source_url,path)
                source_files[source_url]=file_hash(path)
                for i,row in pd.read_parquet(path).iterrows():
                    text=' '.join(html.unescape(re.sub(r'<[^>]+>',' ',row['text'])).split())
                    rows.append(dict(document_id=f'{split}/{i}',text=text,label=int(row['label']),split=split))
        else:
            archive=data_dir/(name+'.zip')
            download(url,archive)
            source_files[url]=file_hash(archive)
            with zipfile.ZipFile(archive) as source:
                path = next(p for p in source.namelist() if p.endswith('/yelp_labelled.txt'))
                for i,line in enumerate(source.read(path).decode('utf-8').splitlines()):
                    text,label = line.rsplit('\t',1)
                    rows.append(dict(document_id=str(i),text=' '.join(text.split()),label=int(label)))
            # Partition once, before calibration or metric inspection.
            _, heldout = train_test_split(np.arange(len(rows)),test_size=.2,
                        stratify=[r['label'] for r in rows],random_state=137)
            heldout = set(heldout.tolist())
            for i,row in enumerate(rows):
                row['split'] = 'test' if i in heldout else 'train'
        before = len(rows)
        # Prefer official/reserved test documents. Never use labels to choose duplicates.
        seen, cleaned, exclusions = set(), [], []
        for row in sorted(rows,key=lambda r:(r['split']!='test',r['document_id'])):
            identity = row['text'].casefold()
            if identity in seen or not identity:
                exclusions.append(dict(document_id=row['document_id'],split=row['split'],
                                       reason='empty or duplicate normalized text'))
                continue
            seen.add(identity)
            cleaned.append(row)
        path = data_dir/(name+'.csv')
        with path.open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['document_id','text','label','split'])
            writer.writeheader();writer.writerows(cleaned)
        write_json(manifest,dict(dataset=name,source_url=url,source_files=source_files,
            prepared_at_utc=datetime.now(timezone.utc).isoformat(),original_rows=before,
            prepared_rows=len(cleaned),split_counts={s:sum(r['split']==s for r in cleaned) for s in ('train','test')},
            csv_sha256=file_hash(path),exclusions=exclusions,
            preparation='HTML removal and whitespace collapse; casefold for duplicate identity only; test preferred',
            holdout='official' if name=='imdb' else 'pre-calibration stratified split, seed 137'))
        print('Prepared',name,len(cleaned),'documents',flush=True)


def download(url,path):
    if path.exists():
        return
    print('Downloading',path.name,flush=True)
    partial=path.with_suffix(path.suffix+'.partial')
    with urllib.request.urlopen(url,timeout=45) as response,partial.open('wb') as target:
        while chunk:=response.read(1024*1024):
            target.write(chunk)
    partial.replace(path)


def calibrate(data_dir, output):
    output.mkdir(parents=True,exist_ok=False)
    contract=dict(declared_at_utc=datetime.now(timezone.utc).isoformat(),
        stage='development-only calibration, not confirmatory results',source_hashes=source_hashes(),
        tree_grid=[100,300,600],eta_grid=[-.0,-1.],
        selection='smallest tree count whose low-rate validation F1 is within 0.005 of the largest count',
        latency_limit='median uncached classic latency across calibration fits; design constraint, not serving SLA',
        reference='loss 1; latency twice the maximum calibration median latency',
        primary_search='static CCD, adaptive CCD, MO-TPE; equal 140 fits per method/search',
        search_replicates=3,blocks=5,confirmation_points=12,
        word_protocol='first eligible development documents; four highest TF-IDF unigrams; no label-based selection',
        uncertainty='paired search-repeat summaries; fold variability descriptive because documents overlap',
        stopping='complete frozen budgets; preserve failures and unfavorable findings; no test-driven revisions')
    write_json(output/'calibration_contract.json',contract)
    for name in SOURCES:
        texts,labels,mask,checksum=load_sentiment_csv(data_dir/(name+'.csv'))
        cfg=BenchmarkConfig(center_replicates=2,adaptive_cycles=2,infill_per_cycle=1,
                            include_static_baseline=True)
        parts=create_strictly_isolated_partitions(texts,labels,cfg.holdout_ratio,
             cfg.n_nuisance_blocks,cfg.random_seed,max_features=10000,official_holdout_mask=mask)
        records=[]
        evaluate=make_development_evaluator(parts,cfg)
        for trees in contract['tree_grid']:
            cfg.n_trees=trees
            for eta in contract['eta_grid']:
                point=DesignPoint(f'CAL_{trees}_{eta}','calibration',dict(x1=eta,x2=0.,x3=0.,x4=0.),0)
                started=time.perf_counter()
                result=evaluate(point,0,cfg.random_seed,cfg)
                records.append(dict(**asdict(result),wall_seconds=time.perf_counter()-started))
                write_json(output/(name+'_pilot.json'),dict(dataset_sha256=checksum,observations=records))
                print(name,trees,eta,'F1',result.val_macro_f1,'seconds',records[-1]['wall_seconds'],flush=True)
        low=[r for r in records if r['coded_params']['x1']==-1.]
        largest=next(r for r in low if r['n_trees']==max(contract['tree_grid']))
        eligible=[r['n_trees'] for r in low if r['val_macro_f1']>=largest['val_macro_f1']-.005]
        cfg.n_trees=min(eligible)
        latencies=[r['latency']['predict_median_us'] for r in records]
        cfg.latency_threshold_us=float(np.median(latencies))
        cfg.hv_ref_loss=1.
        cfg.hv_ref_latency_us=2*max(latencies)
        search=search_budget(cfg)['fits_per_method_search']*cfg.search_replicates*3
        confirm=cfg.confirmation_points*cfg.n_nuisance_blocks*cfg.search_replicates*3
        final=cfg.search_replicates*3
        chosen=[r['wall_seconds'] for r in records if r['n_trees']==cfg.n_trees]
        write_json(output/(name+'_config.json'),asdict(cfg))
        write_json(output/(name+'_calibration.json'),dict(dataset_sha256=checksum,
            selected_tree_count=cfg.n_trees,within_largest_tree_f1_tolerance=True,
            calibration_fits=len(records),planned_search_fits=search,planned_confirmation_fits=confirm,
            planned_final_fits=final,estimated_evaluation_seconds=float(max(chosen)*(search+confirm+final)),
            estimate_scope='pilot center-depth runtime; excludes surrogate/explanation overhead and depth variation',
            high_capacity_validation_improvement=largest['val_macro_f1']-low[0]['val_macro_f1']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','calibrate'])
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.data_dir)
    else:
        if not args.output:parser.error('--output is required for calibration')
        calibrate(args.data_dir,args.output)
