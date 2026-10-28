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
from xgb_doe_xai.study import write_json, source_hashes, run_study, git_provenance

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
                df = pd.read_parquet(path)
                for i, (text_val, label_val) in enumerate(zip(df['text'], df['label'])):
                    text=' '.join(html.unescape(re.sub(r'<[^>]+>',' ',text_val)).split())
                    rows.append(dict(document_id=f'{split}/{i}',text=text,label=int(label_val),split=split))
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
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req,timeout=120) as response,partial.open('wb') as target:
                while chunk:=response.read(1024*1024):
                    target.write(chunk)
            partial.replace(path)
            return
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def make_imdb_cohort(data_dir, limit):
    """Freeze a development-only cohort; preserve every retained official test row."""
    if limit < 100:
        raise ValueError('Cohort needs enough development documents for blocked analysis')
    source=data_dir/'imdb.csv'
    path=data_dir/f'imdb-dev-{limit}.csv'
    with source.open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    development=[r for r in rows if r['split']=='train']
    if limit>=len(development):
        raise ValueError('Use full IMDb instead of a cohort covering the whole development set')
    indices,_=train_test_split(np.arange(len(development)),train_size=limit,
             stratify=[r['label'] for r in development],random_state=239)
    chosen=[development[i] for i in sorted(indices)]
    if path.exists():
        raise FileExistsError('Cohort already exists; preserve the frozen selection')
    with path.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(chosen+[r for r in rows if r['split']=='test'])
    write_json(path.with_suffix('.json'),dict(parent_csv_sha256=file_hash(source),
        csv_sha256=file_hash(path),declared_at_utc=datetime.now(timezone.utc).isoformat(),
        seed=239,development_rows=limit,selected_document_ids=[r['document_id'] for r in chosen],
        holdout='all retained official test documents; no sampling of test performance'))
    return path


def calibrate(data_dir, output, imdb_csv=None, training_threads=1):
    output.mkdir(parents=True,exist_ok=False)
    contract=dict(declared_at_utc=datetime.now(timezone.utc).isoformat(),
        programme_source_sha256=file_hash(__file__),training_threads=training_threads,
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
        dataset_path=imdb_csv if name=='imdb' and imdb_csv else data_dir/(name+'.csv')
        texts,labels,mask,checksum=load_sentiment_csv(dataset_path)
        cfg=BenchmarkConfig(center_replicates=2,adaptive_cycles=2,infill_per_cycle=1,training_threads=training_threads,
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
            dataset_path=str(dataset_path),
            selected_tree_count=cfg.n_trees,within_largest_tree_f1_tolerance=True,
            calibration_fits=len(records),planned_search_fits=search,planned_confirmation_fits=confirm,
            planned_final_fits=final,estimated_evaluation_seconds=float(max(chosen)*(search+confirm+final)),
            estimate_scope='pilot center-depth runtime; excludes surrogate/explanation overhead and depth variation',
            high_capacity_validation_improvement=largest['val_macro_f1']-low[0]['val_macro_f1']))


def run_programme(data_dir,output,calibration):
    """Freeze both applications before either final test evaluation is possible."""
    from xgb_doe_xai.research_analysis import analyze_study
    from xgb_doe_xai.word_study import run_word_study
    output.mkdir(parents=True,exist_ok=False)
    configs={name:json.loads((calibration/(name+'_config.json')).read_text()) for name in SOURCES}
    metadata={name:json.loads((calibration/(name+'_calibration.json')).read_text()) for name in SOURCES}
    dataset_paths={name:Path(metadata[name]['dataset_path']) for name in SOURCES}
    for name,path in dataset_paths.items():
        if file_hash(path)!=metadata[name]['dataset_sha256']:
            raise ValueError('Dataset changed since calibration')
        if search_budget(BenchmarkConfig(**configs[name]))['fits_per_method_search']!=140:
            raise ValueError('Original per-method fit budget was not preserved')
    write_json(output/'programme_contract.json',dict(
        declared_at_utc=datetime.now(timezone.utc).isoformat(),**git_provenance(),
        source_hashes=source_hashes(),programme_source_sha256=file_hash(__file__),configs=configs,
        dataset_sha256={name:file_hash(path) for name,path in dataset_paths.items()},
        dataset_paths={name:str(path) for name,path in dataset_paths.items()},
        source_provenance={name:json.loads((data_dir/(name+'.json')).read_text()) for name in SOURCES},
        imdb_cohort=json.loads(dataset_paths['imdb'].with_suffix('.json').read_text()),
        calibration=metadata,words=dict(documents=10,additional_training_fits_per_dataset=1,
            selection='adaptive DOE, search replicate zero; first eligible development validation documents'),
        analysis=dict(bootstrap_repetitions=2000,bootstrap_seed=812,interval_quantiles=[.025,.975],
            efficiency_target_fraction=.95,paired_unit='search replicate within dataset',
            scope='exploratory repeated-search comparison; shared folds; no equivalence or universal superiority claim'),
        stopping='complete all declared budgets; no test-driven changes; failures remain in evidence'))
    started=time.perf_counter()
    for name in SOURCES:
        cfg=BenchmarkConfig(**configs[name])
        texts,labels,mask,checksum=load_sentiment_csv(dataset_paths[name])
        parts=create_strictly_isolated_partitions(texts,labels,cfg.holdout_ratio,
              cfg.n_nuisance_blocks,cfg.random_seed,max_features=cfg.max_features,official_holdout_mask=mask)
        print('START',name,datetime.now(timezone.utc).isoformat(),flush=True)
        run_study(cfg,make_development_evaluator(parts,cfg),output/name,
                  dataset_id=name+'-sentiment-revision-v1',dataset_sha256=checksum,
                  evidence_kind='research',final_evaluator=parts.final_evaluator)
        write_json(output/(name+'_analysis.json'),analyze_study(output/name))
        selections=json.loads((output/name/'selected.json').read_text())
        selection=next(r for r in selections if r['method']=='doe' and r['replicate']==0)
        run_word_study(parts,cfg,selection,output/(name+'_words'))
        print('COMPLETE',name,datetime.now(timezone.utc).isoformat(),flush=True)
    write_json(output/'completion.json',dict(status='COMPLETED',wall_seconds=time.perf_counter()-started,
                                           completed_at_utc=datetime.now(timezone.utc).isoformat()))
    write_json(output/'programme_registry.json',{
        p.relative_to(output).as_posix():file_hash(p) for p in output.rglob('*') if p.is_file()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','cohort','calibrate','run'])
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--imdb-csv',type=Path)
    parser.add_argument('--cohort-size',type=int)
    parser.add_argument('--training-threads',type=int,default=1)
    parser.add_argument('--calibration',type=Path)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.data_dir)
    elif args.stage=='cohort':
        if not args.cohort_size:parser.error('--cohort-size is required')
        print(make_imdb_cohort(args.data_dir,args.cohort_size))
    elif args.stage=='run':
        if not args.output or not args.calibration:parser.error('--output and --calibration are required')
        run_programme(args.data_dir,args.output,args.calibration)
    else:
        if not args.output:parser.error('--output is required for calibration')
        calibrate(args.data_dir,args.output,args.imdb_csv,args.training_threads)
