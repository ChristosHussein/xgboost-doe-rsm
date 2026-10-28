"""Bounded, real-development-only verification; never scores official test data."""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'phase_b_sequential_xai'))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgb_doe_xai.config import BenchmarkConfig
from xgb_doe_xai.data import create_strictly_isolated_partitions
from xgb_doe_xai.execution import make_development_evaluator
from xgb_doe_xai.study import run_study, check_study, write_json, source_hashes
from xgb_doe_xai.word_study import run_word_study, check_word_study
from run_phase_b import generate_all_findings

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT/'results/cuda_fix_verification')
output = parser.parse_args().output.resolve()
output.mkdir(exist_ok=False)
source = ROOT/'phase_b_sequential_xai/data/imdb.csv'
frame = pd.read_csv(source)
development = frame[frame.split == 'train']
subset, _ = train_test_split(development, train_size=2000, stratify=development.label, random_state=239)
subset = subset.sort_index()
cfg = BenchmarkConfig(n_trees=100, center_replicates=2, adaptive_cycles=2,
    infill_per_cycle=1, search_replicates=1, include_static_baseline=True,
    training_threads=4, device='cuda', hv_ref_loss=1., hv_ref_latency_us=1000.)
# Identity and all protocol choices are written before collection; fixed scales are
# diagnostic constants, not an empirically calibrated deployment requirement.
write_json(output/'verification_contract.json', dict(scope='development-only operational verification',
    subset_ids=subset.document_id.tolist(), seed=239, config=asdict(cfg), source_hashes=source_hashes(),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    search_fits=420, confirmation_fits=180, word_training_fits=1, final_test_fits=0,
    stopping='complete declared 601 fits; no automatic full campaign; no official test scoring'))
# All supplied documents are development documents. This pilot's internal reserve
# is unused, and no official test document is passed to the study.
parts = create_strictly_isolated_partitions(subset.text.tolist(), subset.label.to_numpy(),
    n_blocks=5, random_seed=42, max_features=cfg.max_features)
start = time.perf_counter()
run_study(cfg, make_development_evaluator(parts,cfg), output/'study',
    dataset_id='imdb-development-only-cuda-verification',
    dataset_sha256=hashlib.sha256(subset.to_csv(index=False).encode()).hexdigest(),
    evidence_kind='development_pilot', final_evaluator=None)
summary=check_study(output/'study')
selection=next(row for row in json.loads((output/'study/selected.json').read_text()) if row['method']=='doe')
run_word_study(parts,cfg,selection,output/'words',documents=2)
check_word_study(output/'words')
generate_all_findings(output/'study', output/'findings', dataset_name='imdb-development-pilot')
assert not (output/'study/final_results.json').exists()
assert summary['final_fits']==0 and summary['total_fits']==600
write_json(output/'completion.json',dict(status='PASSED',wall_seconds=time.perf_counter()-start,
    study_fits=600,word_training_fits=1,official_test_evaluations=0,
    checks='study registry, prediction reconstruction, confirmation, word replay and findings export passed'))
print('CUDA DEVELOPMENT PILOT PASSED',flush=True)
