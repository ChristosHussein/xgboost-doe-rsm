"""Explicit real-data input and shared development/final classification execution."""
from dataclasses import asdict
import csv
import hashlib
from pathlib import Path

import numpy as np

from .benchmarks import train_and_evaluate_config
from .profiler import WarmedLatencyProfiler


def load_sentiment_csv(path):
    """Read text,label and optional split (train/test) without network downloads.

    Duplicate documents are rejected so the caller must resolve cross-split copies
    before preregistration. Official test rows, when supplied, remain test rows.
    """
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'text', 'label'} <= set(reader.fieldnames or ()):
            raise ValueError('Sentiment CSV requires text and label columns')
        rows = list(reader)
    texts = [row['text'].strip() for row in rows]
    if not texts or not all(texts) or len(set(texts)) != len(texts):
        raise ValueError('Documents must be nonempty and unique before splitting')
    labels = np.array([int(row['label']) for row in rows])
    if set(labels) != {0, 1}:
        raise ValueError('Binary sentiment labels must contain exactly 0 and 1')
    mask = None
    if 'split' in rows[0]:
        if {r['split'] for r in rows} != {'train', 'test'}:
            raise ValueError('Official split column must contain train and test')
        mask = np.array([row['split'] == 'test' for row in rows])
    return texts, labels, mask, hashlib.sha256(path.read_bytes()).hexdigest()


def make_development_evaluator(partitions, config):
    profiler = WarmedLatencyProfiler(config.warmup_inferences, config.timed_inferences)
    def evaluate(point, block, seed, cfg):
        x_train, y_train, x_valid, y_valid = partitions.block_features[block]
        return train_and_evaluate_config(point.coded, x_train, y_train, x_valid, y_valid,
                                         profiler, point.point_id, block, cfg.n_trees, seed,cfg.training_threads)
    return evaluate


def final_classification(x_train, y_train, x_test, y_test, selection, training_config):
    profiler = WarmedLatencyProfiler(training_config['warmup_inferences'], training_config['timed_inferences'])
    result = train_and_evaluate_config(selection['coded_params'], x_train, y_train, x_test, y_test,
        profiler, f"FINAL_{selection['method']}_{selection['replicate']}", -1,
        training_config['n_trees'], training_config['random_seed']+900000+selection['replicate'],
        training_config.get('training_threads',1))
    return dict(method=selection['method'], replicate=selection['replicate'], **asdict(result))
