"""xgb_doe_xai.data - Data ingestion, TF-IDF representations, and strictly isolated partitions.

Addresses Remark 3:
Keeps holdout observations out of the development-facing partitions. An explicit
final evaluator checks frozen selections before exposing observations to evaluation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy import sparse
from xgb_doe_xai.final_evaluation import FinalEvaluator


@dataclass
class DatasetPartitions:
    holdout_sha256: str
    holdout_count: int
    final_evaluator: FinalEvaluator
    # List of (train_texts, train_labels, val_texts, val_labels) per block
    blocks: List[Tuple[List[str], np.ndarray, List[str], np.ndarray]]
    vectorizers: List[TfidfVectorizer]
    block_features: Optional[
        List[Tuple[sparse.csr_matrix, np.ndarray, sparse.csr_matrix, np.ndarray]]
    ] = None


def generate_synthetic_sentiment_corpus(
    n_samples: int = 2000,
    random_seed: int = 42,
) -> Tuple[List[str], np.ndarray]:
    """Generate a reproducible, hermetic sentiment text corpus for testing and development."""
    rng = np.random.RandomState(random_seed)
    pos_words = [
        "excellent", "superb", "brilliant", "outstanding", "masterpiece",
        "delightful", "compelling", "fantastic", "amazing", "wonderful",
        "stellar", "engaging", "touching", "charming", "triumph",
    ]
    neg_words = [
        "terrible", "awful", "dreadful", "boring", "disaster",
        "waste", "horrible", "shallow", "flawed", "ridiculous",
        "uninspired", "tedious", "cringe", "painful", "mess",
    ]
    neutral_words = [
        "movie", "film", "story", "character", "actor", "scene", "director",
        "plot", "production", "cinema", "performance", "dialogue", "screen",
        "audience", "music", "camera", "tempo", "setting", "style", "release",
    ]

    texts: List[str] = []
    labels: List[int] = []

    for i in range(n_samples):
        is_pos = rng.rand() > 0.5
        label = 1 if is_pos else 0
        sentiment_pool = pos_words if is_pos else neg_words
        n_sent = rng.randint(3, 7)
        n_neut = rng.randint(8, 16)

        words = list(rng.choice(sentiment_pool, size=n_sent)) + list(
            rng.choice(neutral_words, size=n_neut)
        )
        rng.shuffle(words)
        texts.append(" ".join(words))
        labels.append(label)

    return texts, np.array(labels, dtype=np.int32)


def create_strictly_isolated_partitions(
    texts: List[str],
    labels: np.ndarray,
    holdout_ratio: float = 0.20,
    n_blocks: int = 5,
    random_seed: int = 42,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> DatasetPartitions:
    """Partition dataset into an immutable holdout set and n_blocks nuisance blocks.

    The holdout partition is hashed immediately to verify it remains unmodified.
    """
    dev_texts, holdout_texts, dev_labels, holdout_labels = train_test_split(
        texts,
        labels,
        test_size=holdout_ratio,
        stratify=labels,
        random_state=random_seed,
    )

    vectorizer_kwargs = dict(max_features=max_features, ngram_range=ngram_range, sublinear_tf=True)
    final_evaluator = FinalEvaluator(dev_texts, dev_labels, holdout_texts, holdout_labels,
                                     vectorizer_kwargs)

    # Partition development pool into n_blocks nuisance blocks
    skf = StratifiedKFold(n_splits=n_blocks, shuffle=True, random_state=random_seed)
    blocks: List[Tuple[List[str], np.ndarray, List[str], np.ndarray]] = []

    dev_texts_arr = np.array(dev_texts, dtype=object)
    for train_idx, val_idx in skf.split(dev_texts_arr, dev_labels):
        b_train_texts = dev_texts_arr[train_idx].tolist()
        b_train_labels = dev_labels[train_idx]
        b_val_texts = dev_texts_arr[val_idx].tolist()
        b_val_labels = dev_labels[val_idx]
        blocks.append((b_train_texts, b_train_labels, b_val_texts, b_val_labels))

    # Fit preprocessing separately inside each training fold.
    vectorizers = []
    block_feats: List[Tuple[sparse.csr_matrix, np.ndarray, sparse.csr_matrix, np.ndarray]] = []
    for b_tr_t, b_tr_l, b_val_t, b_val_l in blocks:
        vectorizer = TfidfVectorizer(**vectorizer_kwargs)
        tr_feat = vectorizer.fit_transform(b_tr_t)
        val_feat = vectorizer.transform(b_val_t)
        block_feats.append((tr_feat, b_tr_l, val_feat, b_val_l))
        vectorizers.append(vectorizer)

    return DatasetPartitions(
        holdout_sha256=final_evaluator.holdout_sha256,
        holdout_count=len(holdout_texts),
        final_evaluator=final_evaluator,
        blocks=blocks,
        vectorizers=vectorizers,
        block_features=block_feats,
    )
