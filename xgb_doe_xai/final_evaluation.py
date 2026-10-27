"""Explicit final-evaluation gate; hashes detect changes, not past data exposure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


class FinalEvaluator:
    """Holds test observations separately from development-facing partitions.

    This API enforces the workflow, not adversarial access to Python internals.
    A dataset digest cannot demonstrate that data were historically untouched.
    """
    def __init__(self, dev_texts, dev_labels, test_texts, test_labels, vectorizer_kwargs):
        self._dev_texts = tuple(dev_texts)
        self._dev_labels = tuple(int(y) for y in dev_labels)
        self._texts = tuple(test_texts)
        self._labels = tuple(int(y) for y in test_labels)
        self._vectorizer_kwargs = dict(vectorizer_kwargs)
        self.holdout_sha256 = digest([self._texts, self._labels])
        self.development_sha256 = digest([self._dev_texts, self._dev_labels])
        self._contract_digest = None
        self._evaluated = False

    def freeze(self, path, selections, dataset_id, training_config):
        """Persist the complete selected configurations before any final access."""
        if self._contract_digest is not None or not selections or not dataset_id:
            raise ValueError('Freeze requires a dataset identity, selections, and an unused evaluator')
        payload = dict(dataset_id=dataset_id, holdout_sha256=self.holdout_sha256,
                       development_sha256=self.development_sha256,
                       selections=selections, training_config=training_config,
                       preprocessing=self._vectorizer_kwargs)
        serialized = json.dumps(payload, sort_keys=True, allow_nan=False, indent=2)
        with Path(path).open('x', encoding='utf-8') as stream:
            stream.write(serialized)
        self._contract_digest = digest(json.loads(serialized))

    def evaluate(self, contract_path, output_path, evaluate_fn):
        """Call the final trainer only after verifying the frozen contract and data.

        evaluate_fn receives train X/y, test X/y, one selection, and training_config.
        Output creation is exclusive to prevent overwriting previous final evidence.
        """
        if self._contract_digest is None or self._evaluated:
            raise ValueError('Final evaluation requires frozen selections and may run only once')
        contract = json.loads(Path(contract_path).read_text(encoding='utf-8'))
        if (digest(contract) != self._contract_digest
                or digest([self._texts, self._labels]) != self.holdout_sha256
                or digest([self._dev_texts, self._dev_labels]) != self.development_sha256):
            raise ValueError('Final evaluation integrity mismatch')
        if Path(output_path).exists():
            raise FileExistsError(output_path)
        vectorizer = TfidfVectorizer(**self._vectorizer_kwargs)
        train = vectorizer.fit_transform(self._dev_texts)
        test = vectorizer.transform(self._texts)
        self._evaluated = True  # Fail closed if a callback fails after seeing test data.
        rows = [evaluate_fn(train, np.array(self._dev_labels), test, np.array(self._labels),
                            selection, contract['training_config'])
                for selection in contract['selections']]
        with Path(output_path).open('x', encoding='utf-8') as stream:
            json.dump(dict(contract_sha256=self._contract_digest, results=rows), stream,
                      allow_nan=False, indent=2)
        return rows
