import os
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'phase_b_sequential_xai'))
from xgb_doe_xai.benchmarks import train_and_evaluate_config
from xgb_doe_xai.config import BenchmarkConfig
from xgb_doe_xai.data import create_strictly_isolated_partitions, generate_synthetic_sentiment_corpus
from xgb_doe_xai.execution import final_classification
from xgb_doe_xai.profiler import WarmedLatencyProfiler
from xgb_doe_xai.word_study import run_word_study


def reverse_storage(matrix):
    result = matrix.copy()
    for lo, hi in zip(result.indptr[:-1], result.indptr[1:]):
        result.indices[lo:hi] = result.indices[lo:hi][::-1]
        result.data[lo:hi] = result.data[lo:hi][::-1]
    result.has_sorted_indices = False
    return result


class TrainingInputTests(unittest.TestCase):
    def setUp(self):
        texts, labels = generate_synthetic_sentiment_corpus(120)
        self.parts = create_strictly_isolated_partitions(texts, labels, n_blocks=2, max_features=100)
        a, y, b, z = self.parts.block_features[0]
        a.sort_indices(); b.sort_indices()
        self.a, self.y, self.b, self.z = reverse_storage(a), y, b, z
        self.coded = dict(x1=0., x2=-1., x3=1., x4=0.)
        self.cfg = BenchmarkConfig(mode='smoke', n_trees=5, n_nuisance_blocks=2,
                                   warmup_inferences=3, timed_inferences=5)

    def fit_guard(self):
        original = xgb.XGBClassifier.fit
        def guarded(model, matrix, *args, **kwargs):
            if sparse.issparse(matrix):
                self.assertTrue(matrix.has_canonical_format, 'Training received noncanonical sparse data')
            return original(model, matrix, *args, **kwargs)
        return patch.object(xgb.XGBClassifier, 'fit', guarded)

    def test_development_training_canonicalizes_without_mutating_input(self):
        before = self.a.indices.copy()
        with self.fit_guard():
            train_and_evaluate_config(self.coded, self.a, self.y, self.b, self.z,
                                      WarmedLatencyProfiler(3, 5), n_trees=5)
        np.testing.assert_array_equal(self.a.indices, before)

    def test_final_refit_canonicalizes(self):
        from dataclasses import asdict
        with self.fit_guard():
            final_classification(self.a, self.y, self.b, self.z,
                                 dict(method='doe', replicate=0, coded_params=self.coded), asdict(self.cfg))

    def test_word_model_canonicalizes(self):
        self.parts.block_features[0] = (self.a, self.y, self.b, self.z)
        with tempfile.TemporaryDirectory() as tmp, self.fit_guard():
            run_word_study(self.parts, self.cfg, dict(coded_params=self.coded), Path(tmp)/'words', documents=1)

    @unittest.skipUnless(os.environ.get('RUN_CUDA_TESTS') == '1', 'Explicit CUDA hardware check')
    def test_gpu_trained_word_model_replays_on_cpu(self):
        self.cfg.device = 'cuda'
        self.cfg.n_trees = 100
        self.parts.block_features[0] = (self.a, self.y, self.b, self.z)
        with tempfile.TemporaryDirectory() as tmp:
            run_word_study(self.parts, self.cfg, dict(coded_params=self.coded), Path(tmp)/'words', documents=2)

    @unittest.skipUnless(os.environ.get('RUN_CUDA_TESTS') == '1', 'Explicit CUDA hardware check')
    def test_cuda_predictions_are_invariant_to_sparse_storage(self):
        profiler = WarmedLatencyProfiler(3, 5)
        a = self.a.copy(); a.sort_indices()
        original_fit = xgb.XGBClassifier.fit
        devices = []
        def fit_and_record(model, *args, **kwargs):
            result = original_fit(model, *args, **kwargs)
            devices.append(json.loads(model.get_booster().save_config())['learner']['generic_param']['device'])
            return result
        with patch.object(xgb.XGBClassifier, 'fit', fit_and_record):
            results = [train_and_evaluate_config(self.coded, m, self.y, self.b, self.z,
                       profiler, n_trees=20, device='cuda') for m in [self.a, a]]
        self.assertTrue(all(device.startswith('cuda') for device in devices), devices)
        np.testing.assert_allclose(results[0].validation_probabilities,
                                   results[1].validation_probabilities, rtol=0, atol=1e-7)


class ReportingTests(unittest.TestCase):
    def test_undefined_statistic_is_not_zero(self):
        from run_phase_b import format_statistic
        self.assertEqual(format_statistic(None), 'N/A')
        self.assertEqual(format_statistic(0.), '0.0000')

    def test_duplicate_sparse_entries_are_summed_without_changing_values(self):
        from xgb_doe_xai.features import canonical_features
        raw = sparse.csr_matrix(([2., 1., 3.], [1, 0, 1], [0, 3]), shape=(1, 2))
        actual = canonical_features(raw)
        self.assertTrue(actual.has_canonical_format)
        np.testing.assert_array_equal(actual.toarray(), raw.toarray())
        self.assertEqual(raw.nnz, 3)

    def test_development_pilot_rejects_final_test_evaluator(self):
        from xgb_doe_xai.study import run_study
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'must not evaluate final test'):
                run_study(BenchmarkConfig(), None, Path(tmp)/'study', dataset_id='fixture',
                          dataset_sha256='fixture', evidence_kind='development_pilot', final_evaluator=object())


if __name__ == '__main__':
    unittest.main()
