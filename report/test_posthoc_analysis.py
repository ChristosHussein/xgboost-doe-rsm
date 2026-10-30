"""Analytic fixtures for post-hoc estimands, pairing and design rank."""
import unittest
from itertools import product
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from posthoc_analysis import (METHODS, fit_summary, grouped_points, hypervolume,
                              macro_f1_from_counts, paired_bootstrap)


class PosthocAnalysisTests(unittest.TestCase):
    def test_macro_f1_matches_confusion_definition(self):
        counts = np.asarray([[40, 10, 20, 30], [3, 0, 0, 0]])
        np.testing.assert_allclose(macro_f1_from_counts(counts), [(8/11+2/3)/2, .5])

    def test_bootstrap_preserves_identical_predictions_and_rejects_unpaired_labels(self):
        labels = np.tile([0, 1], 100)
        probability = np.where(labels == 1, .8, .2)
        rows = [dict(method=method, replicate=repeat, validation_labels=labels.tolist(),
                     validation_probabilities=probability.tolist()) for method in METHODS for repeat in range(3)]
        result = paired_bootstrap(rows, repetitions=100, seed=91)
        self.assertTrue(all(r['difference'] == 0 and r['interval_95'] == [0., 0.]
                            for r in result['comparisons']))
        rows[1]['validation_labels'] = labels[::-1].tolist()
        with self.assertRaisesRegex(ValueError, 'identical ordered test labels'):
            paired_bootstrap(rows, repetitions=100)

    def test_configuration_averages_weight_blocks_equally(self):
        coded = dict(x1=0, x2=0, x3=0, x4=0)
        rows = [dict(coded_params=coded, block_id=b, val_loss=value, latency_us=100*value)
                for b, value in [(0, .1), (0, .3), (1, .8)]]
        point = grouped_points(rows)[0]
        self.assertAlmostEqual(point['loss'], .5)
        self.assertAlmostEqual(point['latency'], 50.)

    def test_hypervolume_integrates_union_not_sum_of_overlapping_rectangles(self):
        points = [dict(loss=.1, latency=2), dict(loss=.2, latency=1)]
        self.assertAlmostEqual(hypervolume(points, (1, 3)), 1.7)
        self.assertEqual(hypervolume(points, (.05, .5)), 0.)

    def test_restricted_domain_detects_linear_quadratic_aliasing(self):
        coordinates = np.asarray(list(product([-1., 0., 1.], repeat=4)))
        x = np.tile(coordinates, (6, 1))
        frame = pd.DataFrame(x, columns=['x1', 'x2', 'x3', 'x4'])
        frame['block'] = np.repeat(np.arange(3), len(coordinates)*2)
        rng = np.random.default_rng(71)
        response = 1+x[:, 0]+x[:, 1]**2+.3*x[:, 2]*x[:, 3]+rng.normal(0, .02, len(x))
        full = fit_summary(frame, response)
        keep = frame.x1 >= 0
        restricted = fit_summary(frame.loc[keep], response[keep])
        self.assertFalse(full['rank_deficient'])
        self.assertTrue(restricted['rank_deficient'])
        self.assertEqual(restricted['quadratic_rank'], restricted['quadratic_columns']-1)
        self.assertGreater(restricted['df_lack_of_fit'], 0)

    def test_residual_figure_is_identical_under_coefficient_roundoff(self):
        import matplotlib
        matplotlib.use('Agg')
        from posthoc_publication import residual_diagnostics

        root = Path(__file__).resolve().parents[1]
        original = np.linalg.lstsq
        def perturbed(*args, **kwargs):
            result = original(*args, **kwargs)
            return (result[0]+1e-14, *result[1:])

        with tempfile.TemporaryDirectory() as temporary:
            baseline, changed = [Path(temporary)/name for name in ('baseline', 'changed')]
            for output in (baseline, changed):
                (output/'figures').mkdir(parents=True)
            residual_diagnostics(root, baseline)
            with patch('numpy.linalg.lstsq', side_effect=perturbed):
                residual_diagnostics(root, changed)
            self.assertEqual((baseline/'figures/phase_a_diagnostics.pdf').read_bytes(),
                             (changed/'figures/phase_a_diagnostics.pdf').read_bytes())


if __name__ == '__main__':
    unittest.main()
