"""Small synthetic checks for opt-in replication protocols (no dataset training)."""
import unittest
from dataclasses import dataclass, asdict
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import tempfile
import json
import hashlib
from copy import deepcopy
import numpy as np
from scipy import sparse

from protocol_extensions import (ProtocolOptions, capacity_threshold, length_panel, coupled_vectors,
    PanelProfiler, GatedPanelFinalEvaluator, freeze_protocol, validate_calibration,
    validate_calibration_records, check_protocol, load_replication_config)
from xgb_doe_xai.config import BenchmarkConfig


class ProtocolTests(unittest.TestCase):
    def calibration_fixture(self, directory, options=None):
        options = options or ProtocolOptions(latency_panel_size=2)
        config = BenchmarkConfig(mode='smoke', warmup_inferences=3, timed_inferences=5, n_trees=100)
        path = directory/'imdb_protocol_calibration.json'
        identity = freeze_protocol(path, options, config, 'fixture-hash',
            panels={'0': [0, 1]} if options.latency_panel_size > 1 else {},
            calibration_tree_grid=[100], calibration_eta_grid=[0., -1.])
        environment = dict(replication_protocol_sha256=identity)
        if options.latency_panel_size > 1:
            environment.update(latency_panel_indices=[0, 1], latency_timed_panel_positions=[0, 1, 0, 1, 0])
        row = dict(n_trees=100, block_id=0, val_macro_f1=.75, coded_params=dict(x1=0.,x2=0.,x3=0.,x4=0.), latency=dict(environment=environment,
            n_warmup=3, n_timed=5, predict_raw_us=[1.]*5, inplace_raw_us=[1.]*5, predict_median_us=1.))
        low_rate = deepcopy(row); low_rate['coded_params']['x1'] = -1.
        pilot = directory/'imdb_pilot.json'
        pilot.write_text(json.dumps(dict(dataset_sha256='fixture-hash', observations=[row, low_rate])))
        (directory/'imdb_calibration.json').write_text(json.dumps(dict(dataset_sha256='fixture-hash',
            dataset_path='fixture.csv', replication_protocol_options=asdict(options), replication_protocol_sha256=identity,
            pilot_sha256=hashlib.sha256(pilot.read_bytes()).hexdigest(), selected_tree_count=100,
            latency_threshold_us=1., hv_ref_latency_us=2., calibration_fits=2)))
        config.latency_threshold_us = 1.; config.hv_ref_loss = 1.; config.hv_ref_latency_us = 2.
        (directory/'imdb_config.json').write_text(json.dumps(asdict(config)))
        return config, row, path

    def test_defaults_preserve_historical_protocol(self):
        self.assertFalse(ProtocolOptions().active)
        for value in [0, -1, 1.5]:
            with self.assertRaises(ValueError):
                ProtocolOptions(latency_panel_size=value)

    def test_capacity_threshold_uses_all_selected_capacity_rows(self):
        records = [dict(n_trees=n, latency=dict(predict_median_us=t))
                   for n, t in [(100, 100), (600, 178), (600, 182)]]
        self.assertEqual(capacity_threshold(records, 600, 'capacity_matched'), 180)
        self.assertEqual(capacity_threshold(records, 600, 'pooled'), 178)
        with self.assertRaises(ValueError):
            capacity_threshold(records, 300, 'capacity_matched')

    def test_research_panel_warmups_reload_without_weakening_core_defaults(self):
        values = {**asdict(BenchmarkConfig()), 'warmup_inferences': 128}
        options = ProtocolOptions(latency_panel_size=128)
        self.assertEqual(load_replication_config(values, options).warmup_inferences, 128)
        with self.assertRaises(ValueError):
            load_replication_config(values, ProtocolOptions())
        with self.assertRaisesRegex(ValueError, 'declared panel size'):
            load_replication_config({**values, 'warmup_inferences': 127}, options)
        with self.assertRaisesRegex(ValueError, '500 timed'):
            load_replication_config({**values, 'timed_inferences': 499}, options)

    def test_panel_covers_length_strata_without_labels(self):
        lengths = np.arange(100)
        indices = length_panel(lengths, 32)
        self.assertEqual(len(set(indices)), 32)
        deciles = np.bincount(np.asarray(indices)//10, minlength=10)
        self.assertTrue(np.all((deciles == 3) | (deciles == 4)))
        self.assertEqual(indices, length_panel(lengths, 32))
        with self.assertRaises(ValueError):
            length_panel(lengths, 101)

    def test_bigram_products_and_l2_norm(self):
        names = np.array(['good', 'film', 'good film', 'good good', 'other'])
        row = sparse.csr_matrix([[2., 3., 4., 5., 6.]])
        changed = coupled_vectors(row, names, [0, 1], np.array([[.5, .25], [1., 1.], [0., 0.]]))
        expected = np.array([[1., .75, .5, 1.25, 6.], [2., 3., 4., 5., 6.], [0., 0., 0., 0., 6.]])
        expected /= np.linalg.norm(expected, axis=1)[:, None]
        np.testing.assert_allclose(changed.toarray(), expected)
        np.testing.assert_allclose(np.linalg.norm(changed.toarray(), axis=1), 1.)
        self.assertEqual(changed.nnz, np.count_nonzero(expected))

    def test_single_document_contract_describes_historical_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, _, path = self.calibration_fixture(directory, ProtocolOptions('capacity_matched'))
            latency = json.loads(path.read_text())['latency']
            self.assertEqual(latency['selection'], 'historical first validation row')
            self.assertIn('first test row', latency['final'])

    def test_calibration_dataset_and_panel_protocol_are_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, row, path = self.calibration_fixture(directory)
            records, extended = validate_calibration(directory, 'imdb', ProtocolOptions(latency_panel_size=2), 'fixture-hash')
            self.assertTrue(extended)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0], row)
            with self.assertRaisesRegex(ValueError, 'dataset hashes differ'):
                validate_calibration(directory, 'imdb', ProtocolOptions(latency_panel_size=2), 'changed-dataset')
            with self.assertRaisesRegex(ValueError, 'same latency panel'):
                validate_calibration(directory, 'imdb', ProtocolOptions(), 'fixture-hash')
            pilot = directory/'imdb_pilot.json'
            row['latency']['environment']['latency_panel_indices'] = [2, 3]
            pilot.write_text(json.dumps(dict(dataset_sha256='fixture-hash', observations=[row])))
            with self.assertRaisesRegex(ValueError, 'frozen selection'):
                validate_calibration_records(pilot, path, 'fixture-hash')
            row['latency']['environment'].pop('replication_protocol_sha256')
            pilot.write_text(json.dumps(dict(dataset_sha256='fixture-hash', observations=[row])))
            with self.assertRaisesRegex(ValueError, 'protocol identity'):
                validate_calibration_records(pilot, path, 'fixture-hash')

    def test_completed_calibration_pilot_digest_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.calibration_fixture(directory)
            pilot = directory/'imdb_pilot.json'
            observations = json.loads(pilot.read_text())
            observations['observations'][0]['latency']['predict_median_us'] = 42.
            pilot.write_text(json.dumps(observations))
            with self.assertRaisesRegex(ValueError, 'pilot bytes differ'):
                validate_calibration(directory, 'imdb', ProtocolOptions(latency_panel_size=2), 'fixture-hash')

    def test_calibration_requires_every_frozen_tree_eta_coordinate_exactly_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, _, protocol = self.calibration_fixture(directory)
            pilot = directory/'imdb_pilot.json'
            original = json.loads(pilot.read_text())
            missing = deepcopy(original); missing['observations'].pop()
            duplicate = deepcopy(original); duplicate['observations'].append(deepcopy(duplicate['observations'][0]))
            wrong_eta = deepcopy(original); wrong_eta['observations'][0]['coded_params']['x1'] = .5
            wrong_coordinate = deepcopy(original); wrong_coordinate['observations'][0]['coded_params']['x2'] = .5
            for changed in [missing, duplicate, wrong_eta, wrong_coordinate]:
                with self.subTest(observations=changed):
                    pilot.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError, 'tree-by-eta grid'):
                        validate_calibration_records(pilot, protocol, 'fixture-hash')

    def test_calibration_summary_and_config_follow_frozen_selection_rule(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.calibration_fixture(directory)
            summary = directory/'imdb_calibration.json'
            original = json.loads(summary.read_text())
            for field, value in [('selected_tree_count', 600), ('latency_threshold_us', 42.), ('calibration_fits', 1)]:
                summary.write_text(json.dumps({**original, field: value}))
                with self.assertRaisesRegex(ValueError, 'frozen selection rule'):
                    validate_calibration(directory, 'imdb', ProtocolOptions(latency_panel_size=2), 'fixture-hash')
            summary.write_text(json.dumps(original))
            config = directory/'imdb_config.json'
            config.write_text(json.dumps({**json.loads(config.read_text()), 'n_trees': 600}))
            with self.assertRaisesRegex(ValueError, 'frozen selection rule'):
                validate_calibration(directory, 'imdb', ProtocolOptions(latency_panel_size=2), 'fixture-hash')

    def test_legacy_calibration_reuse_is_limited_to_default_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'imdb_calibration.json').write_text(json.dumps(dict(dataset_sha256='fixture-hash')))
            (directory/'imdb_pilot.json').write_text(json.dumps(dict(dataset_sha256='fixture-hash', observations=[dict(latency={})])))
            self.assertFalse(validate_calibration(directory, 'imdb', ProtocolOptions(), 'fixture-hash')[1])
            with self.assertRaisesRegex(ValueError, 'fresh calibration'):
                validate_calibration(directory, 'imdb', ProtocolOptions('capacity_matched'), 'fixture-hash')

    def test_changed_source_and_timing_calls_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, row, path = self.calibration_fixture(directory)
            pilot = directory/'imdb_pilot.json'
            row['latency']['n_timed'] = 4
            pilot.write_text(json.dumps(dict(dataset_sha256='fixture-hash', observations=[row])))
            with self.assertRaisesRegex(ValueError, 'timing calls'):
                validate_calibration_records(pilot, path, 'fixture-hash')
            contract = json.loads(path.read_text())
            contract['source_sha256']['protocol_extensions.py'] = 'changed-source'
            path.write_text(json.dumps(contract))
            with self.assertRaisesRegex(ValueError, 'source differs'):
                validate_calibration_records(pilot, path, 'fixture-hash')

    def test_sidecar_requires_identity_on_every_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            study = directory/'imdb'; study.mkdir()
            (directory/'imdb_protocol_study.json').write_text('{}')
            (study/'evaluations.jsonl').write_text(json.dumps(dict(latency=dict(environment={})))+'\n')
            with self.assertRaisesRegex(ValueError, 'no evaluation protocol identities'):
                check_protocol(study, require_words=False)

    def test_supplemental_contract_must_equal_core_config_and_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config, row, pilot_path = self.calibration_fixture(directory)
            study = directory/'imdb'; study.mkdir()
            path = directory/'imdb_protocol_study.json'
            path.write_bytes(pilot_path.read_bytes())
            identity = hashlib.sha256(path.read_bytes()).hexdigest()
            row['latency']['environment']['replication_protocol_sha256'] = identity
            (study/'evaluations.jsonl').write_text(json.dumps(row)+'\n')
            for changed in [dict(config={**asdict(config), 'latency_threshold_us': 42}, dataset_sha256='fixture-hash'),
                            dict(config=asdict(config), dataset_sha256='changed-dataset')]:
                (study/'measurement_contract.json').write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, 'contracts disagree'):
                    check_protocol(study, require_words=False)

    def test_main_run_validates_dataset_before_partition_or_evaluation(self):
        import run_phase_b
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.calibration_fixture(directory)
            with (patch.object(run_phase_b, 'load_sentiment_csv', return_value=([], np.array([]), None, 'changed-dataset')),
                  patch.object(run_phase_b, 'create_strictly_isolated_partitions') as partitions):
                with self.assertRaisesRegex(ValueError, 'dataset hashes differ'):
                    run_phase_b.run_experiment_study(directory, directory, directory/'run', device='cpu',
                        options=ProtocolOptions(latency_panel_size=2))
                partitions.assert_not_called()

    def test_panel_cycles_identically_for_both_interfaces(self):
        class Booster:
            def __init__(self):
                self.calls = {False: [], True: []}
            def set_param(self, values):
                pass
            def predict(self, row):
                self.calls[False].append(float(row[0, 0]))
            def inplace_predict(self, row):
                self.calls[True].append(float(row[0, 0]))
        profiler = PanelProfiler(sparse.csr_matrix([[10.], [20.], [30.]]), [0, 2], [1, 2, 3], 'test-protocol', 2, 5)
        booster = Booster()
        with patch('protocol_extensions.pin_cpu_affinity', return_value=True), patch('protocol_extensions.xgb.DMatrix', side_effect=lambda row, **kw: row):
            result = profiler._profile_booster_pinned(booster, None)
        self.assertEqual(booster.calls[False], [10., 30., 10., 30., 10., 30., 10.])
        self.assertEqual(booster.calls[True], booster.calls[False])
        self.assertEqual(result.environment['latency_timed_panel_positions'], [0, 1, 0, 1, 0])
        self.assertEqual(len(result.predict_raw_us), 5)
        self.assertEqual(result.batch_size, 1)

    def test_calibration_freezes_options_before_evaluation(self):
        import run_phase_b
        @dataclass
        class Latency:
            predict_median_us: float
            environment: dict
            n_warmup: int
            n_timed: int
            predict_raw_us: list
            inplace_raw_us: list
        @dataclass
        class Evaluation:
            n_trees: int
            coded_params: dict
            natural_params: dict
            val_macro_f1: float
            latency: Latency
            block_id: int
        with tempfile.TemporaryDirectory(prefix='protocol-unit-') as temporary:
            directory = Path(temporary)
            (directory/'imdb.csv').write_text('synthetic fixture only')
            output = directory/'calibration'
            options = ProtocolOptions('capacity_matched', 4, False)
            parts = SimpleNamespace(blocks=[([], np.array([]), ['x'*i for i in range(20)], np.array([]))])
            def factory(partitions, config, passed, identity):
                self.assertEqual(passed, options)
                def evaluate(point, block, seed, cfg):
                    contract = json.loads((output/'imdb_protocol_calibration.json').read_text())
                    self.assertEqual(contract['options']['threshold_mode'], 'capacity_matched')
                    self.assertEqual(contract['calibration_tree_grid'], [100, 300, 600])
                    self.assertTrue(identity)
                    median = cfg.n_trees+(1 if point.coded['x1']==0 else 3)
                    latency = Latency(median,
                        dict(replication_protocol_sha256=identity,
                            latency_panel_indices=contract['latency']['development_panels']['0'],
                            latency_timed_panel_positions=[i % 4 for i in range(cfg.timed_inferences)]),
                        cfg.warmup_inferences, cfg.timed_inferences,
                        [median]*cfg.timed_inferences, [median]*cfg.timed_inferences)
                    return Evaluation(cfg.n_trees, point.coded, dict(learning_rate=.05),
                        .7+cfg.n_trees/3000, latency, block)
                return evaluate
            with (patch.object(run_phase_b, 'load_sentiment_csv', return_value=(['fixture'], np.array([0]), None, 'fixture-hash')),
                 patch.object(run_phase_b, 'create_strictly_isolated_partitions', return_value=parts),
                 patch.object(run_phase_b, 'replication_evaluator', side_effect=factory)):
                result = run_phase_b.run_calibration(directory, output, device='cpu', options=options)
            self.assertEqual(result['selected_tree_count'], 600)
            self.assertEqual(result['latency_threshold_us'], 602)
            self.assertEqual(result['replication_protocol_options']['latency_panel_size'], 4)
            records, extended = validate_calibration(output, 'imdb', options, 'fixture-hash')
            self.assertTrue(extended)
            self.assertEqual(len(records), 6)

    def test_default_calibration_rejects_existing_evidence_before_writes_or_training(self):
        import run_phase_b
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            evidence = directory/'calibration_contract.json'
            evidence.write_bytes(b'preserved published calibration')
            with (patch.object(run_phase_b, 'load_sentiment_csv') as load,
                  patch.object(run_phase_b, 'resolve_device') as resolve,
                  patch.object(run_phase_b, 'replication_evaluator') as evaluate):
                with self.assertRaisesRegex(FileExistsError, 'fresh output directory'):
                    run_phase_b.run_calibration(directory, directory, device='cpu')
                load.assert_not_called()
                resolve.assert_not_called()
                evaluate.assert_not_called()
            self.assertEqual(evidence.read_bytes(), b'preserved published calibration')
            self.assertEqual(list(directory.iterdir()), [evidence])

    def test_final_panel_access_occurs_inside_existing_gate(self):
        class Gate:
            frozen = False
            inside = False
            def freeze(self):
                self.frozen = True
            @property
            def _texts(self):
                if not self.inside:
                    raise ValueError('Premature final-text access')
                return ['a', 'bb', 'ccc', 'dddd']
            def evaluate(self, contract, output, callback):
                if not self.frozen:
                    raise ValueError('Selection not frozen')
                self.inside = True
                return callback(None, None, sparse.csr_matrix(np.ones((4, 2))), None,
                    dict(method='doe', replicate=0, coded_params={}),
                    dict(warmup_inferences=4,timed_inferences=4,n_trees=1,random_seed=42,training_threads=1,device='cpu'))
        @dataclass
        class Result:
            latency: dict
        original = Gate()
        final = GatedPanelFinalEvaluator(original, ProtocolOptions(latency_panel_size=4), 'protocol-id')
        with self.assertRaises(ValueError):
            final.evaluate(None, None, None)
        final.freeze()
        with patch('protocol_extensions.train_and_evaluate_config', return_value=Result(dict(environment={}))):
            result = final.evaluate(None, None, None)
        self.assertEqual(result['latency']['environment']['replication_protocol_sha256'], 'protocol-id')


if __name__ == '__main__':
    unittest.main()
