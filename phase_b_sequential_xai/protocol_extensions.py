"""Opt-in replication protocols, separate from the frozen published implementation.

These options define new measurements. They never reinterpret archived timings
or turn feature-weight perturbations into causal language edits.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
import platform
import time

import numpy as np
from scipy import sparse
import xgboost as xgb

from xgb_doe_xai.benchmarks import train_and_evaluate_config
from xgb_doe_xai.config import BenchmarkConfig
from xgb_doe_xai.execution import make_development_evaluator, final_classification
from xgb_doe_xai.features import canonical_features
from xgb_doe_xai.profiler import WarmedLatencyProfiler, LatencyProfileResult, pin_cpu_affinity, _compute_iqm


@dataclass(frozen=True)
class ProtocolOptions:
    threshold_mode: str = 'pooled'
    latency_panel_size: int = 1
    consistent_tfidf_perturbations: bool = False

    def __post_init__(self):
        if self.threshold_mode not in ('pooled', 'capacity_matched'):
            raise ValueError('Unknown calibration threshold mode')
        if type(self.latency_panel_size) is not int or not 1 <= self.latency_panel_size <= 500:
            raise ValueError('Latency panel size must be an integer between 1 and 500')

    @property
    def active(self):
        return self != ProtocolOptions()


def capacity_threshold(records, selected_trees, mode):
    if mode not in ('pooled', 'capacity_matched'):
        raise ValueError('Unknown threshold mode')
    eligible = [r for r in records if mode == 'pooled' or r['n_trees'] == selected_trees]
    if not eligible:
        raise ValueError('No calibration observations at the selected capacity')
    return float(np.median([r['latency']['predict_median_us'] for r in eligible]))


def load_replication_config(values, options):
    """Retain core validation while explicitly extending research panel warmups."""
    values = dict(values)
    warmups = values.get('warmup_inferences', 100)
    extended_warmups = values.get('mode', 'research') == 'research' and options.latency_panel_size > 100
    if extended_warmups:
        if warmups != options.latency_panel_size:
            raise ValueError('Research panel warmups differ from the declared panel size')
        values['warmup_inferences'] = 100  # Validate every historical core constraint.
    config = BenchmarkConfig(**values)
    if extended_warmups:
        config.warmup_inferences = warmups  # Supplemental pre-run contract declares the extension.
    return config


def length_panel(lengths, size):
    """Allocate nearly equally across equal-count length strata; no labels used."""
    lengths = np.asarray(lengths)
    if type(size) is not int or size < 1 or size > len(lengths):
        raise ValueError('Panel size must fit the available validation documents')
    ordered = np.lexsort((np.arange(len(lengths)), lengths))
    strata = np.array_split(ordered, min(10, len(ordered)))
    allocation = np.zeros(len(strata), dtype=int)
    order = np.linspace(0, len(strata)-1, min(size, len(strata)), dtype=int)
    remaining = size
    while remaining:
        for index in order:
            if allocation[index] < len(strata[index]):
                allocation[index] += 1
                remaining -= 1
                if not remaining:
                    break
    return [int(chunk[len(chunk)//2]) for index, count in enumerate(allocation)
            if count for chunk in np.array_split(strata[index], count)]


def freeze_protocol(path, options, config, dataset_sha256, *, panels=None, calibration_files=(), calibration_tree_grid=()):
    """Create an exclusive protocol/source record before any classifier evaluation."""
    from datetime import datetime, timezone
    path = Path(path)
    sources = [Path(__file__), Path(__file__).with_name('run_phase_b.py')]
    payload = dict(protocol='optional-replication-v1', declared_at_utc=datetime.now(timezone.utc).isoformat(),
        options=asdict(options), training_config=asdict(config), dataset_sha256=dataset_sha256,
        calibration_tree_grid=list(calibration_tree_grid),
        source_sha256={p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sources},
        calibration_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in calibration_files},
        calibration_paths={Path(p).name: Path(p).resolve().relative_to(path.parent.resolve()).as_posix() for p in calibration_files},
        latency=dict(unit='microseconds per single already-vectorized document',
            selection=('label-free equal-count text-length decile strata; midpoint within each allocated stratum sub-bin'
                       if options.latency_panel_size > 1 else 'historical first validation row'),
            length_unit='Unicode character count', weighting='cyclic panel order, each timed interface starts at panel index zero',
            primary='fresh DMatrix construction plus Booster.predict; uncached',
            warmup='cycle panel for declared warmups; at least one per panel member',
            final=('same selection rule applied inside the frozen final-evaluation gate'
                   if options.latency_panel_size > 1 else 'historical first test row inside the frozen final-evaluation gate'),
            development_panels=panels or {}),
        perturbation=('multiply selected unigrams by their weights; multiply active bigrams by the product of selected-token weights, including repeated tokens; omit zero-valued terms; L2 renormalize; zero vectors remain zero'
                      if options.consistent_tfidf_perturbations else 'historical independent unigram attenuation without renormalization'),
        scope='new replication protocol; conditional feature perturbations, not natural sentences or causal effects')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel_indices(partitions, size):
    return {str(b): length_panel([len(t) for t in block[2]], size)
            for b, block in enumerate(partitions.blocks)}


def snapshot_calibration(files, destination):
    """Retain exact calibration input bytes beside a new study's protocol."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    result = []
    for source in files:
        target = destination/Path(source).name
        with target.open('xb') as stream:
            stream.write(Path(source).read_bytes())
        result.append(target)
    return result


def _check_sources(contract):
    if set(contract['source_sha256']) != {'protocol_extensions.py', 'run_phase_b.py'}:
        raise ValueError('Extended replication source inventory differs')
    for name, expected in contract['source_sha256'].items():
        source = Path(__file__).with_name(name).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError('Extended replication source differs: '+name)


def _check_panel_records(records, contract, identity):
    size = contract['options']['latency_panel_size']
    for row in records:
        latency = row['latency']; environment = latency.get('environment', {})
        if environment.get('replication_protocol_sha256') != identity:
            raise ValueError('An extended evaluation omits or changes its protocol identity')
        config = contract['training_config']
        if (latency['n_warmup'] != config['warmup_inferences'] or latency['n_timed'] != config['timed_inferences']
                or len(latency['predict_raw_us']) != latency['n_timed']
                or len(latency['inplace_raw_us']) != latency['n_timed']):
            raise ValueError('Recorded timing calls differ from the frozen protocol')
        if size > 1:
            indices = environment.get('latency_panel_indices', [])
            if (len(indices) != size or len(set(indices)) != size
                    or environment.get('latency_timed_panel_positions') != [i % size for i in range(latency['n_timed'])]
                    or latency['n_warmup'] < size or latency['n_timed'] < size):
                raise ValueError('Recorded latency panel differs from the declared cyclic protocol')
            if row['block_id'] >= 0 and indices != contract['latency']['development_panels'][str(row['block_id'])]:
                raise ValueError('Recorded validation panel differs from the frozen selection')


def validate_calibration_records(pilot_path, protocol_path, dataset_sha256):
    """Validate prospective calibration identities before using their threshold."""
    pilot_path, protocol_path = Path(pilot_path), Path(protocol_path)
    pilot = json.loads(pilot_path.read_text())
    contract = json.loads(protocol_path.read_text())
    if pilot['dataset_sha256'] != dataset_sha256 or contract['dataset_sha256'] != dataset_sha256:
        raise ValueError('Calibration observations belong to a different dataset')
    identity = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    _check_sources(contract)
    _check_panel_records(pilot['observations'], contract, identity)
    if set(r['n_trees'] for r in pilot['observations']) != set(contract['calibration_tree_grid']):
        raise ValueError('Calibration capacities differ from the frozen grid')
    for name, expected in contract['calibration_sha256'].items():
        source = protocol_path.parent/contract['calibration_paths'][name]
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError('Calibration plan differs from its frozen protocol')
    return pilot['observations']


def validate_calibration(directory, dataset_name, options, dataset_sha256):
    directory = Path(directory)
    metadata = json.loads((directory/f'{dataset_name}_calibration.json').read_text())
    pilot_path = directory/f'{dataset_name}_pilot.json'
    pilot = json.loads(pilot_path.read_text())
    if metadata['dataset_sha256'] != dataset_sha256 or pilot['dataset_sha256'] != dataset_sha256:
        raise ValueError('Calibration and study dataset hashes differ; recalibration is required')
    protocol_path = directory/f'{dataset_name}_protocol_calibration.json'
    if not protocol_path.exists():
        if (options.active or metadata.get('replication_protocol_options')
                or any(r['latency'].get('environment', {}).get('replication_protocol_sha256') for r in pilot['observations'])):
            raise ValueError('Opt-in replications require a fresh calibration with a source protocol')
        return pilot['observations'], False
    identity = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if metadata.get('replication_protocol_sha256') != identity:
        raise ValueError('Calibration summary omits or changes its source protocol identity')
    contract = json.loads(protocol_path.read_text())
    if metadata.get('replication_protocol_options') != contract['options']:
        raise ValueError('Calibration summary options differ from their frozen protocol')
    if contract['options']['latency_panel_size'] != options.latency_panel_size:
        raise ValueError('Recalibrate using the same latency panel size as the main experiment')
    return validate_calibration_records(pilot_path, protocol_path, dataset_sha256), True


class PanelProfiler(WarmedLatencyProfiler):
    def __init__(self, features, indices, lengths, protocol_sha256, n_warmup, n_timed):
        super().__init__(n_warmup, n_timed)
        if n_warmup < len(indices) or n_timed < len(indices):
            raise ValueError('Warmups and timed calls must cover every panel document')
        self.rows = [canonical_features(features[i:i+1]) for i in indices]
        self.indices = list(indices)
        self.lengths = [int(lengths[i]) for i in indices]
        self.protocol_sha256 = protocol_sha256

    def _profile_booster_pinned(self, booster, sample_input):
        core = min(__import__('os').sched_getaffinity(0)) if hasattr(__import__('os'), 'sched_getaffinity') else 0
        applied = pin_cpu_affinity(core)
        if not applied:
            raise RuntimeError('Required single-core affinity could not be applied')
        booster.set_param({'nthread': 1, 'device': 'cpu'})
        def measure(inplace):
            def predict(i):
                row = self.rows[i % len(self.rows)]
                return booster.inplace_predict(row) if inplace else booster.predict(xgb.DMatrix(row, nthread=1))
            for i in range(self.n_warmup):
                predict(i)
            timings = []
            for i in range(self.n_timed):
                started = time.perf_counter_ns(); predict(i)
                timings.append((time.perf_counter_ns()-started)/1000)
            return np.asarray(timings)
        classic, inplace = measure(False), measure(True)
        return LatencyProfileResult(
            predict_median_us=float(np.median(classic)), predict_mean_us=float(classic.mean()),
            predict_iqm_us=_compute_iqm(classic), predict_sd_us=float(classic.std(ddof=1)),
            predict_p95_us=float(np.percentile(classic, 95)), inplace_median_us=float(np.median(inplace)),
            inplace_mean_us=float(inplace.mean()), inplace_iqm_us=_compute_iqm(inplace),
            n_warmup=self.n_warmup, n_timed=self.n_timed, batch_size=1,
            predict_raw_us=classic.tolist(), inplace_raw_us=inplace.tolist(), affinity_applied=applied, cpu_core=core,
            predict_operation='fresh DMatrix construction plus Booster.predict; uncached cyclic document panel',
            environment=dict(platform=platform.platform(), processor=platform.processor(), xgboost=xgb.__version__,
                clock='perf_counter_ns', replication_protocol_sha256=self.protocol_sha256,
                latency_panel_indices=self.indices, latency_panel_character_lengths=self.lengths,
                latency_timed_panel_positions=[i % len(self.rows) for i in range(self.n_timed)]))


def evaluator(partitions, config, options, protocol_sha256):
    legacy = make_development_evaluator(partitions, config)
    if not options.active and protocol_sha256 is None:
        return legacy
    indices = panel_indices(partitions, options.latency_panel_size) if options.latency_panel_size > 1 else None
    def evaluate(point, block, seed, cfg):
        if indices is None:
            result = legacy(point, block, seed, cfg)
        else:
            train, labels, valid, truth = partitions.block_features[block]
            profiler = PanelProfiler(valid, indices[str(block)], [len(t) for t in partitions.blocks[block][2]],
                                     protocol_sha256, cfg.warmup_inferences, cfg.timed_inferences)
            result = train_and_evaluate_config(point.coded, train, labels, valid, truth, profiler,
                point.point_id, block, cfg.n_trees, seed, cfg.training_threads, cfg.device)
        environment = {**result.latency.environment, 'replication_protocol_sha256': protocol_sha256}
        return replace(result, latency=replace(result.latency, environment=environment))
    return evaluate


class GatedPanelFinalEvaluator:
    def __init__(self, original, options, protocol_sha256):
        self.original, self.options, self.protocol_sha256 = original, options, protocol_sha256

    def __getattr__(self, name):
        return getattr(self.original, name)

    def evaluate(self, contract_path, output_path, evaluate_fn):
        def final(train, labels, test, truth, selection, cfg):
            if self.options.latency_panel_size == 1:
                result = final_classification(train, labels, test, truth, selection, cfg)
            else:
                # Called only after the original evaluator validates frozen selections.
                lengths = [len(t) for t in self.original._texts]
                indices = length_panel(lengths, self.options.latency_panel_size)
                profiler = PanelProfiler(test, indices, lengths, self.protocol_sha256,
                                         cfg['warmup_inferences'], cfg['timed_inferences'])
                result = dict(method=selection['method'], replicate=selection['replicate'], **asdict(
                    train_and_evaluate_config(selection['coded_params'], train, labels, test, truth, profiler,
                        f"FINAL_{selection['method']}_{selection['replicate']}", -1, cfg['n_trees'],
                        cfg['random_seed']+900000+selection['replicate'], cfg['training_threads'], cfg['device'])))
            result['latency']['environment']['replication_protocol_sha256'] = self.protocol_sha256
            return result
        return self.original.evaluate(contract_path, output_path, final)


def coupled_vectors(row, names, selected, weights):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or weights.shape[1] != len(selected) or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError('Perturbation weights must lie in the unit cube')
    selected_words = {names[index]: j for j, index in enumerate(selected)}
    vectors = sparse.vstack([row]*len(weights), format='lil')
    for index in row.indices:
        positions = [selected_words[token] for token in names[index].split() if token in selected_words]
        if positions:
            vectors[:, index] = np.prod(weights[:, positions], axis=1)[:, None]*row[0, index]
    vectors = canonical_features(vectors.tocsr())
    vectors.eliminate_zeros()
    norms = np.sqrt(np.asarray(vectors.multiply(vectors).sum(axis=1)).ravel())
    scales = np.divide(1., norms, out=np.zeros_like(norms), where=norms > 0)
    vectors.data *= np.repeat(scales, np.diff(vectors.indptr))
    return vectors


def explain_consistent_vector(model, row, names, *, seed=42, confirmation_points=64):
    from xgb_doe_xai.local_explanations import explain_word_response
    names = np.asarray(names)
    present = [i for i in row.indices if ' ' not in names[i]]
    selected = sorted(present, key=lambda i: (-row[0, i], names[i]))[:4]
    if len(selected) != 4:
        raise ValueError('Document needs four nonzero unigram features')
    response = lambda weights: model.predict_proba(coupled_vectors(row, names, selected, weights))[:, 1]
    result = explain_word_response(response, names[selected].tolist(), seed=seed, confirmation_points=confirmation_points)
    result['selected_feature_indices'] = selected
    return result


def run_consistent_word_study(partitions, config, selection, output, protocol_sha256, *, documents=10):
    from xgb_doe_xai.config import uncode_configuration
    from xgb_doe_xai.study import write_json
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write_json(output/'contract.json', dict(selection=selection, config=asdict(config), training_fits=1,
        replication_protocol_sha256=protocol_sha256, document_limit=documents, confirmation_points=64,
        explanation_prediction_device='cpu',
        document_rule='first eligible documents in development validation block zero; labels unused',
        perturbation='coupled unigram and bigram attenuation followed by L2 normalization',
        scope='conditional feature-weight edits; not natural-language or causal counterfactuals'))
    train, labels, valid, _ = partitions.block_features[0]
    train, valid = canonical_features(train), canonical_features(valid)
    model = xgb.XGBClassifier(**uncode_configuration(selection['coded_params']), n_estimators=config.n_trees,
        n_jobs=config.training_threads, device=config.device, random_state=config.random_seed, objective='binary:logistic')
    started = time.perf_counter(); model.fit(train, labels); duration = time.perf_counter()-started
    model.get_booster().set_param({'nthread': 1, 'device': 'cpu'}); model.save_model(output/'classifier.ubj')
    names = partitions.vectorizers[0].get_feature_names_out()
    results, indices = [], []
    for i in range(valid.shape[0]):
        if sum(' ' not in names[j] for j in valid[i].indices) < 4:
            continue
        result = explain_consistent_vector(model, valid[i], names, seed=config.random_seed+len(results))
        result['validation_document_index'] = i; results.append(result); indices.append(i)
        if len(results) == documents:
            break
    if not results:
        raise ValueError('No eligible documents for coupled perturbations')
    sparse.save_npz(output/'inputs.npz', valid[indices]); write_json(output/'feature_names.json', names.tolist())
    write_json(output/'observations.json', dict(training_seconds=duration, documents=results))
    write_json(output/'registry.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    check_consistent_word_study(output)


def check_consistent_word_study(output):
    from xgb_doe_xai.word_study import equivalent_reconstruction
    output = Path(output)
    expected = json.loads((output/'registry.json').read_text())
    if expected != {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file() and p.name != 'registry.json'}:
        raise ValueError('Coupled word evidence integrity mismatch')
    booster = xgb.Booster(); booster.load_model(output/'classifier.ubj'); booster.set_param({'nthread': 1, 'device': 'cpu'})
    class Oracle:
        def predict_proba(self, values):
            probability = booster.predict(xgb.DMatrix(values, nthread=1))
            return np.column_stack([1-probability, probability])
    vectors = sparse.load_npz(output/'inputs.npz'); names = json.loads((output/'feature_names.json').read_text())
    documents = json.loads((output/'observations.json').read_text())['documents']
    if vectors.shape[0] != len(documents):
        raise ValueError('Coupled explanation document inventory differs')
    for row, original in zip(vectors, documents):
        replayed = explain_consistent_vector(Oracle(), row, names, seed=original['seed'], confirmation_points=len(original['confirmation']['weights']))
        if not equivalent_reconstruction(replayed, {k: v for k, v in original.items() if k != 'validation_document_index'}):
            raise ValueError('Coupled word explanation replay differs')


def check_protocol(study_output, *, require_words=True):
    """Validate the supplemental protocol linked by every extended evaluation."""
    study_output = Path(study_output)
    records = [json.loads(line) for line in (study_output/'evaluations.jsonl').read_text().splitlines()]
    final = study_output/'final_results.json'
    if final.exists():
        records += json.loads(final.read_text())['results']
    filename = 'smoke_protocol.json' if study_output.name == 'smoke_study' else study_output.name+'_protocol_study.json'
    path = study_output.parent/filename
    identities = {r['latency']['environment'].get('replication_protocol_sha256') for r in records}
    if identities == {None}:
        if path.exists():
            raise ValueError('An extended study has a sidecar but no evaluation protocol identities')
        return  # Historical observations have no extension contract.
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    if identities != {expected}:
        raise ValueError('Supplemental protocol identity differs or some evaluations omit it')
    contract = json.loads(path.read_text())
    core = json.loads((study_output/'measurement_contract.json').read_text())
    if contract['training_config'] != core['config'] or contract['dataset_sha256'] != core['dataset_sha256']:
        raise ValueError('Supplemental and core measurement contracts disagree')
    _check_sources(contract)
    _check_panel_records(records, contract, expected)
    for name, value in contract['calibration_sha256'].items():
        target = path.parent/contract['calibration_paths'][name]
        if hashlib.sha256(target.read_bytes()).hexdigest() != value:
            raise ValueError('Retained calibration evidence differs: '+name)
    options = ProtocolOptions(**contract['options'])
    if study_output.name == 'smoke_study':
        validate_calibration_records(path.parent/'smoke_calibration.json',
            path.parent/'smoke_protocol_calibration.json', core['dataset_sha256'])
    else:
        validate_calibration(path.parent/(study_output.name+'_calibration_inputs'),
            study_output.name, options, core['dataset_sha256'])
    if require_words and contract['options']['consistent_tfidf_perturbations']:
        name = 'smoke_words' if study_output.name == 'smoke_study' else study_output.name+'_words'
        words = study_output.parent/name
        if json.loads((words/'contract.json').read_text())['replication_protocol_sha256'] != expected:
            raise ValueError('Coupled word study uses a different replication protocol')
        check_consistent_word_study(words)
