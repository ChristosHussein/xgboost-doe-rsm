"""Replay the saved word experiment at float32 prediction precision.

Probabilities may differ by a few float32 rounding units between C runtimes.
The saved observations remain authoritative. Refit their quadratic surfaces
independently and require the published RMSE precision to survive model replay.
"""
from pathlib import Path
from itertools import combinations, permutations
from math import factorial
import hashlib
import json

import numpy as np
from scipy import sparse
import xgboost as xgb


def verify_words(root: Path):
    folder = root/'results/phase_b/imdb_words'
    registry = json.loads((folder/'registry.json').read_text())
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in folder.iterdir() if p.is_file() and p.name != 'registry.json'}
    if actual != registry:
        raise ValueError('Local word evidence checksum mismatch')
    model = xgb.Booster()
    model.load_model(folder/'classifier.ubj')
    model.set_param({'nthread': 1, 'device': 'cpu'})
    vectors = sparse.load_npz(folder/'inputs.npz')
    names = np.asarray(json.loads((folder/'feature_names.json').read_text()))
    documents = json.loads((folder/'observations.json').read_text())['documents']
    if vectors.shape[0] != len(documents):
        raise ValueError('Local document inventory differs')
    epsilon = np.finfo(np.float32).eps
    max_ulp = 0

    def matrix(weights):
        x = 2*np.asarray(weights)-1
        return np.column_stack([np.ones(len(x)), *x.T, *(x*x).T,
                                *[x[:, i]*x[:, j] for i, j in combinations(range(4), 2)]])

    def probability_check(expected, observed):
        nonlocal max_ulp
        expected = np.asarray(expected, dtype=np.float32)
        observed = np.asarray(observed, dtype=np.float32)
        if expected.shape != observed.shape or not np.isfinite(observed).all() or np.any((observed < 0) | (observed > 1)):
            raise ValueError('Invalid replayed classifier probabilities')
        ulp = np.abs(expected.view(np.uint32).astype(np.int64)-observed.view(np.uint32).astype(np.int64))
        max_ulp = max(max_ulp, int(ulp.max()))
        if np.any(ulp > 2):
            raise ValueError(f'Word classifier replay exceeds two float32 ULPs: {max_ulp}')

    for row, document in zip(vectors, documents):
        present = [j for j in row.indices if ' ' not in names[j]]
        selected = sorted(present, key=lambda j: (-row[0, j], names[j]))[:4]
        if selected != document['selected_feature_indices'] or names[selected].tolist() != document['words']:
            raise ValueError('Word selection does not match the saved input vector')
        def predict(weights):
            changed = sparse.vstack([row]*len(weights), format='lil')
            for j, index in enumerate(selected):
                changed[:, index] = np.asarray(weights)[:, j, None]*row[0, index]
            changed = changed.tocsr(); changed.sum_duplicates(); changed.sort_indices()
            return model.predict(xgb.DMatrix(changed, nthread=1))
        training = document['training']; confirmation = document['confirmation']
        observed = predict(training['weights'])
        reserved = predict(confirmation['weights'])
        probability_check(training['observed'], observed)
        probability_check(confirmation['observed'], reserved)
        X = matrix(training['weights']); Z = matrix(confirmation['weights'])
        beta = np.linalg.lstsq(X, training['observed'], rcond=None)[0]
        predicted = Z@beta
        if not np.allclose(beta, document['rsm']['coefficients'], atol=1e-10, rtol=1e-10) or not np.allclose(predicted, confirmation['predicted'], atol=1e-10, rtol=1e-10):
            raise ValueError('Saved local quadratic surface does not reconstruct')
        replayed = Z@np.linalg.lstsq(X, observed, rcond=None)[0]
        error = np.sqrt(np.mean((reserved-replayed)**2))
        if format(error, '.4f') != format(confirmation['rmse'], '.4f'):
            raise ValueError('Model replay changes the published local-fidelity precision')
        coalitions = np.asarray([[(mask >> j) & 1 for j in range(4)] for mask in range(16)])
        values = predict(coalitions).astype(float)
        shap = np.zeros(4)
        for order in permutations(range(4)):
            mask = 0
            for j in order:
                shap[j] += (values[mask | (1 << j)]-values[mask])/factorial(4)
                mask |= 1 << j
        interactions = np.zeros((4, 4))
        for i, j in combinations(range(4), 2):
            for mask in range(16):
                if mask & ((1 << i) | (1 << j)):
                    continue
                size = mask.bit_count()
                weight = factorial(size)*factorial(2-size)/(2*factorial(3))
                interactions[i, j] += weight*(values[mask | (1 << i) | (1 << j)]-values[mask | (1 << i)]-values[mask | (1 << j)]+values[mask])
            interactions[j, i] = interactions[i, j]
        np.fill_diagonal(interactions, shap-interactions.sum(axis=1))
        saved_shap = np.asarray(document['shap']['raw_shap_values'])[0]
        saved_interactions = np.asarray(document['shap']['raw_interaction_values'])[0]
        bounds = np.full((4, 4), 4*epsilon); np.fill_diagonal(bounds, 16*epsilon)
        if np.any(np.abs(saved_shap-shap) > 4*epsilon) or np.any(np.abs(saved_interactions-interactions) > bounds):
            raise ValueError('Word Shapley values exceed propagated float32 roundoff bounds')
    print(f'Word model replay: {len(documents)} documents; maximum probability difference {max_ulp} float32 ULPs.')
    return dict(documents=len(documents), probability_ulp_bound=2, published_rmse_decimals=4)
