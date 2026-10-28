"""Canonical sparse storage at XGBoost boundaries, without changing feature values."""
from scipy import sparse


def canonical_features(values):
    if not sparse.issparse(values):
        return values
    if values.format == 'csr' and values.has_canonical_format:
        return values
    result = sparse.csr_matrix(values, copy=True)
    result.sum_duplicates()
    result.sort_indices()
    return result
