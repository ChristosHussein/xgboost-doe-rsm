"""Local RSM and exact SHAP for specified word-feature perturbations.

These conditional changes to TF-IDF weights do not imply causal word effects
or that fractionally attenuated vectors are natural sentences.
"""
from dataclasses import asdict
import numpy as np
from .features import canonical_features

from .design import generate_fccd_points
from .surrogates import PolynomialRSMSurrogate
from .xai import compute_surrogate_shap_attributions, extract_anova_attributions


def explain_word_response(response, words, *, seed=42, confirmation_points=64):
    """Fit on FCCD perturbations and validate on separate uniform perturbations."""
    if len(words) != 4 or len(set(words)) != 4 or not all(words):
        raise ValueError('Specify four distinct word-feature names')
    if confirmation_points < 1:
        raise ValueError('Confirmation budget must be positive')
    coded = np.array([p.vector for p in generate_fccd_points(n_center=2)])
    weights = (coded+1)/2
    observed = np.asarray(response(weights), dtype=float)
    if observed.shape != (len(weights),) or not np.isfinite(observed).all():
        raise ValueError('Oracle must return one finite response per perturbation')
    surface = PolynomialRSMSurrogate(False).fit(coded, observed)
    rng = np.random.default_rng(seed)
    confirmation = rng.uniform(0, 1, (confirmation_points, 4))
    actual = np.asarray(response(confirmation), dtype=float)
    predicted = surface.predict(2*confirmation-1)
    if actual.shape != predicted.shape or not np.isfinite(actual).all():
        raise ValueError('Invalid independent perturbation observations')
    class Oracle:
        def predict(self, x):
            return response(x)
    shap = compute_surrogate_shap_attributions(Oracle(), np.zeros((1,4)), np.ones((1,4)), words)
    effects = asdict(extract_anova_attributions(surface))
    effects['feature_names'] = list(words)
    effects['rank_defined']=bool(np.ptp(list(effects['main_effect_ss'].values()))>1e-12)
    if not effects['rank_defined']:
        effects['main_effect_ranking']=[]
    return dict(scope='conditional feature-weight perturbations; not causal word effects',
                words=list(words), seed=seed,
                training=dict(weights=weights.tolist(), observed=observed.tolist()),
                rsm={**effects, 'coefficients':surface.beta_.tolist(),
                     'interaction_matrix_ss':effects['interaction_matrix_ss'].tolist()},
                shap={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in asdict(shap).items()},
                confirmation=dict(weights=confirmation.tolist(), observed=actual.tolist(),
                    predicted=predicted.tolist(), rmse=float(np.sqrt(np.mean((actual-predicted)**2))),
                    max_absolute_error=float(np.max(np.abs(actual-predicted)))))


def explain_document(model, vectorizer, text, *, seed=42, confirmation_points=64):
    """Select the strongest present unigram weights without looking at the label."""
    from scipy import sparse
    row = vectorizer.transform([text]).tocsr()
    names = vectorizer.get_feature_names_out()
    return explain_feature_vector(model,row,names,seed=seed,
                                  confirmation_points=confirmation_points)


def explain_feature_vector(model,row,names,*,seed=42,confirmation_points=64):
    """Replay an explanation from a saved sparse vector and vocabulary."""
    from scipy import sparse
    names=np.asarray(names)
    present = [i for i in row.indices if ' ' not in names[i]]
    selected = sorted(present, key=lambda i:(-row[0,i], names[i]))[:4]
    if len(selected) != 4:
        raise ValueError('Document needs four nonzero unigram features')
    def response(weights):
        vectors = sparse.vstack([row]*len(weights), format='lil')
        for j, index in enumerate(selected):
            vectors[:,index] = weights[:,j,None]*row[0,index]
        return model.predict_proba(canonical_features(vectors.tocsr()))[:,1]
    result = explain_word_response(response, names[selected].tolist(), seed=seed,
                                   confirmation_points=confirmation_points)
    result['selected_feature_indices'] = selected
    return result
