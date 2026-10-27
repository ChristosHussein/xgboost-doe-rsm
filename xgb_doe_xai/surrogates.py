"""xgb_doe_xai.surrogates - Parametric 2nd-order RSM and non-parametric surrogates.

Provides partial-SS ANOVA, within-block replicated-cell lack-of-fit tests,
fixed-block OLS prediction intervals, and flexible surrogate modeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


@dataclass
class AnovaTableResult:
    source: List[str]
    df: List[int]
    sum_sq: List[float]
    mean_sq: List[float]
    f_stat: List[Optional[float]]
    p_value: List[Optional[float]]
    r2: float
    adj_r2: float
    ss_lof: Optional[float] = None
    ss_pe: Optional[float] = None
    f_lof: Optional[float] = None
    p_lof: Optional[float] = None

    def to_dataframe(self) -> pd.DataFrame:
        data = {
            "Source": self.source,
            "DF": self.df,
            "Sum of Squares": self.sum_sq,
            "Mean Square": self.mean_sq,
            "F Value": [f"{f:.3f}" if f is not None else "-" for f in self.f_stat],
            "p-value": [f"{p:.4e}" if p is not None else "-" for p in self.p_value],
        }
        return pd.DataFrame(data)


class PolynomialRSMSurrogate:
    """Second-order Response Surface model with ANOVA decomposition and Lack-of-Fit testing."""

    def __init__(self, include_blocks: bool = True) -> None:
        self.include_blocks = include_blocks
        self.factors = ["x1", "x2", "x3", "x4"]
        self.feature_names: List[str] = []
        self.beta_: Optional[np.ndarray] = None
        self.xtx_inv_: Optional[np.ndarray] = None
        self.mse_resid_: float = 0.0
        self.df_resid_: int = 0
        self.anova_: Optional[AnovaTableResult] = None
        self.block_var_: float = 0.0
        self.block_levels_ = np.array([])

    def _build_design_matrix(
        self,
        X_coded: np.ndarray,
        block_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Construct [1, x1..x4, x1^2..x4^2, x1x2..x3x4, blocks...] design matrix."""
        n = X_coded.shape[0]
        cols = [np.ones((n, 1))]  # Intercept
        names = ["(Intercept)"]

        # Linear terms
        for i, f in enumerate(self.factors):
            cols.append(X_coded[:, i : i + 1])
            names.append(f)

        # Pure quadratic terms
        for i, f in enumerate(self.factors):
            cols.append((X_coded[:, i : i + 1] ** 2))
            names.append(f"{f}^2")

        # 2-factor interaction terms
        for i in range(len(self.factors)):
            for j in range(i + 1, len(self.factors)):
                cols.append(X_coded[:, i : i + 1] * X_coded[:, j : j + 1])
                names.append(f"{self.factors[i]}:{self.factors[j]}")

        # Block indicators (sum-to-zero or dummy coding if present)
        if self.include_blocks and block_ids is not None:
            unique_blocks = np.unique(block_ids)
            for b in unique_blocks[1:]:
                b_col = (block_ids == b).astype(float).reshape(-1, 1)
                cols.append(b_col)
                names.append(f"Block_{b}")

        if not getattr(self, '_fitted', False):
            self.feature_names = names
        return np.hstack(cols)

    def fit(
        self,
        X_coded: np.ndarray,
        y: np.ndarray,
        block_ids: Optional[np.ndarray] = None,
    ) -> "PolynomialRSMSurrogate":
        """Fit OLS 2nd-order response surface and compute complete ANOVA decomposition."""
        self._fitted = False
        X_coded, y = np.asarray(X_coded, dtype=float), np.asarray(y, dtype=float)
        if X_coded.ndim != 2 or X_coded.shape[1] != 4 or y.shape != (len(X_coded),):
            raise ValueError('Expected an n-by-4 design and n responses')
        if not np.isfinite(X_coded).all() or not np.isfinite(y).all():
            raise ValueError('Design and responses must be finite')
        self.block_levels_ = np.unique(block_ids) if self.include_blocks and block_ids is not None else np.array([])
        X_mat = self._build_design_matrix(X_coded, block_ids)
        n, p = X_mat.shape

        # Solve OLS via SVD / pseudo-inverse for numerical stability
        beta, residuals, rank, s = np.linalg.lstsq(X_mat, y, rcond=1e-10)
        if rank != p or n <= rank:
            raise ValueError('Surface requires full column rank and positive residual degrees of freedom')
        self.beta_ = beta
        self.xtx_inv_ = np.linalg.pinv(X_mat.T @ X_mat)

        y_pred = X_mat @ beta
        e = y - y_pred
        ss_total = float(np.sum((y - np.mean(y)) ** 2))
        ss_resid = float(np.sum(e ** 2))
        ss_model = ss_total - ss_resid
        df_total = n - 1
        df_model = p - 1
        df_resid = n - p

        self.mse_resid_ = ss_resid / max(df_resid, 1)
        self.df_resid_ = df_resid

        # ANOVA partition: Main effects, Interactions, Curvature, Blocks
        # Compute subset model SS:
        idx_main = [i for i, name in enumerate(self.feature_names) if name in self.factors]
        idx_quad = [i for i, name in enumerate(self.feature_names) if "^2" in name]
        idx_inter = [i for i, name in enumerate(self.feature_names) if ":" in name]
        idx_block = [i for i, name in enumerate(self.feature_names) if "Block_" in name]

        # Partial SS via projection:
        def _partial_ss(indices: List[int]) -> float:
            if not indices:
                return 0.0
            # Reduced matrix excluding indices
            keep = [i for i in range(p) if i not in indices]
            X_red = X_mat[:, keep]
            b_red = np.linalg.lstsq(X_red, y, rcond=1e-10)[0]
            ss_red_resid = np.sum((y - X_red @ b_red) ** 2)
            return float(max(0.0, ss_red_resid - ss_resid))

        ss_main = _partial_ss(idx_main)
        ss_quad = _partial_ss(idx_quad)
        ss_inter = _partial_ss(idx_inter)
        ss_block = _partial_ss(idx_block)

        # Pure error uses all exact repeated configurations within each block.
        # The resulting LoF tests departures from the additive fixed-block RSM,
        # including configuration-by-block departures. It assumes equal error variance.
        cell_keys = X_coded if block_ids is None else np.column_stack([X_coded, block_ids])
        _, cell_ids = np.unique(cell_keys, axis=0, return_inverse=True)
        df_pe = len(y) - len(np.unique(cell_ids))
        ss_pe = None
        if df_pe > 0:
            ss_pe = float(sum(np.sum((y[cell_ids == c] - y[cell_ids == c].mean())**2)
                              for c in np.unique(cell_ids)))

        ss_lof: Optional[float] = None
        df_lof = 0
        f_lof: Optional[float] = None
        p_lof: Optional[float] = None
        if ss_pe is not None and df_pe > 0:
            ss_lof = max(0.0, ss_resid - ss_pe)
            df_lof = df_resid - df_pe
            if df_lof <= 0:
                raise ValueError('Lack-of-fit has no estimable degrees of freedom')
            ms_lof = ss_lof / df_lof
            ms_pe = ss_pe / df_pe
            f_lof = ms_lof / max(ms_pe, 1e-12)
            p_lof = 1.0 - stats.f.cdf(f_lof, df_lof, df_pe)

        # Assemble ANOVA sources
        sources = ["Model", "  Main Effects", "  2-Factor Interactions", "  Curvature (Quadratic)"]
        dfs = [df_model, len(idx_main), len(idx_inter), len(idx_quad)]
        sss = [ss_model, ss_main, ss_inter, ss_quad]

        if idx_block:
            sources.append("  Blocks")
            dfs.append(len(idx_block))
            sss.append(ss_block)

        sources.append("Residual")
        dfs.append(df_resid)
        sss.append(ss_resid)

        if ss_lof is not None and ss_pe is not None:
            sources.extend(["  Lack of Fit", "  Pure Error"])
            dfs.extend([df_lof, df_pe])
            sss.extend([ss_lof, ss_pe])

        sources.append("Total")
        dfs.append(df_total)
        sss.append(ss_total)

        mss = [s / max(d, 1) for s, d in zip(sss, dfs)]
        f_stats: List[Optional[float]] = []
        p_vals: List[Optional[float]] = []

        for s_name, ms, d in zip(sources, mss, dfs):
            if s_name in ["Model", "  Main Effects", "  2-Factor Interactions", "  Curvature (Quadratic)", "  Blocks"]:
                f_val = ms / max(self.mse_resid_, 1e-12)
                p_val = 1.0 - stats.f.cdf(f_val, d, df_resid)
                f_stats.append(f_val)
                p_vals.append(p_val)
            elif s_name == "  Lack of Fit":
                f_stats.append(f_lof)
                p_vals.append(p_lof)
            else:
                f_stats.append(None)
                p_vals.append(None)

        r2 = max(0.0, 1.0 - (ss_resid / max(ss_total, 1e-12)))
        adj_r2 = max(0.0, 1.0 - (ss_resid / max(df_resid, 1)) / (ss_total / max(df_total, 1)))

        self.anova_ = AnovaTableResult(
            source=sources,
            df=dfs,
            sum_sq=sss,
            mean_sq=mss,
            f_stat=f_stats,
            p_value=p_vals,
            r2=r2,
            adj_r2=adj_r2,
            ss_lof=ss_lof,
            ss_pe=ss_pe,
            f_lof=f_lof,
            p_lof=p_lof,
        )
        self._fitted = True
        return self

    def _prediction_matrix(self, X_coded, block_ids=None):
        if not getattr(self, '_fitted', False):
            raise ValueError('Surface is not fitted')
        X_mat = self._build_design_matrix(X_coded, block_ids=None)
        if len(self.block_levels_) > 1:
            if block_ids is None:
                # Equal-weight adjusted mean across the observed fixed blocks.
                indicators = np.full((len(X_coded), len(self.block_levels_)-1),
                                     1 / len(self.block_levels_))
            else:
                block_ids = np.asarray(block_ids)
                if block_ids.shape != (len(X_coded),) or not np.isin(block_ids, self.block_levels_).all():
                    raise ValueError('Cannot predict an unseen block with a fixed-block OLS model')
                indicators = np.column_stack([block_ids == b for b in self.block_levels_[1:]])
            X_mat = np.column_stack([X_mat, indicators])
        return X_mat

    def predict(self, X_coded: np.ndarray, block_ids=None) -> np.ndarray:
        """Predict a known fixed block, or its equal-weight adjusted block mean."""
        return self._prediction_matrix(X_coded, block_ids) @ self.beta_

    def predict_interval(
        self,
        X_coded: np.ndarray,
        alpha: float = 0.05,
        block_ids=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """OLS intervals for independent observations in known fixed blocks.

        With no block_ids, return a confidence interval for the adjusted mean,
        not a prediction interval for a future stochastic block.
        """
        if not 0 < alpha < 1:
            raise ValueError('alpha must be between zero and one')
        X_mat = self._prediction_matrix(X_coded, block_ids)
        y_hat = X_mat @ self.beta_
        # Standard error of prediction: se = sqrt(MSE * (1 + x^T (X^T X)^-1 x))
        leverage = np.sum((X_mat @ self.xtx_inv_) * X_mat, axis=1)
        observation_noise = 0 if len(self.block_levels_) > 1 and block_ids is None else 1
        se_pred = np.sqrt(self.mse_resid_ * (observation_noise + leverage))

        t_crit = stats.t.ppf(1.0 - alpha / 2.0, max(self.df_resid_, 1))
        pi_low = y_hat - t_crit * se_pred
        pi_high = y_hat + t_crit * se_pred
        return y_hat, pi_low, pi_high


class GaussianProcessSurrogate:
    """Matern Gaussian Process surrogate providing flexible predictions and uncertainty."""

    def __init__(self, random_state: int = 42) -> None:
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
            length_scale=[1.0, 1.0, 1.0, 1.0],
            length_scale_bounds=(1e-1, 1e2),
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_state,
        )

    def fit(self, X_coded: np.ndarray, y: np.ndarray) -> "GaussianProcessSurrogate":
        self.gp.fit(X_coded, y)
        return self

    def predict(self, X_coded: np.ndarray, return_std: bool = False) -> Any:
        return self.gp.predict(X_coded, return_std=return_std)
