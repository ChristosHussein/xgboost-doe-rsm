"""xgb_doe_xai.surrogates - Parametric 2nd-order RSM and non-parametric surrogates.

Provides rigorous ANOVA decomposition, Lack-of-Fit tests against center pure error,
Satterthwaite prediction intervals, and flexible surrogate modeling for SHAP explanation.
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

        self.feature_names = names
        return np.hstack(cols)

    def fit(
        self,
        X_coded: np.ndarray,
        y: np.ndarray,
        block_ids: Optional[np.ndarray] = None,
    ) -> "PolynomialRSMSurrogate":
        """Fit OLS 2nd-order response surface and compute complete ANOVA decomposition."""
        X_mat = self._build_design_matrix(X_coded, block_ids)
        n, p = X_mat.shape

        # Solve OLS via SVD / pseudo-inverse for numerical stability
        beta, residuals, rank, s = np.linalg.lstsq(X_mat, y, rcond=1e-10)
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

        # Lack-of-Fit vs Pure Error at center points
        center_mask = np.all(np.isclose(X_coded, 0.0, atol=1e-4), axis=1)
        ss_pe: Optional[float] = None
        df_pe = 0
        if np.sum(center_mask) > 1:
            y_centers = y[center_mask]
            if block_ids is not None:
                # Within-block center pure error
                ss_pe = 0.0
                for b in np.unique(block_ids[center_mask]):
                    b_yc = y_centers[block_ids[center_mask] == b]
                    if len(b_yc) > 1:
                        ss_pe += float(np.sum((b_yc - np.mean(b_yc)) ** 2))
                        df_pe += len(b_yc) - 1
            else:
                ss_pe = float(np.sum((y_centers - np.mean(y_centers)) ** 2))
                df_pe = len(y_centers) - 1

        ss_lof: Optional[float] = None
        df_lof = 0
        f_lof: Optional[float] = None
        p_lof: Optional[float] = None
        if ss_pe is not None and df_pe > 0:
            ss_lof = max(0.0, ss_resid - ss_pe)
            df_lof = max(1, df_resid - df_pe)
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
        return self

    def predict(self, X_coded: np.ndarray) -> np.ndarray:
        """Predict response at coded coordinates without block bias (neutral block=0)."""
        X_mat = self._build_design_matrix(X_coded, block_ids=None)
        # Select beta coefficients excluding block terms
        p_base = X_mat.shape[1]
        return X_mat @ self.beta_[:p_base]

    def predict_interval(
        self,
        X_coded: np.ndarray,
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute point prediction, lower PI bound, and upper PI bound (95% Satterthwaite PI)."""
        X_mat = self._build_design_matrix(X_coded, block_ids=None)
        p_base = X_mat.shape[1]
        y_hat = X_mat @ self.beta_[:p_base]

        sub_xtx_inv = self.xtx_inv_[:p_base, :p_base]
        # Standard error of prediction: se = sqrt(MSE * (1 + x^T (X^T X)^-1 x))
        leverage = np.sum((X_mat @ sub_xtx_inv) * X_mat, axis=1)
        se_pred = np.sqrt(self.mse_resid_ * (1.0 + leverage))

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
