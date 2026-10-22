"""
Task 5 — Conformal prediction intervals.

Split-conformal approach (exchangeability assumption):
1. Calibrate non-conformity scores on a held-out calibration set.
2. At inference time, wrap any point forecaster with guaranteed
   (1 - α) marginal coverage.

Also implements quantile LightGBM wrappers for native quantile forecasting,
and a coverage + sharpness report.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from m5forecast.evaluation.metrics import coverage, interval_width

logger = logging.getLogger(__name__)


# ── Split-conformal wrapper ────────────────────────────────────────────────────

class ConformalForecaster:
    """
    Wraps any Forecaster with split-conformal prediction intervals.

    Parameters
    ----------
    forecaster   : a fitted Forecaster (must implement predict())
    coverage_level : desired coverage, e.g. 0.80 for 80% intervals
    """

    def __init__(self, forecaster, coverage_level: float = 0.80):
        self.forecaster = forecaster
        self.coverage_level = coverage_level
        self._quantile: float | None = None
        self._cal_scores: np.ndarray | None = None

    def calibrate(
        self,
        y_cal_true: np.ndarray,
        y_cal_pred: np.ndarray,
    ) -> ConformalForecaster:
        """
        Compute non-conformity scores on a calibration split.

        Score = |y_true - y_pred|  (absolute residual).
        The (1-α) empirical quantile of scores defines the interval half-width.
        """
        scores = np.abs(np.asarray(y_cal_true) - np.asarray(y_cal_pred))
        # Finite-sample correction: use ceil((n+1)(1-α))/n quantile
        n = len(scores)
        alpha = 1 - self.coverage_level
        q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        self._quantile = float(np.quantile(scores, q_level))
        self._cal_scores = scores
        logger.info(
            "Conformal interval half-width: %.4f (coverage=%.0f%%, n_cal=%d)",
            self._quantile,
            self.coverage_level * 100,
            n,
        )
        return self

    def predict_interval(
        self,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (point_forecast, lower_bound, upper_bound).

        Requires calibrate() to have been called.
        """
        if self._quantile is None:
            raise RuntimeError("Call calibrate() first.")
        point = self.forecaster.predict(horizon)
        lower = np.clip(point - self._quantile, 0, None)
        upper = point + self._quantile
        return point, lower, upper

    def evaluate_coverage(
        self,
        y_true: np.ndarray,
        horizon: int,
    ) -> dict:
        """Compute empirical coverage and mean interval width."""
        point, lower, upper = self.predict_interval(horizon)
        return {
            "empirical_coverage": coverage(y_true, lower, upper),
            "target_coverage": self.coverage_level,
            "mean_interval_width": interval_width(lower, upper),
            "conformal_quantile": self._quantile,
        }


# ── Adaptive conformal (time-series aware) ────────────────────────────────────

class AdaptiveConformalForecaster(ConformalForecaster):
    """
    Adaptive conformal prediction that adjusts the quantile online
    to maintain the target coverage under distribution shift.

    See: Gibbs & Candes (2021) — "Adaptive Conformal Inference"
    """

    def __init__(self, forecaster, coverage_level: float = 0.80, lr: float = 0.005):
        super().__init__(forecaster, coverage_level)
        self.lr = lr  # step size for the online adjustment
        self._alpha_t: float = 1 - coverage_level

    def update(self, y_true: float, y_pred: float) -> None:
        """
        Update alpha after observing one new outcome.
        Call this at each new time step during deployment.
        """
        err = abs(y_true - y_pred)
        covered = float(err <= (self._quantile or 0))
        # gradient step: push towards target coverage
        self._alpha_t += self.lr * ((1 - self.coverage_level) - (1 - covered))
        self._alpha_t = float(np.clip(self._alpha_t, 0.01, 0.99))
        if self._cal_scores is not None:
            self._quantile = float(np.quantile(self._cal_scores, 1 - self._alpha_t))


# ── Calibration report ─────────────────────────────────────────────────────────

def calibration_report(
    y_true: np.ndarray,
    quantile_forecasts: dict[float, np.ndarray],
) -> pd.DataFrame:
    """
    For each quantile level, report empirical coverage and pinball loss.
    Well-calibrated forecasts should have empirical coverage ≈ nominal.

    Parameters
    ----------
    y_true            : actual values (1-D)
    quantile_forecasts: dict {q → predictions array}

    Returns
    -------
    DataFrame with columns: quantile, nominal_coverage, empirical_coverage,
                            pinball_loss, calibration_error
    """
    from m5forecast.evaluation.metrics import pinball_loss

    rows = []
    for q, y_pred in sorted(quantile_forecasts.items()):
        empirical = float(np.mean(np.asarray(y_true) <= np.asarray(y_pred)))
        pl = pinball_loss(y_true, y_pred, q)
        rows.append({
            "quantile": q,
            "nominal_coverage": q,
            "empirical_coverage": empirical,
            "pinball_loss": pl,
            "calibration_error": abs(empirical - q),
        })
    return pd.DataFrame(rows).set_index("quantile")
