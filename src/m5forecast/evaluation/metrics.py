"""
Evaluation metrics used throughout the platform.

WAPE  — Weighted Absolute Percentage Error (scale-independent, robust to zeros)
RMSSE — Root Mean Squared Scaled Error (M5 official metric)
Pinball / Quantile loss — for probabilistic evaluation
Coverage — empirical interval coverage rate
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── point forecast metrics ─────────────────────────────────────────────────────

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error.

    WAPE = sum(|y - ŷ|) / sum(|y|)

    Returns a value in [0, ∞).  Lower is better.
    Robust to series with zeros because it uses the sum of actuals as denominator.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def rmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """
    Root Mean Squared Scaled Error (M5 official metric).

    Scale = mean squared naive-seasonal error on training set.
    RMSSE = sqrt( MSE(forecast) / Scale )

    Parameters
    ----------
    y_true : actuals for the forecast window
    y_pred : point forecasts
    y_train : full training history (used to compute the scale)
    seasonality : seasonal period for the naive benchmark (default 1 = random walk)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    mse_forecast = np.mean((y_true - y_pred) ** 2)

    # Denominator: mean squared one-step naive-seasonal error on training set
    n = len(y_train)
    if n <= seasonality:
        return float("nan")
    naive_errors = y_train[seasonality:] - y_train[:-seasonality]
    scale = np.mean(naive_errors**2)
    if scale == 0:
        return float("nan")
    return float(np.sqrt(mse_forecast / scale))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ── probabilistic metrics ──────────────────────────────────────────────────────

def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> float:
    """
    Pinball (quantile) loss for a single quantile level.

    L_q(y, ŷ) = (y - ŷ) * q       if y >= ŷ
               = (ŷ - y) * (1 - q) otherwise

    Returns mean pinball loss over all observations.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    return float(np.mean(np.maximum(quantile * errors, (quantile - 1) * errors)))


def mean_pinball_loss(
    y_true: np.ndarray,
    quantile_forecasts: dict[float, np.ndarray],
) -> float:
    """Average pinball loss across all quantile levels."""
    losses = [
        pinball_loss(y_true, y_pred, q)
        for q, y_pred in quantile_forecasts.items()
    ]
    return float(np.mean(losses))


def coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Empirical coverage: fraction of actuals that fall within [lower, upper].
    A well-calibrated 80% interval should have coverage ≈ 0.80.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean width of prediction intervals (smaller = sharper)."""
    return float(np.mean(np.asarray(upper, float) - np.asarray(lower, float)))


# ── summary helper ─────────────────────────────────────────────────────────────

def summarise_metrics(results: list[dict]) -> pd.DataFrame:
    """
    Aggregate per-fold metric dicts into a summary DataFrame with
    mean ± std across folds.

    Parameters
    ----------
    results : list of dicts, one per fold, each containing metric names as keys.
    """
    df = pd.DataFrame(results)
    summary = df.agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]
    return summary.round(4)
