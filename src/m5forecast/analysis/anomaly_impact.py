"""
Task 7 — Anomaly detection and promotion / holiday impact analysis.

Two capabilities:
1. Anomaly detection — residual-based Z-score and IQR flagging.
2. Causal impact — interrupted time-series regression (pre/post event)
   with a control series as counterfactual, inspired by CausalImpact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


# ── 1. Anomaly Detection ──────────────────────────────────────────────────────

def detect_anomalies_zscore(
    series: pd.Series,
    window: int = 28,
    threshold: float = 3.0,
) -> pd.Series:
    """
    Flag anomalies where the rolling Z-score exceeds `threshold`.

    Returns a boolean Series (True = anomaly).
    """
    roll_mean = series.rolling(window, min_periods=7, center=True).mean()
    roll_std = series.rolling(window, min_periods=7, center=True).std()
    z = (series - roll_mean) / (roll_std + 1e-6)
    return z.abs() > threshold


def detect_anomalies_iqr(
    series: pd.Series,
    window: int = 28,
    multiplier: float = 2.5,
) -> pd.Series:
    """
    Flag anomalies where the value lies outside
    [Q1 − multiplier × IQR, Q3 + multiplier × IQR] in a rolling window.
    """
    def _iqr_flag(s: np.ndarray) -> bool:
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        v = s[-1]  # current value is last in rolling window
        return bool(v < q1 - multiplier * iqr or v > q3 + multiplier * iqr)

    return series.rolling(window, min_periods=7).apply(_iqr_flag, raw=True).astype(bool)


def anomaly_report(
    df: pd.DataFrame,
    series_col: str = "sales",
    date_col: str = "date",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run Z-score anomaly detection on each series (grouped by group_cols)
    and return a filtered DataFrame of flagged rows.
    """
    if group_cols is None:
        flags = detect_anomalies_zscore(df[series_col])
        return df[flags].copy()

    results = []
    for key, grp in df.groupby(group_cols):
        flags = detect_anomalies_zscore(grp[series_col].reset_index(drop=True))
        flagged = grp.iloc[flags[flags].index].copy()
        flagged["anomaly_score"] = (
            ((grp[series_col] - grp[series_col].rolling(28).mean())
             / (grp[series_col].rolling(28).std() + 1e-6)).values[flags[flags].index]
        )
        results.append(flagged)

    return pd.concat(results).sort_values(date_col) if results else pd.DataFrame()


# ── 2. Promotion / Holiday Impact Analysis ────────────────────────────────────

@dataclass
class ImpactResult:
    """Summary of an interrupted time-series regression."""
    event_name: str
    event_date: date
    pre_mean: float
    post_mean: float
    lift_abs: float
    lift_pct: float
    p_value: float
    confidence_interval: tuple[float, float]
    significant: bool


def estimate_event_impact(
    series: pd.Series,
    event_date: pd.Timestamp | str,
    pre_window: int = 28,
    post_window: int = 14,
    event_name: str = "event",
    control_series: pd.Series | None = None,
) -> ImpactResult:
    """
    Estimate the causal impact of an event (promotion, holiday) using
    interrupted time-series regression.

    Strategy:
    - Regress sales on [time_trend, post_dummy, time × post_dummy, ...]
    - The post_dummy coefficient captures the level shift.
    - Optionally include a control series as a covariate.

    Parameters
    ----------
    series        : target time series (DatetimeIndex)
    event_date    : date of intervention
    pre_window    : days before event to include
    post_window   : days after event to include
    event_name    : label for the ImpactResult
    control_series: concurrent unaffected series (control store / category)
    """
    event_date = pd.Timestamp(event_date)
    start = event_date - pd.Timedelta(days=pre_window)
    end = event_date + pd.Timedelta(days=post_window)

    mask = (series.index >= start) & (series.index <= end)
    window_series = series.loc[mask]

    if len(window_series) < 10:
        raise ValueError(
            f"Not enough data around {event_date} "
            f"(need ≥10 points, got {len(window_series)})."
        )

    t = np.arange(len(window_series))
    post = (window_series.index >= event_date).astype(float).values
    t_post = t * post

    X = np.column_stack([np.ones_like(t), t, post, t_post])
    if control_series is not None:
        ctrl = control_series.reindex(window_series.index, method="nearest").fillna(0).values
        X = np.column_stack([X, ctrl])

    y = window_series.values
    X_sm = sm.add_constant(X[:, 1:])   # statsmodels adds intercept separately
    model = sm.OLS(y, X).fit()

    # post_dummy is the 3rd coefficient (index 2)
    lift_abs = float(model.params[2])
    lift_pct = lift_abs / (window_series.loc[window_series.index < event_date].mean() + 1e-6) * 100
    p_value = float(model.pvalues[2])
    ci = (float(model.conf_int()[0][2]), float(model.conf_int()[1][2]))
    pre_mean = float(window_series.loc[window_series.index < event_date].mean())
    post_mean = float(window_series.loc[window_series.index >= event_date].mean())

    return ImpactResult(
        event_name=event_name,
        event_date=event_date.date(),
        pre_mean=pre_mean,
        post_mean=post_mean,
        lift_abs=lift_abs,
        lift_pct=lift_pct,
        p_value=p_value,
        confidence_interval=ci,
        significant=p_value < 0.05,
    )


def batch_event_impact(
    series: pd.Series,
    events: list[dict],
    pre_window: int = 28,
    post_window: int = 14,
) -> pd.DataFrame:
    """
    Run estimate_event_impact for a list of events and return a summary DataFrame.

    events : list of dicts with keys 'name' and 'date'
    """
    rows = []
    for ev in events:
        try:
            result = estimate_event_impact(
                series,
                event_date=ev["date"],
                event_name=ev["name"],
                pre_window=pre_window,
                post_window=post_window,
            )
            rows.append({
                "event": result.event_name,
                "date": result.event_date,
                "pre_mean": round(result.pre_mean, 2),
                "post_mean": round(result.post_mean, 2),
                "lift_abs": round(result.lift_abs, 2),
                "lift_pct": round(result.lift_pct, 1),
                "p_value": round(result.p_value, 4),
                "significant": result.significant,
            })
        except ValueError as e:
            logger.warning("Skipping %s: %s", ev["name"], e)

    return pd.DataFrame(rows)
