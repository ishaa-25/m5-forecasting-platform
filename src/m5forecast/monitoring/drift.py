"""
Task 8 — Model monitoring and drift detection.

Two flavours:
1. Performance drift — WAPE on recent actuals vs. a reference window.
   Triggers an alert when degradation exceeds the configured threshold.
2. Data drift — Population Stability Index (PSI) on feature distributions.
   Flags features that shift significantly between reference and current data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "config.yaml"
with open(CONFIG_PATH) as fh:
    _CFG = yaml.safe_load(fh)

DRIFT_THRESHOLD: float = _CFG["monitoring"]["drift_threshold"]
REFERENCE_WINDOW: int = _CFG["monitoring"]["reference_window"]


# ── Performance drift ──────────────────────────────────────────────────────────

@dataclass
class DriftReport:
    reference_wape: float
    current_wape: float
    relative_degradation: float
    drift_detected: bool
    message: str


def check_performance_drift(
    y_true_ref: np.ndarray,
    y_pred_ref: np.ndarray,
    y_true_cur: np.ndarray,
    y_pred_cur: np.ndarray,
    threshold: float = DRIFT_THRESHOLD,
) -> DriftReport:
    """
    Compare WAPE on a reference window vs. the current evaluation window.

    Parameters
    ----------
    y_true_ref / y_pred_ref : actuals and predictions in the reference period
    y_true_cur / y_pred_cur : actuals and predictions in the current period
    threshold               : relative degradation that triggers an alert
                              (e.g. 0.10 = 10% worse)

    Returns
    -------
    DriftReport dataclass
    """
    from m5forecast.evaluation.metrics import wape

    ref_wape = wape(y_true_ref, y_pred_ref)
    cur_wape = wape(y_true_cur, y_pred_cur)
    rel_deg = (cur_wape - ref_wape) / (ref_wape + 1e-9)

    drift_detected = rel_deg > threshold
    msg = (
        f"DRIFT DETECTED: WAPE degraded by {rel_deg:.1%} "
        f"(ref={ref_wape:.4f}, cur={cur_wape:.4f})"
        if drift_detected
        else f"No drift (ref={ref_wape:.4f}, cur={cur_wape:.4f}, Δ={rel_deg:+.1%})"
    )
    logger.info(msg)
    return DriftReport(
        reference_wape=ref_wape,
        current_wape=cur_wape,
        relative_degradation=rel_deg,
        drift_detected=drift_detected,
        message=msg,
    )


# ── Data / feature drift (PSI) ─────────────────────────────────────────────────

def psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index between two distributions.

    PSI < 0.10 : No significant change
    0.10–0.25  : Moderate shift — investigate
    > 0.25     : Major shift — likely drift
    """
    # Build bins from reference distribution
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    ref_counts = np.histogram(reference, bins=bins)[0].astype(float)
    cur_counts = np.histogram(current, bins=bins)[0].astype(float)

    # Add small epsilon to avoid log(0)
    ref_pct = ref_counts / (ref_counts.sum() + 1e-9) + 1e-9
    cur_pct = cur_counts / (cur_counts.sum() + 1e-9) + 1e-9

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_drift_report(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    numeric_cols: list[str],
    psi_threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Compute PSI for each numeric feature and flag drifted ones.

    Returns a DataFrame with columns: feature, psi, drifted.
    """
    rows = []
    for col in numeric_cols:
        if col not in ref_df.columns or col not in cur_df.columns:
            continue
        ref_vals = ref_df[col].dropna().values
        cur_vals = cur_df[col].dropna().values
        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue
        score = psi(ref_vals, cur_vals)
        rows.append({
            "feature": col,
            "psi": round(score, 4),
            "drifted": score > psi_threshold,
            "severity": "major" if score > 0.25 else ("moderate" if score > 0.10 else "none"),
        })

    df = pd.DataFrame(rows).sort_values("psi", ascending=False)
    n_drifted = df["drifted"].sum()
    if n_drifted:
        logger.warning("%d features show drift (PSI > %.2f)", n_drifted, psi_threshold)
    return df


# ── Scheduled retraining trigger ──────────────────────────────────────────────

def should_retrain(
    recent_actuals: np.ndarray,
    recent_predictions: np.ndarray,
    reference_actuals: np.ndarray,
    reference_predictions: np.ndarray,
) -> bool:
    """
    Convenience function: returns True if retraining is recommended.
    Called by a scheduler (e.g. GitHub Actions cron or Airflow DAG).
    """
    report = check_performance_drift(
        reference_actuals, reference_predictions,
        recent_actuals, recent_predictions,
    )
    return report.drift_detected
