"""
Task 6 — Hierarchical forecast reconciliation.

M5 hierarchy (bottom → top):
  item × store  →  dept × store  →  cat × store  →  store  →  state  →  total

We implement MinT (Minimum Trace) reconciliation, which is the gold standard
for making forecasts coherent across levels.

Two paths:
1. Native path via hierarchicalforecast (statsforecast ecosystem) — preferred.
2. Manual OLS path — works without the optional package.

Install advanced extras for path 1:
    pip install -e ".[advanced]"
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# M5 hierarchy definition
HIERARCHY = {
    "total": [],
    "state": ["state_id"],
    "store": ["store_id"],
    "cat_store": ["cat_id", "store_id"],
    "dept_store": ["dept_id", "store_id"],
    "item_store": ["item_id", "store_id"],   # bottom level
}

LEVEL_ORDER = ["total", "state", "store", "cat_store", "dept_store", "item_store"]


# ── Summing matrix ────────────────────────────────────────────────────────────

def build_summing_matrix(
    bottom_df: pd.DataFrame,
    id_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Build the summing matrix S such that y_all = S @ y_bottom.

    Parameters
    ----------
    bottom_df : DataFrame with id columns for the bottom-level series
    id_cols   : list of identifier columns (e.g. ['item_id', 'store_id'])

    Returns
    -------
    S         : (n_all × n_bottom) summing matrix
    all_ids   : list of series identifiers for every row of S
    """
    bottom_ids = bottom_df[id_cols].astype(str).agg("_".join, axis=1).tolist()
    n_bottom = len(bottom_ids)

    agg_dfs = []
    for level_name, level_cols in HIERARCHY.items():
        if level_name == "item_store":
            # bottom level — identity block
            block = np.eye(n_bottom, dtype=float)
            ids = bottom_ids
        else:
            if not level_cols:
                # total — all ones
                block = np.ones((1, n_bottom), dtype=float)
                ids = ["total"]
            else:
                groups = bottom_df.groupby(level_cols).ngroup()
                n_groups = groups.nunique()
                block = np.zeros((n_groups, n_bottom), dtype=float)
                for g_idx, b_idx in zip(groups, range(n_bottom)):
                    block[g_idx, b_idx] = 1.0
                ids = (
                    bottom_df.groupby(level_cols)
                    .first()
                    .reset_index()[level_cols]
                    .astype(str)
                    .agg("_".join, axis=1)
                    .tolist()
                )
        agg_dfs.append((block, ids))

    S = np.vstack([b for b, _ in agg_dfs])
    all_ids = [i for _, ids in agg_dfs for i in ids]
    return S, all_ids


# ── OLS (WLS) reconciliation ──────────────────────────────────────────────────

def reconcile_ols(
    base_forecasts: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """
    OLS reconciliation:  ỹ = S (S'S)^{-1} S' ŷ

    Parameters
    ----------
    base_forecasts : (n_all, horizon) array of unreconciled forecasts
    S              : (n_all, n_bottom) summing matrix

    Returns
    -------
    reconciled forecasts of shape (n_all, horizon)
    """
    P = S @ np.linalg.pinv(S.T @ S) @ S.T   # projection matrix
    return P @ base_forecasts


def reconcile_mint_shrink(
    base_forecasts: np.ndarray,
    S: np.ndarray,
    residuals: np.ndarray,
) -> np.ndarray:
    """
    MinT with shrinkage covariance estimator.

    Parameters
    ----------
    base_forecasts : (n_all, horizon)
    S              : (n_all, n_bottom) summing matrix
    residuals      : (n_all, n_time) in-sample residuals for covariance estimation

    Returns
    -------
    reconciled forecasts of shape (n_all, horizon)
    """
    n, T = residuals.shape

    # Shrinkage covariance (Ledoit-Wolf style diagonal target)
    sample_cov = residuals @ residuals.T / T
    diag_cov = np.diag(np.diag(sample_cov))
    # Shrinkage intensity (simple oracle: 0.5)
    lamda = 0.5
    W = (1 - lamda) * sample_cov + lamda * diag_cov

    W_inv = np.linalg.pinv(W)
    G = np.linalg.pinv(S.T @ W_inv @ S) @ S.T @ W_inv
    reconciled = S @ G @ base_forecasts
    return reconciled


# ── HierarchicalForecast wrapper (optional advanced) ─────────────────────────

def reconcile_with_hierarchicalforecast(
    forecasts_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    S_df: pd.DataFrame,
    tags: dict,
    method: str = "MinTrace",
) -> pd.DataFrame:
    """
    Use the hierarchicalforecast library for reconciliation.

    Parameters
    ----------
    forecasts_df : DataFrame with columns [unique_id, ds, model_name]
    Y_df         : Historical actuals in long format [unique_id, ds, y]
    S_df         : Summing matrix as DataFrame (index=all series, columns=bottom series)
    tags         : dict mapping level name to list of series ids at that level
    method       : 'MinTrace', 'BottomUp', 'TopDown'

    Returns
    -------
    Reconciled forecast DataFrame in the same format as forecasts_df.
    """
    try:
        from hierarchicalforecast.core import HierarchicalReconciliation  # type: ignore[import]
        from hierarchicalforecast.methods import MinTrace, BottomUp, TopDown  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "hierarchicalforecast is required. Install with: pip install -e '.[advanced]'"
        )

    method_map = {
        "MinTrace": MinTrace(method="mint_shrink"),
        "BottomUp": BottomUp(),
        "TopDown": TopDown(method="average_proportions"),
    }
    hrec = HierarchicalReconciliation(reconcilers=[method_map[method]])
    return hrec.reconcile(
        Y_hat_df=forecasts_df,
        Y_df=Y_df,
        S=S_df,
        tags=tags,
    )
