"""
Task 2 — Rolling-origin backtesting harness.

The RollingOriginSplitter generates train/test index splits without ever
letting future data leak into training.  The Backtester drives any model
that implements the Forecaster interface (see models/base.py) through
multiple folds and collects metrics.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from m5forecast.evaluation.metrics import mae, rmse, rmsse, wape

logger = logging.getLogger(__name__)


# ── splitter ───────────────────────────────────────────────────────────────────

@dataclass
class RollingOriginSplitter:
    """
    Generates non-overlapping rolling-origin train/test splits.

    Parameters
    ----------
    n_folds   : number of evaluation windows
    horizon   : forecast horizon (test window size) in periods
    step_size : how many periods to advance the cutoff between folds
                (defaults to horizon — non-overlapping test windows)
    min_train : minimum number of training periods required
    """

    n_folds: int = 4
    horizon: int = 28
    step_size: int | None = None   # defaults to horizon
    min_train: int = 365

    def split(self, index: pd.DatetimeIndex | pd.RangeIndex | np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_indices, test_indices) for each fold.

        The test window for fold k starts at position:
            len(index) - horizon - step * (n_folds - 1 - k)
        so that the last fold uses all available history for training.
        """
        step = self.step_size if self.step_size is not None else self.horizon
        n = len(index)

        for k in range(self.n_folds):
            # cutoff: last training index (exclusive)
            cutoff = n - self.horizon - step * (self.n_folds - 1 - k)
            if cutoff < self.min_train:
                logger.warning(
                    "Fold %d: only %d training points (min_train=%d). Skipping.",
                    k,
                    cutoff,
                    self.min_train,
                )
                continue
            train_idx = np.arange(cutoff)
            test_idx = np.arange(cutoff, cutoff + self.horizon)
            yield train_idx, test_idx

    def n_valid_folds(self, n: int) -> int:
        """Count how many folds are actually valid for a series of length n."""
        step = self.step_size if self.step_size is not None else self.horizon
        count = 0
        for k in range(self.n_folds):
            cutoff = n - self.horizon - step * (self.n_folds - 1 - k)
            if cutoff >= self.min_train:
                count += 1
        return count


# ── backtester ─────────────────────────────────────────────────────────────────

@dataclass
class Backtester:
    """
    Drives a Forecaster through multiple rolling-origin folds and
    collects WAPE, RMSSE, MAE, and RMSE.

    Works on aggregate (store-level) data; for item-level backtesting
    use BacktestRunner which iterates over series.
    """

    splitter: RollingOriginSplitter = field(default_factory=RollingOriginSplitter)

    def evaluate(
        self,
        model,          # any object with .fit(train_df) and .predict(horizon) -> np.ndarray
        series: pd.Series,
        horizon: int | None = None,
    ) -> pd.DataFrame:
        """
        Run the model over all folds on a single univariate series.

        Parameters
        ----------
        model   : Forecaster instance (fit/predict interface)
        series  : pd.Series with a DatetimeIndex, sorted chronologically
        horizon : override splitter horizon for this run

        Returns
        -------
        DataFrame with one row per fold and columns for each metric.
        """
        if horizon is not None:
            self.splitter.horizon = horizon

        fold_results: list[dict] = []
        values = series.values

        for fold_idx, (train_idx, test_idx) in enumerate(self.splitter.split(values)):
            train_values = values[train_idx]
            test_values = values[test_idx]

            # Build a Series the model can consume
            train_series = series.iloc[train_idx]

            model.fit(train_series)
            y_pred = model.predict(len(test_idx))

            y_pred = np.asarray(y_pred, dtype=float)

            fold_results.append({
                "fold": fold_idx,
                "n_train": len(train_idx),
                "wape": wape(test_values, y_pred),
                "rmsse": rmsse(test_values, y_pred, train_values),
                "mae": mae(test_values, y_pred),
                "rmse": rmse(test_values, y_pred),
            })
            logger.debug(
                "Fold %d: WAPE=%.4f RMSSE=%.4f",
                fold_idx,
                fold_results[-1]["wape"],
                fold_results[-1]["rmsse"],
            )

        return pd.DataFrame(fold_results)

    def compare_models(
        self,
        models: dict[str, object],
        series: pd.Series,
    ) -> pd.DataFrame:
        """
        Evaluate multiple models and return a comparison table sorted by mean WAPE.

        Parameters
        ----------
        models : dict mapping model_name → Forecaster instance
        series : single univariate series to evaluate on
        """
        rows = []
        for name, model in models.items():
            fold_df = self.evaluate(model, series)
            for _, row in fold_df.iterrows():
                rows.append({"model": name, **row.to_dict()})
        df = pd.DataFrame(rows)
        # Summary: mean across folds
        summary = (
            df.groupby("model")[["wape", "rmsse", "mae", "rmse"]]
            .mean()
            .sort_values("wape")
        )
        return summary
