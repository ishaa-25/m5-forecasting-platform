"""Tests for the rolling-origin backtesting harness."""

import numpy as np
import pandas as pd
import pytest

from m5forecast.evaluation.backtest import Backtester, RollingOriginSplitter
from m5forecast.models.baselines import SeasonalNaive


class TestRollingOriginSplitter:
    def _make_index(self, n: int) -> np.ndarray:
        return np.arange(n)

    def test_returns_correct_number_of_folds(self):
        splitter = RollingOriginSplitter(n_folds=4, horizon=28, min_train=100)
        splits = list(splitter.split(self._make_index(500)))
        assert len(splits) == 4

    def test_no_overlap_between_train_and_test(self):
        splitter = RollingOriginSplitter(n_folds=3, horizon=14, min_train=50)
        for train_idx, test_idx in splitter.split(self._make_index(200)):
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_test_immediately_follows_train(self):
        splitter = RollingOriginSplitter(n_folds=2, horizon=10, min_train=20)
        for train_idx, test_idx in splitter.split(self._make_index(100)):
            assert test_idx[0] == train_idx[-1] + 1

    def test_test_length_equals_horizon(self):
        horizon = 14
        splitter = RollingOriginSplitter(n_folds=2, horizon=horizon, min_train=20)
        for _, test_idx in splitter.split(self._make_index(100)):
            assert len(test_idx) == horizon

    def test_no_future_data_in_train(self):
        """Critical: no training index should be >= the first test index."""
        splitter = RollingOriginSplitter(n_folds=4, horizon=28, min_train=100)
        for train_idx, test_idx in splitter.split(self._make_index(400)):
            assert train_idx.max() < test_idx.min()

    def test_skips_folds_with_insufficient_train_data(self):
        splitter = RollingOriginSplitter(n_folds=4, horizon=28, min_train=300)
        splits = list(splitter.split(self._make_index(200)))
        # 200 points is not enough for 4 folds with min_train=300
        assert len(splits) == 0

    def test_train_grows_across_folds(self):
        splitter = RollingOriginSplitter(n_folds=3, horizon=10, step_size=10, min_train=20)
        sizes = [len(tr) for tr, _ in splitter.split(self._make_index(100))]
        assert sizes == sorted(sizes), "Train set should grow (or stay equal) fold-over-fold"


class TestBacktester:
    def _make_series(self, n: int = 400) -> pd.Series:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        values = 10 + rng.normal(0, 1, n).cumsum()
        values = np.clip(values, 1, None)
        return pd.Series(values, index=dates)

    def test_returns_dataframe_with_fold_column(self):
        bt = Backtester(splitter=RollingOriginSplitter(n_folds=2, horizon=14, min_train=50))
        model = SeasonalNaive(seasonality=7)
        series = self._make_series(200)
        result = bt.evaluate(model, series)
        assert "fold" in result.columns
        assert len(result) == 2

    def test_metrics_are_non_negative(self):
        bt = Backtester(splitter=RollingOriginSplitter(n_folds=2, horizon=14, min_train=50))
        model = SeasonalNaive(seasonality=7)
        result = bt.evaluate(model, self._make_series(200))
        for col in ["wape", "mae", "rmse"]:
            assert (result[col] >= 0).all(), f"{col} should be non-negative"

    def test_compare_models_returns_sorted_by_wape(self):
        bt = Backtester(splitter=RollingOriginSplitter(n_folds=2, horizon=7, min_train=50))
        models = {
            "naive_1": SeasonalNaive(seasonality=1),
            "naive_7": SeasonalNaive(seasonality=7),
        }
        summary = bt.compare_models(models, self._make_series(200))
        wapes = summary["wape"].values
        assert list(wapes) == sorted(wapes), "compare_models should sort by WAPE ascending"
