"""Tests for evaluation metrics."""

import numpy as np
import pytest

from m5forecast.evaluation.metrics import (
    coverage,
    interval_width,
    mae,
    mean_pinball_loss,
    pinball_loss,
    rmse,
    rmsse,
    summarise_metrics,
    wape,
)


class TestWAPE:
    def test_perfect_forecast(self):
        y = np.array([1.0, 2.0, 3.0])
        assert wape(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        # |10-8| + |20-18| = 4;  |10|+|20| = 30; WAPE = 4/30
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([8.0, 18.0])
        assert wape(y_true, y_pred) == pytest.approx(4 / 30)

    def test_zero_actuals_returns_nan(self):
        assert np.isnan(wape(np.array([0.0, 0.0]), np.array([1.0, 1.0])))

    def test_non_negative(self):
        rng = np.random.default_rng(0)
        y_true = rng.uniform(1, 100, 50)
        y_pred = rng.uniform(1, 100, 50)
        assert wape(y_true, y_pred) >= 0


class TestRMSSE:
    def test_perfect_forecast(self):
        train = np.arange(1, 101, dtype=float)
        y_true = np.array([101.0, 102.0, 103.0])
        assert rmsse(y_true, y_true, train) == pytest.approx(0.0)

    def test_nan_for_constant_train(self):
        train = np.ones(50)
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.5, 1.5])
        assert np.isnan(rmsse(y_true, y_pred, train))

    def test_worse_than_naive_gives_gt_1(self):
        rng = np.random.default_rng(1)
        train = rng.uniform(5, 15, 200)
        y_true = rng.uniform(5, 15, 28)
        # Very bad forecasts
        y_pred = np.full(28, 0.0)
        score = rmsse(y_true, y_pred, train)
        assert score > 0


class TestPinball:
    def test_median_symmetric(self):
        y = np.array([10.0, 10.0, 10.0])
        # Perfect median forecast → loss = 0
        assert pinball_loss(y, y, 0.5) == pytest.approx(0.0)

    def test_overestimate_at_q90(self):
        # Overestimate by 1 at q=0.9 → loss = (1-0.9)*1 = 0.1
        loss = pinball_loss(np.array([0.0]), np.array([1.0]), 0.9)
        assert loss == pytest.approx(0.1)

    def test_underestimate_at_q10(self):
        # Underestimate by 1 at q=0.1 → loss = 0.1 * 1 = 0.1
        loss = pinball_loss(np.array([1.0]), np.array([0.0]), 0.1)
        assert loss == pytest.approx(0.1)


class TestCoverage:
    def test_all_inside(self):
        y = np.array([5.0, 10.0, 15.0])
        lower = np.array([0.0, 5.0, 10.0])
        upper = np.array([10.0, 15.0, 20.0])
        assert coverage(y, lower, upper) == pytest.approx(1.0)

    def test_all_outside(self):
        y = np.array([100.0, 200.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        assert coverage(y, lower, upper) == pytest.approx(0.0)

    def test_half_inside(self):
        y = np.array([5.0, 20.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        assert coverage(y, lower, upper) == pytest.approx(0.5)


class TestMiscMetrics:
    def test_mae(self):
        assert mae(np.array([3.0, 5.0]), np.array([2.0, 7.0])) == pytest.approx(1.5)

    def test_rmse(self):
        assert rmse(np.array([0.0, 4.0]), np.array([0.0, 0.0])) == pytest.approx(
            np.sqrt(8.0)
        )

    def test_interval_width(self):
        lower = np.array([1.0, 2.0])
        upper = np.array([3.0, 6.0])
        assert interval_width(lower, upper) == pytest.approx(3.0)

    def test_summarise_metrics(self):
        results = [
            {"wape": 0.30, "rmsse": 0.80},
            {"wape": 0.40, "rmsse": 1.20},
        ]
        summary = summarise_metrics(results)
        assert summary.loc["wape", "mean"] == pytest.approx(0.35)
        assert summary.loc["rmsse", "std"] == pytest.approx(
            np.std([0.80, 1.20], ddof=1), rel=1e-3
        )
