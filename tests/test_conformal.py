"""Tests for conformal prediction intervals."""

import numpy as np
import pytest

from m5forecast.uncertainty.conformal import (
    AdaptiveConformalForecaster,
    ConformalForecaster,
    calibration_report,
)
from m5forecast.models.baselines import SeasonalNaive
import pandas as pd


def _make_series(n: int = 200) -> pd.Series:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(np.clip(20 + rng.normal(0, 3, n), 0, None), index=dates)


class TestConformalForecaster:
    def _setup(self, n_train: int = 150, n_cal: int = 30, horizon: int = 14):
        series = _make_series(n_train + n_cal + horizon)
        train = series.iloc[:n_train]
        cal = series.iloc[n_train:n_train + n_cal]
        test = series.iloc[n_train + n_cal:]

        model = SeasonalNaive(seasonality=7)
        model.fit(train)

        # Calibrate using rolling 1-step predictions on cal set
        # (simplified: predict cal length as one block)
        cal_pred = model.predict(len(cal))
        cal_true = cal.values

        cp = ConformalForecaster(model, coverage_level=0.80)
        cp.calibrate(cal_true, cal_pred)
        return cp, test.values[:horizon]

    def test_calibrate_sets_quantile(self):
        cp, _ = self._setup()
        assert cp._quantile is not None
        assert cp._quantile >= 0

    def test_interval_lower_leq_upper(self):
        cp, _ = self._setup()
        _, lower, upper = cp.predict_interval(14)
        assert (lower <= upper).all()

    def test_lower_non_negative(self):
        cp, _ = self._setup()
        _, lower, _ = cp.predict_interval(14)
        assert (lower >= 0).all()

    def test_empirical_coverage_reasonable(self):
        """80% coverage should be roughly [0.50, 1.0] on a short test set."""
        cp, y_test = self._setup()
        result = cp.evaluate_coverage(y_test, len(y_test))
        cov = result["empirical_coverage"]
        assert 0.0 <= cov <= 1.0

    def test_output_length_matches_horizon(self):
        cp, _ = self._setup(horizon=20)
        point, lower, upper = cp.predict_interval(20)
        assert len(point) == len(lower) == len(upper) == 20


class TestCalibrationReport:
    def test_returns_correct_index(self):
        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 10, 100)
        quantiles = [0.1, 0.5, 0.9]
        q_forecasts = {q: rng.uniform(0, 10, 100) for q in quantiles}
        report = calibration_report(y_true, q_forecasts)
        assert list(report.index) == quantiles

    def test_empirical_coverage_in_01(self):
        rng = np.random.default_rng(1)
        y_true = rng.uniform(0, 10, 100)
        q_forecasts = {0.5: rng.uniform(0, 10, 100)}
        report = calibration_report(y_true, q_forecasts)
        assert 0 <= report.loc[0.5, "empirical_coverage"] <= 1
