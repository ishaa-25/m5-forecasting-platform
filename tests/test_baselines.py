"""Tests for classical baseline models."""

import numpy as np
import pandas as pd
import pytest

from m5forecast.models.baselines import ARIMAForecaster, ETSForecaster, SeasonalNaive


def _make_seasonal_series(n: int = 200, period: int = 7) -> pd.Series:
    """Simple additive seasonal series for unit tests."""
    rng = np.random.default_rng(0)
    t = np.arange(n)
    seasonal = 5 * np.sin(2 * np.pi * t / period)
    trend = 0.01 * t
    noise = rng.normal(0, 0.5, n)
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.Series(np.clip(20 + trend + seasonal + noise, 0, None), index=dates)


class TestSeasonalNaive:
    def test_output_length_matches_horizon(self):
        s = _make_seasonal_series()
        model = SeasonalNaive(seasonality=7)
        model.fit(s)
        assert len(model.predict(28)) == 28

    def test_predict_repeats_last_season(self):
        s = _make_seasonal_series()
        model = SeasonalNaive(seasonality=7)
        model.fit(s)
        preds = model.predict(7)
        last_season = s.values[-7:]
        np.testing.assert_array_almost_equal(preds, last_season)

    def test_predict_longer_than_season(self):
        """Horizon > seasonality should tile correctly."""
        s = _make_seasonal_series()
        model = SeasonalNaive(seasonality=7)
        model.fit(s)
        preds = model.predict(10)
        assert len(preds) == 10
        # First 7 values == last season; next 3 == first 3 of last season
        np.testing.assert_array_almost_equal(preds[:7], s.values[-7:])
        np.testing.assert_array_almost_equal(preds[7:10], s.values[-7:-4])

    def test_raises_if_not_fitted(self):
        model = SeasonalNaive()
        with pytest.raises(RuntimeError):
            model.predict(7)


class TestETSForecaster:
    def test_output_length(self):
        s = _make_seasonal_series(n=100)
        model = ETSForecaster(seasonal_periods=7)
        model.fit(s)
        preds = model.predict(14)
        assert len(preds) == 14

    def test_forecasts_non_negative(self):
        s = _make_seasonal_series(n=100)
        model = ETSForecaster(seasonal_periods=7)
        model.fit(s)
        preds = model.predict(28)
        assert (preds >= 0).all()

    def test_quantile_intervals_ordered(self):
        s = _make_seasonal_series(n=100)
        model = ETSForecaster(seasonal_periods=7)
        model.fit(s)
        qs = model.predict_quantiles(14, [0.1, 0.5, 0.9])
        # Q10 ≤ Q50 ≤ Q90 at every step
        assert (qs[0.1] <= qs[0.5]).all()
        assert (qs[0.5] <= qs[0.9]).all()


class TestARIMAForecaster:
    def test_output_length(self):
        s = _make_seasonal_series(n=100)
        model = ARIMAForecaster(order=(1, 1, 1))
        model.fit(s)
        assert len(model.predict(14)) == 14

    def test_forecasts_non_negative(self):
        s = _make_seasonal_series(n=100)
        model = ARIMAForecaster(order=(1, 1, 1))
        model.fit(s)
        assert (model.predict(14) >= 0).all()
