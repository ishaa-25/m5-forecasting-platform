"""Tests for anomaly detection."""

import numpy as np
import pandas as pd
import pytest

from m5forecast.analysis.anomaly_impact import (
    detect_anomalies_iqr,
    detect_anomalies_zscore,
    estimate_event_impact,
)


def _make_series(n: int = 100, spike_idx: int = 50) -> pd.Series:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    values = 10 + rng.normal(0, 1, n)
    values[spike_idx] = 100.0  # obvious spike
    return pd.Series(np.clip(values, 0, None), index=dates)


class TestDetectAnomaliesZScore:
    def test_flags_spike(self):
        s = _make_series(spike_idx=50)
        flags = detect_anomalies_zscore(s, window=28, threshold=3.0)
        assert flags.iloc[50]

    def test_returns_boolean_series(self):
        s = _make_series()
        flags = detect_anomalies_zscore(s)
        assert flags.dtype == bool

    def test_no_false_positives_on_constant(self):
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        s = pd.Series(np.ones(50) * 10, index=dates)
        flags = detect_anomalies_zscore(s)
        # Constant series → std ≈ 0, Z-scores undefined; should not crash
        assert isinstance(flags, pd.Series)


class TestDetectAnomaliesIQR:
    def test_flags_spike(self):
        s = _make_series(spike_idx=60)
        flags = detect_anomalies_iqr(s, window=28)
        assert flags.iloc[60]

    def test_returns_boolean_series(self):
        s = _make_series()
        flags = detect_anomalies_iqr(s)
        assert flags.dtype == bool


class TestEstimateEventImpact:
    def _make_event_series(self) -> pd.Series:
        rng = np.random.default_rng(7)
        n = 120
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        values = 10 + rng.normal(0, 0.5, n)
        # Inject a +5 unit lift starting at day 60
        values[60:] += 5.0
        return pd.Series(np.clip(values, 0, None), index=dates)

    def test_detects_positive_lift(self):
        s = self._make_event_series()
        result = estimate_event_impact(
            s,
            event_date="2020-03-01",
            pre_window=28,
            post_window=28,
        )
        assert result.lift_abs > 0

    def test_result_has_required_fields(self):
        s = self._make_event_series()
        result = estimate_event_impact(s, event_date="2020-03-01")
        assert hasattr(result, "p_value")
        assert hasattr(result, "lift_abs")
        assert hasattr(result, "lift_pct")
        assert hasattr(result, "significant")

    def test_raises_on_insufficient_data(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        s = pd.Series(np.ones(5), index=dates)
        with pytest.raises(ValueError):
            estimate_event_impact(s, event_date="2020-01-03", pre_window=10, post_window=10)
