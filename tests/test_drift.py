"""Tests for drift monitoring."""

import numpy as np
import pandas as pd

from m5forecast.monitoring.drift import (
    check_performance_drift,
    feature_drift_report,
    psi,
)


class TestPSI:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 1000)
        score = psi(data, data)
        assert score < 0.05

    def test_different_distributions_higher_psi(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(5, 1, 1000)   # very different mean
        score = psi(ref, cur)
        assert score > 0.10

    def test_non_negative(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(0, 10, 100)
        b = rng.uniform(0, 10, 100)
        assert psi(a, b) >= 0


class TestCheckPerformanceDrift:
    def test_no_drift_identical(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(1, 10, 100)
        p = rng.uniform(1, 10, 100)
        report = check_performance_drift(y, p, y, p, threshold=0.10)
        assert not report.drift_detected

    def test_drift_detected_on_bad_predictions(self):
        rng = np.random.default_rng(0)
        y_ref = rng.uniform(5, 15, 100)
        p_ref = y_ref + rng.normal(0, 0.1, 100)   # good predictions
        y_cur = rng.uniform(5, 15, 100)
        p_cur = np.zeros(100)   # terrible predictions
        report = check_performance_drift(y_ref, p_ref, y_cur, p_cur, threshold=0.10)
        assert report.drift_detected

    def test_report_has_correct_fields(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(1, 10, 50)
        p = rng.uniform(1, 10, 50)
        report = check_performance_drift(y, p, y, p)
        assert hasattr(report, "reference_wape")
        assert hasattr(report, "current_wape")
        assert hasattr(report, "relative_degradation")
        assert hasattr(report, "drift_detected")


class TestFeatureDriftReport:
    def test_returns_dataframe(self):
        rng = np.random.default_rng(0)
        ref = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(5, 2, 100)})
        cur = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(5, 2, 100)})
        report = feature_drift_report(ref, cur, ["a", "b"])
        assert "psi" in report.columns
        assert "drifted" in report.columns

    def test_flags_drifted_feature(self):
        rng = np.random.default_rng(0)
        ref = pd.DataFrame({"x": rng.normal(0, 1, 500)})
        cur = pd.DataFrame({"x": rng.normal(10, 1, 500)})  # huge shift
        report = feature_drift_report(ref, cur, ["x"])
        assert report.loc[report["feature"] == "x", "drifted"].values[0]
