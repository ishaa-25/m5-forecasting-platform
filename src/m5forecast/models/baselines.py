"""
Task 2 — Classical baseline models.

Implements three families:
1. SeasonalNaive   — repeat the last full seasonal cycle
2. ETSForecaster   — Exponential Smoothing (Holt-Winters) via statsmodels
3. ARIMAForecaster — ARIMA via statsmodels (auto-orders supported)

All inherit from Forecaster and plug into the Backtester without changes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from m5forecast.models.base import Forecaster

logger = logging.getLogger(__name__)


# ── 1. Seasonal Naive ──────────────────────────────────────────────────────────

class SeasonalNaive(Forecaster):
    """
    Forecast = value from `seasonality` periods ago, repeated.

    This is the naïve seasonal benchmark used in RMSSE's denominator.
    """

    name = "seasonal_naive"

    def __init__(self, seasonality: int = 7):
        self.seasonality = seasonality
        self._last_season: np.ndarray | None = None

    def fit(self, series: pd.Series) -> SeasonalNaive:
        values = np.asarray(series, dtype=float)
        self._last_season = values[-self.seasonality:]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._last_season is None:
            raise RuntimeError("Call fit() first.")
        # Tile the last season to cover the full horizon
        reps = -(-horizon // self.seasonality)   # ceiling division
        tiled = np.tile(self._last_season, reps)
        return tiled[:horizon]


# ── 2. ETS / Holt-Winters ─────────────────────────────────────────────────────

class ETSForecaster(Forecaster):
    """
    Exponential Smoothing (Holt-Winters) wrapper around statsmodels ExponentialSmoothing.

    Parameters
    ----------
    trend     : 'add', 'mul', or None
    seasonal  : 'add', 'mul', or None
    seasonal_periods : int (e.g. 7 for weekly, 365 for annual)
    damped_trend : bool
    """

    name = "ets"

    def __init__(
        self,
        trend: str | None = "add",
        seasonal: str | None = "add",
        seasonal_periods: int = 7,
        damped_trend: bool = True,
    ):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self._fitted = None
        self._residual_std: float = 0.0

    def fit(self, series: pd.Series) -> ETSForecaster:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # lazy import

        values = np.asarray(series, dtype=float)
        # Holt-Winters needs > 2 × seasonal periods of data
        if len(values) < 2 * self.seasonal_periods + 1:
            logger.warning(
                "Not enough data for seasonal ETS (need %d, got %d). "
                "Falling back to simple exponential smoothing.",
                2 * self.seasonal_periods + 1,
                len(values),
            )
            model = ExponentialSmoothing(values, trend=None, seasonal=None)
        else:
            model = ExponentialSmoothing(
                values,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                damped_trend=self.damped_trend,
            )
        self._fitted = model.fit(optimized=True)
        residuals = self._fitted.resid
        self._residual_std = float(np.std(residuals))
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("Call fit() first.")
        fc = self._fitted.forecast(horizon)
        return np.clip(np.asarray(fc, dtype=float), 0, None)


# ── 3. ARIMA ──────────────────────────────────────────────────────────────────

class ARIMAForecaster(Forecaster):
    """
    ARIMA wrapper around statsmodels ARIMA.

    Parameters
    ----------
    order         : (p, d, q) tuple — use None to auto-select via AIC
    seasonal_order: (P, D, Q, s) tuple for SARIMA
    auto_order    : if True, try a small grid search over (p,d,q) in {0,1,2}
    """

    name = "arima"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] | None = None,
        auto_order: bool = False,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_order = auto_order
        self._fitted = None
        self._residual_std: float = 0.0

    def _select_order(self, values: np.ndarray) -> tuple[int, int, int]:
        """Tiny grid search over ARIMA(p,1,q) for p,q in 0..2."""
        from statsmodels.tsa.arima.model import ARIMA  # lazy import

        best_aic = np.inf
        best_order = (1, 1, 1)
        for p in range(3):
            for q in range(3):
                try:
                    m = ARIMA(values, order=(p, 1, q)).fit()
                    if m.aic < best_aic:
                        best_aic = m.aic
                        best_order = (p, 1, q)
                except Exception:
                    pass
        return best_order

    def fit(self, series: pd.Series) -> ARIMAForecaster:
        from statsmodels.tsa.arima.model import ARIMA  # lazy import

        values = np.asarray(series, dtype=float)
        order = self._select_order(values) if self.auto_order else self.order
        kwargs = {"order": order}
        if self.seasonal_order:
            kwargs["seasonal_order"] = self.seasonal_order
        self._fitted = ARIMA(values, **kwargs).fit()
        self._residual_std = float(np.std(self._fitted.resid))
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("Call fit() first.")
        fc = self._fitted.forecast(steps=horizon)
        return np.clip(np.asarray(fc, dtype=float), 0, None)


# ── CLI: run all baselines and print a quick comparison ───────────────────────
if __name__ == "__main__":
    import logging
    from pathlib import Path

    import pandas as pd

    from m5forecast.evaluation.backtest import Backtester, RollingOriginSplitter

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    processed = Path(__file__).resolve().parents[3] / "data" / "processed"
    parquet = processed / "m5_long.parquet"
    if not parquet.exists():
        print("No processed data found. Run `make data` first.")
        raise SystemExit(1)

    df = pd.read_parquet(parquet)
    # Aggregate to one store for a quick baseline run
    agg = (
        df[df["store_id"] == "CA_1"]
        .groupby("date")["sales"]
        .sum()
        .sort_index()
    )

    models = {
        "seasonal_naive": SeasonalNaive(seasonality=7),
        "ets": ETSForecaster(),
        "arima": ARIMAForecaster(order=(2, 1, 2)),
    }
    bt = Backtester(splitter=RollingOriginSplitter(n_folds=4, horizon=28))
    comparison = bt.compare_models(models, agg)
    print("\nBaseline comparison (mean across 4 folds):\n")
    print(comparison.to_string())
