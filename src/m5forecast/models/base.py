"""
Abstract base class for all forecasters.

Every model in this platform must implement:
  fit(series)          — accept a pd.Series or pd.DataFrame
  predict(horizon)     — return a numpy array of point forecasts

Optionally:
  predict_quantiles(horizon, quantiles) — return a dict {q: array}
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Forecaster(ABC):
    """Minimal interface that every model must satisfy."""

    name: str = "base"

    @abstractmethod
    def fit(self, series: pd.Series) -> "Forecaster":
        """
        Train the model on historical data.

        Parameters
        ----------
        series : pd.Series with DatetimeIndex, values are the target.
        """

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """
        Return `horizon` point forecasts as a 1-D numpy array.
        """

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: list[float],
    ) -> dict[float, np.ndarray]:
        """
        Return quantile forecasts.

        Default implementation: symmetric Gaussian intervals around the
        point forecast using the in-sample residual std.  Override in
        subclasses that have native probabilistic output.
        """
        point = self.predict(horizon)
        if hasattr(self, "_residual_std") and self._residual_std > 0:
            std = self._residual_std
        else:
            std = np.abs(point).mean() * 0.20  # rough 20% fallback

        from scipy.stats import norm  # lazy import

        result: dict[float, np.ndarray] = {}
        for q in quantiles:
            z = norm.ppf(q)
            result[q] = np.clip(point + z * std, 0, None)
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
