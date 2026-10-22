"""
Task 4 — Deep learning models: N-BEATS and PatchTST via NeuralForecast.

Both models are accessed through the NeuralForecast unified API so they
plug straight into the same comparison table as LightGBM and Chronos.

Install with: pip install -e ".[advanced]"
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from m5forecast.models.base import Forecaster

logger = logging.getLogger(__name__)


def _build_nf_df(series: pd.Series, series_id: str = "series_0") -> pd.DataFrame:
    """
    Convert a univariate pd.Series to the NeuralForecast long-format DataFrame:
    columns: unique_id, ds, y
    """
    return pd.DataFrame({
        "unique_id": series_id,
        "ds": series.index,
        "y": series.values,
    })


class NBEATSForecaster(Forecaster):
    """
    N-BEATS (Neural Basis Expansion) via neuralforecast.

    Parameters
    ----------
    horizon          : forecast horizon
    input_size       : lookback window (default = 2 × horizon)
    max_steps        : training steps
    accelerator      : 'cpu' or 'gpu'
    """

    name = "nbeats"

    def __init__(
        self,
        horizon: int = 28,
        input_size: int | None = None,
        max_steps: int = 500,
        accelerator: str = "cpu",
    ):
        self.horizon = horizon
        self.input_size = input_size or 2 * horizon
        self.max_steps = max_steps
        self.accelerator = accelerator
        self._nf: Any | None = None
        self._series_id = "series_0"
        self._last_series: pd.Series | None = None

    def fit(self, series: pd.Series) -> NBEATSForecaster:
        try:
            from neuralforecast import NeuralForecast  # type: ignore[import]
            from neuralforecast.models import NBEATS  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "neuralforecast is required. Install with: pip install -e '.[advanced]'"
            )

        self._last_series = series
        nf_df = _build_nf_df(series, self._series_id)
        model = NBEATS(
            h=self.horizon,
            input_size=self.input_size,
            max_steps=self.max_steps,
            accelerator=self.accelerator,
        )
        self._nf = NeuralForecast(models=[model], freq=pd.infer_freq(series.index) or "D")
        self._nf.fit(df=nf_df)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._nf is None or self._last_series is None:
            raise RuntimeError("Call fit() first.")
        fc_df = self._nf.predict()
        preds = fc_df["NBEATS"].values[:horizon]
        return np.clip(preds, 0, None)


class PatchTSTForecaster(Forecaster):
    """
    PatchTST (Patch Time-Series Transformer) via neuralforecast.

    Parameters
    ----------
    horizon   : forecast horizon
    input_size: lookback window
    max_steps : training steps
    accelerator: 'cpu' or 'gpu'
    """

    name = "patchtst"

    def __init__(
        self,
        horizon: int = 28,
        input_size: int | None = None,
        max_steps: int = 500,
        accelerator: str = "cpu",
    ):
        self.horizon = horizon
        self.input_size = input_size or 4 * horizon
        self.max_steps = max_steps
        self.accelerator = accelerator
        self._nf: Any | None = None
        self._last_series: pd.Series | None = None

    def fit(self, series: pd.Series) -> PatchTSTForecaster:
        try:
            from neuralforecast import NeuralForecast  # type: ignore[import]
            from neuralforecast.models import PatchTST  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "neuralforecast is required. Install with: pip install -e '.[advanced]'"
            )

        self._last_series = series
        nf_df = _build_nf_df(series, "series_0")
        model = PatchTST(
            h=self.horizon,
            input_size=self.input_size,
            max_steps=self.max_steps,
            accelerator=self.accelerator,
        )
        self._nf = NeuralForecast(models=[model], freq=pd.infer_freq(series.index) or "D")
        self._nf.fit(df=nf_df)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._nf is None:
            raise RuntimeError("Call fit() first.")
        fc_df = self._nf.predict()
        preds = fc_df["PatchTST"].values[:horizon]
        return np.clip(preds, 0, None)
