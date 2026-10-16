"""
Task 4 — Foundation model: Amazon Chronos.

Wraps the Chronos pipeline (chronos-forecasting package) to match
the Forecaster interface.  Falls back gracefully if the package is
not installed (core extra doesn't include it).

Install with: pip install -e ".[advanced]"
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from m5forecast.models.base import Forecaster

logger = logging.getLogger(__name__)


class ChronosForecaster(Forecaster):
    """
    Zero-shot forecasting with Amazon Chronos.

    Parameters
    ----------
    model_size : 'tiny', 'mini', 'small', 'base', 'large'
                 Larger = more accurate, slower.  'tiny' is good for experiments.
    device     : 'cpu' or 'cuda'
    num_samples: number of sample paths for quantile estimation
    """

    name = "chronos"

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        num_samples: int = 20,
    ):
        self.model_size = model_size
        self.device = device
        self.num_samples = num_samples
        self._pipeline: Any | None = None
        self._context: np.ndarray | None = None

    def _load_pipeline(self) -> None:
        try:
            import torch
            from chronos import ChronosPipeline  # type: ignore[import]

            model_name = f"amazon/chronos-t5-{self.model_size}"
            self._pipeline = ChronosPipeline.from_pretrained(
                model_name,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
            )
            logger.info("Loaded Chronos pipeline: %s", model_name)
        except ImportError:
            raise ImportError(
                "chronos-forecasting and torch are required. "
                "Install with: pip install -e '.[advanced]'"
            )

    def fit(self, series: pd.Series) -> "ChronosForecaster":
        """Chronos is zero-shot — fit just stores the context window."""
        self._context = np.asarray(series, dtype=float)
        if self._pipeline is None:
            self._load_pipeline()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Return median (0.5 quantile) forecast."""
        import torch

        if self._pipeline is None or self._context is None:
            raise RuntimeError("Call fit() first.")

        context_tensor = torch.tensor(self._context, dtype=torch.float32).unsqueeze(0)
        forecast = self._pipeline.predict(
            inputs=context_tensor,
            prediction_length=horizon,
            num_samples=self.num_samples,
        )
        # forecast shape: (1, num_samples, horizon) → median
        samples = forecast[0].numpy()   # (num_samples, horizon)
        return np.clip(np.median(samples, axis=0), 0, None)

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: list[float],
    ) -> dict[float, np.ndarray]:
        """Return empirical quantiles from Monte-Carlo samples."""
        import torch

        if self._pipeline is None or self._context is None:
            raise RuntimeError("Call fit() first.")

        context_tensor = torch.tensor(self._context, dtype=torch.float32).unsqueeze(0)
        forecast = self._pipeline.predict(
            context=context_tensor,
            prediction_length=horizon,
            num_samples=max(self.num_samples, 100),
        )
        samples = forecast[0].numpy()   # (num_samples, horizon)
        samples = np.clip(samples, 0, None)
        return {q: np.quantile(samples, q, axis=0) for q in quantiles}
