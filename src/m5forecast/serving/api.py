"""
Task 8 — FastAPI forecast serving endpoint.

Endpoints:
  GET  /health            — liveness check
  GET  /info              — model registry info (from MLflow)
  POST /forecast          — get point + interval forecasts for a series
  POST /forecast/batch    — batch version for multiple series IDs
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Load config ────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "config.yaml"
with open(CONFIG_PATH) as fh:
    CONFIG = yaml.safe_load(fh)

app = FastAPI(
    title="M5 Forecast API",
    description="Probabilistic hierarchical forecasting on M5 Walmart data",
    version="0.1.0",
)


# ── Pydantic models ────────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    store_id: str = Field(..., example="CA_1")
    item_id: str | None = Field(None, example="HOBBIES_1_001")
    horizon: int = Field(28, ge=1, le=365, description="Forecast horizon in days")
    quantiles: list[float] = Field(
        default=[0.1, 0.5, 0.9],
        description="Quantile levels to return",
    )
    model: str = Field("lgbm", description="Model to use: lgbm, ets, chronos, nbeats")


class QuantileForecast(BaseModel):
    quantile: float
    values: list[float]


class ForecastResponse(BaseModel):
    store_id: str
    item_id: str | None
    horizon: int
    point_forecast: list[float]
    quantile_forecasts: list[QuantileForecast]
    model_used: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchForecastRequest(BaseModel):
    requests: list[ForecastRequest]


# ── Model registry (simple in-memory for now; swap for MLflow in prod) ─────────

_MODEL_CACHE: dict[str, Any] = {}


def _load_model(model_name: str) -> Any:
    """Load a model from the MLflow registry or local artefacts."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    try:
        import mlflow

        mlflow.set_tracking_uri(CONFIG["mlflow"]["tracking_uri"])
  
        # Try to load the latest production model
        model_uri = f"models:/{model_name}/Production"
        loaded = mlflow.pyfunc.load_model(model_uri)
        _MODEL_CACHE[model_name] = loaded
        return loaded
    except Exception as e:
        logger.warning("Could not load %s from MLflow: %s. Returning None.", model_name, e)
        return None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.get("/info")
async def info() -> dict:
    """Return available models and config."""
    return {
        "available_models": ["seasonal_naive", "ets", "arima", "lgbm", "chronos", "nbeats"],
        "horizon": CONFIG["forecast"]["horizon"],
        "quantiles": CONFIG["uncertainty"]["quantiles"],
    }


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest) -> ForecastResponse:
    """
    Return point and quantile forecasts for a single series.

    In production this would:
    1. Pull the latest actuals from the data warehouse.
    2. Run feature engineering.
    3. Load the registered model from MLflow.
    4. Apply conformal calibration.
    5. Return the forecast.

    Currently returns a placeholder until models are trained.
    """
    model = _load_model(req.model)
    if model is None:
        # Return a stub response so the API is testable before training
        import numpy as np
        rng = np.random.default_rng(42)
        point = rng.uniform(5, 50, req.horizon).round(2).tolist()
        quantile_forecasts = [
            QuantileForecast(quantile=q, values=(np.array(point) * (0.5 + q)).round(2).tolist())
            for q in req.quantiles
        ]
        return ForecastResponse(
            store_id=req.store_id,
            item_id=req.item_id,
            horizon=req.horizon,
            point_forecast=point,
            quantile_forecasts=quantile_forecasts,
            model_used=f"{req.model} (stub — model not yet trained)",
            metadata={"warning": "Model not loaded; returning synthetic forecast"},
        )

    # Real inference path (populated once models are trained)
    raise HTTPException(status_code=501, detail="Real inference not yet implemented.")


@app.post("/forecast/batch")
async def forecast_batch(req: BatchForecastRequest) -> list[ForecastResponse]:
    """Batch version — runs /forecast for each request in the list."""
    return [await forecast(r) for r in req.requests]


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=CONFIG["api"]["host"], port=CONFIG["api"]["port"])
