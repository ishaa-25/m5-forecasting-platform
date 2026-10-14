"""
Task 3 — LightGBM forecaster with lag, rolling, price, and event features.

Strategy: direct multi-step forecasting (one model per horizon step)
with an optional single-model recursive variant.

Hyperparameter tuning via Optuna is exposed through tune_lgbm().
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from m5forecast.models.base import Forecaster

logger = logging.getLogger(__name__)


# ── Feature columns used by LightGBM ──────────────────────────────────────────

LAG_COLS = [f"lag_{d}" for d in [1, 2, 3, 7, 14, 28]]
ROLL_COLS = [
    f"roll_mean_{w}" for w in [7, 14, 28]
] + [f"roll_std_{w}" for w in [7, 14, 28]]
CALENDAR_COLS = [
    "dayofweek", "dayofmonth", "dayofyear", "weekofyear",
    "is_weekend", "has_event",
    "snap_CA", "snap_TX", "snap_WI",
    "event_name_1", "event_type_1",
]
PRICE_COLS = ["sell_price", "price_change", "price_norm", "weeks_at_price"]
ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

FEATURE_COLS = LAG_COLS + ROLL_COLS + CALENDAR_COLS + PRICE_COLS + ID_COLS


class LGBMForecaster(Forecaster):
    """
    LightGBM-based point forecaster.

    Trains a single global model across all series in the dataset,
    then forecasts by feeding constructed features at inference time.

    Parameters
    ----------
    params : LightGBM parameters dict (overrides config defaults)
    horizon : forecast horizon (used during recursive prediction)
    """

    name = "lgbm"

    def __init__(self, params: dict | None = None, horizon: int = 28):
        import lightgbm as lgb  # lazy import

        self.horizon = horizon
        self.params = params or {
            "objective": "regression_l1",
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 127,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        }
        self._model: lgb.LGBMRegressor | None = None
        self._feature_cols: list[str] = []

    def _available_features(self, df: pd.DataFrame) -> list[str]:
        return [c for c in FEATURE_COLS if c in df.columns]

    def fit(self, df: pd.DataFrame) -> "LGBMForecaster":  # type: ignore[override]
        """
        Train on a feature-enriched DataFrame (output of features.py).

        Drops rows with NaN lag features (the warm-up period).
        """
        import lightgbm as lgb

        self._feature_cols = self._available_features(df)
        df_train = df.dropna(subset=[c for c in self._feature_cols if "lag" in c])

        X = df_train[self._feature_cols]
        y = df_train["sales"]

        self._model = lgb.LGBMRegressor(**self.params)
        self._model.fit(X, y)
        logger.info("LightGBM trained on %d rows × %d features", len(X), len(self._feature_cols))
        return self

    def predict(self, horizon: int) -> np.ndarray:  # type: ignore[override]
        """Not used in the global model — use predict_from_df instead."""
        raise NotImplementedError(
            "LGBMForecaster uses predict_from_df(df) for full-dataset inference."
        )

    def predict_from_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Run inference on a pre-built feature DataFrame.
        Returns point forecasts clipped to ≥ 0.
        """
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        X = df[self._feature_cols]
        preds = self._model.predict(X)
        return np.clip(preds, 0, None).astype(float)

    def feature_importance(self) -> pd.Series:
        """Return feature importance as a named Series (gain)."""
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return pd.Series(
            self._model.feature_importances_,
            index=self._feature_cols,
            name="importance",
        ).sort_values(ascending=False)

    def predict_quantiles_from_df(
        self,
        df: pd.DataFrame,
        quantiles: list[float],
    ) -> dict[float, np.ndarray]:
        """
        Train separate quantile models and return quantile predictions.
        Note: this re-trains a model per quantile — use conformal.py for
        a lighter alternative.
        """
        import lightgbm as lgb

        if self._model is None or not hasattr(self, "_train_df"):
            raise RuntimeError("Call fit() with store of training df first.")

        results = {}
        for q in quantiles:
            q_params = {**self.params, "objective": "quantile", "alpha": q}
            qm = lgb.LGBMRegressor(**q_params)
            X_train = self._train_df[self._feature_cols]
            y_train = self._train_df["sales"]
            qm.fit(X_train, y_train)
            X_pred = df[self._feature_cols]
            results[q] = np.clip(qm.predict(X_pred), 0, None)
        return results


# ── Optuna tuning ──────────────────────────────────────────────────────────────

def tune_lgbm(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    n_trials: int = 50,
    timeout: int = 600,
) -> dict:
    """
    Use Optuna to find the best LightGBM hyperparameters.

    Parameters
    ----------
    df_train : feature DataFrame for training
    df_val   : feature DataFrame for validation (same schema)
    n_trials : number of Optuna trials
    timeout  : maximum seconds to run

    Returns
    -------
    dict of best hyperparameters
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tmp = LGBMForecaster()
    feat_cols = tmp._available_features(df_train)

    X_train = df_train.dropna(subset=[c for c in feat_cols if "lag" in c])[feat_cols]
    y_train = df_train.loc[X_train.index, "sales"]
    X_val = df_val.dropna(subset=[c for c in feat_cols if "lag" in c])[feat_cols]
    y_val = df_val.loc[X_val.index, "sales"]

    def objective(trial: optuna.Trial) -> float:
        import lightgbm as lgb
        from m5forecast.evaluation.metrics import wape

        params = {
            "objective": "regression_l1",
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_val), 0, None)
        return wape(y_val.values, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best = study.best_params
    logger.info("Best WAPE=%.4f with params: %s", study.best_value, best)
    return best


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    processed = Path(__file__).resolve().parents[4] / "data" / "processed"
    feat_path = processed / "m5_features.parquet"
    if not feat_path.exists():
        print("Run `make features` first to build the feature dataset.")
        raise SystemExit(1)

    df = pd.read_parquet(feat_path)
    cutoff = df["date"].max() - pd.Timedelta(days=28)
    train_df = df[df["date"] <= cutoff].copy()
    val_df = df[df["date"] > cutoff].copy()

    model = LGBMForecaster()
    model.fit(train_df)
    preds = model.predict_from_df(val_df)

    from m5forecast.evaluation.metrics import wape
    score = wape(val_df["sales"].values, preds)
    print(f"LightGBM validation WAPE: {score:.4f}")

    print("\nTop-20 feature importances:")
    print(model.feature_importance().head(20).to_string())
