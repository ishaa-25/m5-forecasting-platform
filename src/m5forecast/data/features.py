"""
Task 3 — Feature engineering.

Adds lag, rolling-window, calendar, and price features to the
long-format DataFrame produced by ingest.py.

All transforms are leak-free: features at time t only use data
available strictly before t (shifted appropriately).

Usage:
    python -m m5forecast.data.features
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"


# ── lag / rolling helpers ──────────────────────────────────────────────────────

LAG_DAYS = [1, 2, 3, 7, 14, 28]
ROLL_WINDOWS = [7, 14, 28]


def add_lag_features(df: pd.DataFrame, target: str = "sales") -> pd.DataFrame:
    """Add lag features per series (item × store)."""
    g = df.groupby(["item_id", "store_id"])[target]
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = g.shift(lag).astype("float32")
    return df


def add_rolling_features(df: pd.DataFrame, target: str = "sales") -> pd.DataFrame:
    """Add rolling mean and std per series (item × store)."""
    g = df.groupby(["item_id", "store_id"])[target]
    for w in ROLL_WINDOWS:
        # shift(1) ensures no same-day leakage
        df[f"roll_mean_{w}"] = (
            g.shift(1).rolling(w).mean().reset_index(level=[0, 1], drop=True).astype("float32")
        )
        df[f"roll_std_{w}"] = (
            g.shift(1).rolling(w).std().reset_index(level=[0, 1], drop=True).astype("float32")
        )
    return df


# ── calendar features ──────────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode calendar columns; expects df to have a 'date' column (datetime)."""
    df["dayofweek"] = df["date"].dt.dayofweek.astype("int8")
    df["dayofmonth"] = df["date"].dt.day.astype("int8")
    df["dayofyear"] = df["date"].dt.dayofyear.astype("int16")
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype("int8")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")

    # Binary event flags (any event that day)
    df["has_event"] = (df["event_name_1"] != "none").astype("int8")

    # SNAP eligibility per state (already columns in the dataset)
    for state in ["CA", "TX", "WI"]:
        snap_col = f"snap_{state}"
        if snap_col in df.columns:
            df[snap_col] = df[snap_col].astype("int8")

    return df


# ── price features ─────────────────────────────────────────────────────────────

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Relative price, price change, and weeks-on-shelf."""
    g = df.groupby(["item_id", "store_id"])["sell_price"]
    df["price_lag1"] = g.shift(1).astype("float32")
    df["price_change"] = (df["sell_price"] - df["price_lag1"]).astype("float32")
    df["price_norm"] = (
        df["sell_price"] / g.transform("mean")
    ).astype("float32")

    # Weeks at current price (proxy for promotion freshness)
    df["price_same_as_lag1"] = (
        (df["sell_price"] == df["price_lag1"]).astype("int8")
    )
    df["weeks_at_price"] = (
        df.groupby(["item_id", "store_id"])["price_same_as_lag1"]
        .transform(lambda s: s.groupby((s != s.shift()).cumsum()).cumcount())
        .astype("int16")
    )
    return df


# ── label-encode categoricals ──────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id",
                "event_name_1", "event_type_1", "event_name_2", "event_type_2",
                "weekday"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


# ── pipeline ───────────────────────────────────────────────────────────────────

def build_features(
    processed_dir: Path = PROCESSED,
    out_name: str = "m5_features.parquet",
) -> Path:
    src = processed_dir / "m5_long.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Run `make data` first.")

    logger.info("Loading processed dataset…")
    df = pd.read_parquet(src)
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    logger.info("Adding calendar features…")
    df = add_calendar_features(df)

    logger.info("Adding lag features…")
    df = add_lag_features(df)

    logger.info("Adding rolling features…")
    df = add_rolling_features(df)

    logger.info("Adding price features…")
    df = add_price_features(df)

    logger.info("Encoding categoricals…")
    df = encode_categoricals(df)

    out_path = processed_dir / out_name
    logger.info("Writing features to %s (%d rows)", out_path, len(df))
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        out_path,
        compression="snappy",
    )
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    build_features()
