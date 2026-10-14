"""
Task 1 — Data ingestion.

Reads raw M5 CSV files from data/raw/, melts to long format,
joins calendar and sell-prices, applies optional store subset,
and writes Parquet files to data/processed/.

Usage:
    python -m m5forecast.data.ingest
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def load_config() -> dict:
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


# ── helpers ────────────────────────────────────────────────────────────────────

def _read_sales(raw_dir: Path) -> pd.DataFrame:
    """Melt sales_train_evaluation.csv from wide to long format."""
    path = raw_dir / "sales_train_evaluation.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make data` (scripts/download_data.sh) first."
        )
    df = pd.read_csv(path)
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in df.columns if c.startswith("d_")]
    df_long = df.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    df_long["sales"] = df_long["sales"].astype("float32")
    return df_long


def _read_calendar(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "calendar.csv"
    cal = pd.read_csv(path, parse_dates=["date"])
    # Fill NA event names/types with "none"
    for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        cal[col] = cal[col].fillna("none")
    return cal


def _read_prices(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "sell_prices.csv"
    return pd.read_csv(path)


# ── main pipeline ──────────────────────────────────────────────────────────────

def build_dataset(
    raw_dir: Path = RAW,
    processed_dir: Path = PROCESSED,
    stores: list[str] | None = None,
) -> Path:
    """
    Joins sales, calendar, and prices into a single long-format Parquet.

    Returns the path to the written file.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Reading raw files…")

    sales = _read_sales(raw_dir)
    cal = _read_calendar(raw_dir)
    prices = _read_prices(raw_dir)

    # Filter stores early to save memory
    if stores:
        logger.info("Filtering to stores: %s", stores)
        sales = sales[sales["store_id"].isin(stores)].copy()
        prices = prices[prices["store_id"].isin(stores)].copy()

    # Join calendar  (sales has 'd' column matching calendar's 'd')
    logger.info("Joining calendar…")
    df = sales.merge(
        cal[["d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
             "event_name_1", "event_type_1", "event_name_2", "event_type_2",
             "snap_CA", "snap_TX", "snap_WI"]],
        on="d",
        how="left",
    )

    # Join prices  (keyed on store_id × item_id × wm_yr_wk)
    logger.info("Joining sell prices…")
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # Tidy up
    df["sell_price"] = df["sell_price"].astype("float32")
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    out_path = processed_dir / "m5_long.parquet"
    logger.info("Writing %s rows to %s", len(df), out_path)
    pq.write_table(
        pa.Table.from_pandas(df),
        out_path,
        compression="snappy",
    )
    logger.info("Done.")
    return out_path


def query(sql: str, processed_dir: Path = PROCESSED) -> pd.DataFrame:
    """Run an ad-hoc DuckDB query against the processed Parquet."""
    parquet_path = processed_dir / "m5_long.parquet"
    con = duckdb.connect()
    # Register the parquet file as a virtual table called 'm5'
    con.execute(f"CREATE VIEW m5 AS SELECT * FROM read_parquet('{parquet_path}')")
    return con.execute(sql).df()


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    cfg = load_config()
    build_dataset(stores=cfg["data"].get("stores"))
