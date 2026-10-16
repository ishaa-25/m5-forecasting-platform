"""
Task 1 — Data ingestion.

Reads raw M5 CSV files from data/raw/, melts to long format,
joins calendar and sell-prices, applies optional store subset,
and writes Parquet files to data/processed/.

Memory-efficient strategy:
- Filter to target stores BEFORE melting (cuts rows by ~90%)
- Use int16 for sales counts and category dtypes for string columns
- Melt only the already-filtered subset

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

def _read_sales(raw_dir: Path, stores: list[str] | None = None) -> pd.DataFrame:
    """
    Read sales_train_evaluation.csv and melt from wide to long format.

    Key memory optimisation: filter to `stores` BEFORE melting so we
    only allocate memory for the rows we actually need.

    Full dataset:  30,490 items × 1,941 days  ≈ 59 M rows  (~4 GB)
    3-store subset: ~3,049 items × 1,941 days  ≈  6 M rows  (~400 MB)
    """
    path = raw_dir / "sales_train_evaluation.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Download the M5 data first:\n"
            "  kaggle competitions download -c m5-forecasting-accuracy -p data/raw\n"
            "  Expand-Archive data/raw/m5-forecasting-accuracy.zip data/raw"
        )

    logger.info("Reading sales CSV header to get column list…")
    # Read only the id columns first to identify which rows belong to our stores
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    # Use low_memory=False + explicit dtype for the id columns
    # Read in two passes: first get just id cols to build a filter mask,
    # then read the full file for only the matched rows.
    logger.info("Pass 1 — reading id columns to build store filter…")
    id_df = pd.read_csv(path, usecols=id_cols, dtype=str)

    if stores:
        mask = id_df["store_id"].isin(stores)
        logger.info(
            "Keeping %d / %d series (stores: %s)",
            mask.sum(), len(id_df), stores,
        )
        row_indices = id_df.index[mask].tolist()
    else:
        row_indices = None   # keep all

    del id_df   # free memory before the big read

    logger.info("Pass 2 — reading full sales file for selected rows…")
    # Read all columns; skiprows skips rows NOT in our index set
    # We need to add 0 back (header row is always kept)
    all_df = pd.read_csv(path, dtype=str)

    if row_indices is not None:
        all_df = all_df.iloc[row_indices].copy()

    # Convert day columns to int16 (sales counts fit easily in int16: 0–32767)
    day_cols = [c for c in all_df.columns if c.startswith("d_")]
    for c in day_cols:
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce").fillna(0).astype("int16")

    logger.info(
        "Melting %d series × %d day columns → long format…",
        len(all_df), len(day_cols),
    )
    df_long = all_df.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    df_long["sales"] = df_long["sales"].astype("float32")

    # Save memory: convert string id cols to category
    for c in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        df_long[c] = df_long[c].astype("category")

    logger.info("Long-format sales: %d rows", len(df_long))
    return df_long


def _read_calendar(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "calendar.csv"
    cal = pd.read_csv(path, parse_dates=["date"])
    for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        cal[col] = cal[col].fillna("none").astype("category")
    cal["weekday"] = cal["weekday"].astype("category")
    return cal


def _read_prices(raw_dir: Path, stores: list[str] | None = None) -> pd.DataFrame:
    path = raw_dir / "sell_prices.csv"
    prices = pd.read_csv(path)
    if stores:
        prices = prices[prices["store_id"].isin(stores)].copy()
    prices["sell_price"] = prices["sell_price"].astype("float32")
    prices["store_id"] = prices["store_id"].astype("category")
    prices["item_id"] = prices["item_id"].astype("category")
    return prices


# ── main pipeline ──────────────────────────────────────────────────────────────

def build_dataset(
    raw_dir: Path = RAW,
    processed_dir: Path = PROCESSED,
    stores: list[str] | None = None,
) -> Path:
    """
    Joins sales, calendar, and prices into a single long-format Parquet.
    Filters to `stores` before melting to keep memory usage low.

    Returns the path to the written Parquet file.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Step 1/4: Reading sales data ===")
    sales = _read_sales(raw_dir, stores=stores)

    logger.info("=== Step 2/4: Reading calendar ===")
    cal = _read_calendar(raw_dir)

    logger.info("=== Step 3/4: Reading sell prices ===")
    prices = _read_prices(raw_dir, stores=stores)

    # Join calendar  (sales has 'd' column matching calendar's 'd')
    logger.info("=== Step 4/4: Joining tables ===")
    cal_cols = [
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    df = sales.merge(cal[cal_cols], on="d", how="left")

    # Join prices  (keyed on store_id × item_id × wm_yr_wk)
    # Need to align category dtypes before merge
    for col in ["store_id", "item_id"]:
        df[col] = df[col].astype(str)
        prices[col] = prices[col].astype(str)

    df = df.merge(prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
                  on=["store_id", "item_id", "wm_yr_wk"],
                  how="left")

    # Re-apply categories after merge
    for c in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    out_path = processed_dir / "m5_long.parquet"
    logger.info("Writing %d rows to %s …", len(df), out_path)
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        out_path,
        compression="snappy",
    )
    logger.info("=== Done! Parquet written to %s ===", out_path)
    return out_path


def query(sql: str, processed_dir: Path = PROCESSED) -> pd.DataFrame:
    """Run an ad-hoc DuckDB SQL query against the processed Parquet file."""
    parquet_path = processed_dir / "m5_long.parquet"
    con = duckdb.connect()
    con.execute(f"CREATE VIEW m5 AS SELECT * FROM read_parquet('{parquet_path}')")
    return con.execute(sql).df()


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    cfg = load_config()
    build_dataset(stores=cfg["data"].get("stores"))
