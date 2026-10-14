# Project Plan — M5 Forecasting Platform

## Goal

Build a production-grade probabilistic, hierarchical forecasting system
on the M5 Walmart dataset that demonstrates every skill currently thin on
the resume.

---

## Task table (~40 hours)

| # | Task | Deliverable | Est. |
|---|------|-------------|------|
| 1 | Download M5, choose a subset (3 stores, all depts), load into Parquet with DuckDB, run EDA on seasonality, intermittency, and price effects | `data/ingest.py` + `notebooks/01_eda.ipynb` | 4 h |
| 2 | Build rolling-origin backtest framework (WAPE + RMSSE), run baselines: seasonal naïve, ETS/Holt-Winters, ARIMA | `evaluation/backtest.py` + `notebooks/02_baselines.ipynb` | 5 h |
| 3 | LightGBM with lag, rolling, price, event features; tune with Optuna | `models/lgbm.py` + `notebooks/03_lgbm.ipynb` | 6 h |
| 4 | Foundation model (Chronos) + deep model (N-BEATS or PatchTST via NeuralForecast); cost vs accuracy | `models/foundation.py` + `models/deep.py` + `notebooks/04_model_comparison.ipynb` | 6 h |
| 5 | Uncertainty: quantile forecasts + conformal intervals, check coverage | `uncertainty/conformal.py` | 4 h |
| 6 | Hierarchical reconciliation: MinT via HierarchicalForecast | `reconciliation/hierarchy.py` | 3 h |
| 7 | Anomaly detection + promotion/holiday impact (ITS regression) | `analysis/anomaly_impact.py` + `notebooks/05_anomaly_impact.ipynb` | 4 h |
| 8 | MLflow tracking, FastAPI endpoint, scheduled retraining with drift check, Dockerfile, GitHub Actions CI | `serving/api.py` · `monitoring/drift.py` · `Dockerfile` · `.github/workflows/ci.yml` | 5 h |
| 9 | Streamlit dashboard (forecast fans + drill-down) + optional LLM "explain this forecast" | `app/dashboard.py` · `llm/explain.py` | 3 h |
| 10 | README (business framing, results, limitations), resume bullets, LinkedIn post | `README.md` · `docs/RESUME_BULLETS.md` | 3 h |

**Total: ~43 hours**

---

## Definition of done (per task)

- Code is in `src/m5forecast/` and importable.
- At least one test in `tests/` covers the core logic.
- CI passes (ruff lint + pytest).
- Results are logged to MLflow (Tasks 2–7).
- Notebook has a conclusions cell summarising the key numbers.

---

## Key design decisions

### Why DuckDB?
SQL-style EDA on Parquet files without spinning up a database server.
Interviewers recognise it as a modern data-engineering choice.

### Why direct multi-step for LightGBM?
One model per horizon step avoids compounding errors from recursive
forecasting.  Slower to train but more accurate for longer horizons.

### Why split-conformal intervals?
Model-agnostic — wraps any forecaster.  Coverage guarantee holds by
construction (no distributional assumptions), which is the right answer
for a retail setting with event-driven spikes.

### Why MinT reconciliation?
OLS projection (BottomUp) is unbiased but ignores error correlations.
MinT minimises total forecast variance across the hierarchy, giving
better overall accuracy at every level simultaneously.
