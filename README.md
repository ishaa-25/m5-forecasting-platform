# M5 Forecasting Platform
A Streamlit-powered interface providing hierarchical drill-downs, probabilistic forecast fans (with conformal intervals), and LLM-driven explanations.

![M5 Dashboard Overview](Screenshot (134).png)
> Probabilistic, hierarchical forecasting on the M5 Walmart dataset —
> built as a small production system.

[![CI](https://github.com/ishaa-25/m5-forecasting-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/ishaa-25/m5-forecasting-platform/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Business framing

Walmart runs ~42 000 related daily sales series across stores, departments,
and states.  Getting those forecasts wrong cascades into over-stocking (margin
loss) or stock-outs (lost revenue).  This platform shows how to go beyond a
single model by building an end-to-end system that:

- compares classical, gradient-boosting, and foundation models on equal footing
- produces calibrated probability ranges instead of point estimates
- reconciles item-level forecasts so they sum correctly at every hierarchy level
- monitors itself and flags when it needs retraining

Benchmarked on a 3-store subset (`CA_1`, `TX_1`, `WI_1`) with a 28-day
horizon and 4-fold rolling-origin backtest.

---

## Results (3-store subset, 28-day horizon)

| Model | WAPE ↓ | RMSSE ↓ | Notes |
|---|---|---|---|
| Seasonal Naive (baseline) | 0.42 | 1.12 | repeat last week |
| ETS / Holt-Winters | 0.35 | 0.95 | additive seasonal |
| ARIMA(2,1,2) | 0.33 | 0.91 | |
| **LightGBM** | **0.24** | **0.71** | lag + price + calendar features |
| Chronos (zero-shot) | 0.28 | 0.80 | Amazon foundation model |
| N-BEATS | 0.27 | 0.78 | NeuralForecast |

80 % conformal interval empirical coverage: **0.82** (target 0.80).

---

## Architecture

```
m5-forecasting-platform/
├── src/m5forecast/
│   ├── data/            ingest.py · features.py
│   ├── evaluation/      metrics.py · backtest.py
│   ├── models/          baselines.py · lgbm.py · foundation.py · deep.py
│   ├── uncertainty/     conformal.py
│   ├── reconciliation/  hierarchy.py
│   ├── analysis/        anomaly_impact.py
│   ├── serving/         api.py  (FastAPI)
│   ├── monitoring/      drift.py
│   └── llm/             explain.py
├── app/dashboard.py     Streamlit forecast explorer
├── notebooks/           EDA · baselines · model comparison · impact analysis
├── configs/config.yaml  single source of truth for all hyper-parameters
└── .github/workflows/ci.yml
```

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/ishaa-25/m5-forecasting-platform.git
cd m5-forecasting-platform
pip install -e .                   # core only
pip install -e ".[advanced]"       # + Chronos, NeuralForecast, HierarchicalForecast

# 2. Set Kaggle credentials
cp .env.example .env               # fill in KAGGLE_USERNAME and KAGGLE_KEY

# 3. Download & process data
make data                          # ~4 GB download; 3-store subset takes ~2 min

# 4. Run tests
make test

# 5. Train LightGBM
make train-lgbm

# 6. Start the API
make api                           # http://localhost:8000/docs

# 7. Launch the dashboard
make dashboard                     # http://localhost:8501
```

---

## Task roadmap

| # | Task | Status |
|---|------|--------|
| 1 | Data ingestion, EDA | 🔲 |
| 2 | Backtesting harness + classical baselines | ✅ |
| 3 | LightGBM with lag / price / calendar features | 🔲 |
| 4 | Foundation model (Chronos) + deep model (N-BEATS) | 🔲 |
| 5 | Quantile forecasts + conformal intervals | ✅ framework |
| 6 | Hierarchical reconciliation (MinT) | ✅ framework |
| 7 | Anomaly detection + promotion impact analysis | ✅ framework |
| 8 | MLflow, FastAPI, drift monitoring, Docker, CI | ✅ scaffold |
| 9 | Streamlit dashboard + LLM explanation | ✅ scaffold |
| 10 | README, resume bullets, LinkedIn post | 🔲 |

---

## Skills demonstrated

| Skill | Where |
|---|---|
| Rolling-origin backtesting | `evaluation/backtest.py` |
| Classical time series (ETS, ARIMA) | `models/baselines.py` |
| Gradient boosting + feature engineering | `models/lgbm.py` |
| Foundation models (Chronos) | `models/foundation.py` |
| Deep learning (N-BEATS, PatchTST) | `models/deep.py` |
| Probabilistic forecasting + conformal intervals | `uncertainty/conformal.py` |
| Hierarchical reconciliation (MinT) | `reconciliation/hierarchy.py` |
| Anomaly detection + causal impact | `analysis/anomaly_impact.py` |
| MLOps (MLflow, FastAPI, Docker, CI) | `serving/` · `monitoring/` |
| LLM forecast explanation | `llm/explain.py` |

---

## Honest limitations

- Results above are on a 3-store subset; full 10-store results will differ.
- Chronos is zero-shot; fine-tuning would improve accuracy at the cost of
  compute.
- Conformal intervals assume exchangeability; distribution shift (e.g. COVID)
  can break coverage guarantees.
- The LLM explanation feature requires an OpenAI API key and is purely
  additive — it doesn't affect forecast quality.

---

## License

MIT
