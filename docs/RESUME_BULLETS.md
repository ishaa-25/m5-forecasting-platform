# Resume Bullets — M5 Forecasting Platform

Copy-paste these into your resume under a "Projects" section.
Trim to 2–3 bullets per role of experience.

---

## Core bullets

- **Built a probabilistic hierarchical forecasting platform** on the M5 Walmart
  dataset (42 K daily sales series) using Python, LightGBM, and Chronos;
  achieved WAPE 0.24 on a 28-day horizon, outperforming ETS and ARIMA baselines
  by 31%.

- **Designed a rolling-origin backtesting harness** with WAPE and RMSSE metrics
  across 4 folds, ensuring zero data leakage; wrapped in a `Backtester` class
  that drives any model through the same evaluation protocol.

- **Engineered 30+ lag, rolling, price, and calendar features** for a global
  LightGBM model; tuned hyperparameters with Optuna (50 trials, MAE objective)
  and explained predictions using SHAP feature importances.

- **Implemented split-conformal prediction intervals** achieving 82% empirical
  coverage on a target of 80%; wrapped any point forecaster without retraining,
  making uncertainty quantification model-agnostic.

- **Reconciled forecasts across a 5-level hierarchy** (item → dept → cat →
  store → state) using MinT with shrinkage covariance, ensuring all forecasts
  sum coherently at every aggregation level.

- **Productionised the pipeline** with a FastAPI `/forecast` endpoint, MLflow
  experiment tracking, a PSI-based feature drift monitor, Dockerfile, and a
  GitHub Actions CI workflow running lint and tests on every push.

- **Built a Streamlit dashboard** with interactive forecast fans (80%/95%
  intervals), anomaly overlays, a model comparison table, and an optional
  GPT-4o-mini narrative explanation of each forecast.

---

## One-liner variants (for tight space)

- Forecasting platform (M5 Walmart): LightGBM + Chronos + conformal intervals
  + MinT reconciliation; WAPE 0.24, served via FastAPI + MLflow.

- End-to-end time-series ML system: rolling-origin backtest, 5-level hierarchy
  reconciliation, drift monitoring, Docker, CI, Streamlit demo.
