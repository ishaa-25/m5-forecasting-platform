"""
Task 9 — Streamlit dashboard.

Features:
- Store / department / item drill-down
- Forecast fan (point + confidence intervals)
- Model comparison table
- Anomaly overlay
- Optional LLM explanation button
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="M5 Forecast Platform",
    page_icon="📈",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


# ── Data loader (cached) ───────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame | None:
    parquet = PROCESSED / "m5_long.parquet"
    if not parquet.exists():
        return None
    return pd.read_parquet(parquet)


# ── Forecast stub (replace with real model calls after Task 3) ────────────────

def make_stub_forecast(history: pd.Series, horizon: int = 28) -> dict:
    """
    Generate a synthetic forecast fan for UI development.
    Replace this with real model inference once training is done.
    """
    rng = np.random.default_rng(42)
    last = float(history.iloc[-1]) if len(history) else 10.0
    noise = rng.normal(0, last * 0.05, horizon)
    point = np.clip(last + noise.cumsum() * 0.1, 0, None)
    width = last * 0.15
    return {
        "point": point,
        "lower_80": np.clip(point - 1.28 * width, 0, None),
        "upper_80": point + 1.28 * width,
        "lower_95": np.clip(point - 1.96 * width, 0, None),
        "upper_95": point + 1.96 * width,
    }


# ── Plot helpers ───────────────────────────────────────────────────────────────

def plot_forecast_fan(
    history: pd.Series,
    forecast: dict,
    title: str = "Sales Forecast",
    n_history: int = 90,
) -> go.Figure:
    hist = history.iloc[-n_history:]
    future_dates = pd.date_range(
        start=hist.index[-1] + pd.Timedelta(days=1),
        periods=len(forecast["point"]),
        freq="D",
    )

    fig = go.Figure()

    # History
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        mode="lines", name="Actual",
        line={"color": "#2563eb", "width": 1.5},
    ))

    # 95% interval
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(forecast["upper_95"]) + list(forecast["lower_95"][::-1]),
        fill="toself",
        fillcolor="rgba(147,197,253,0.25)",
        line={"color": "rgba(0,0,0,0)"},
        showlegend=True, name="95% interval",
    ))

    # 80% interval
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(forecast["upper_80"]) + list(forecast["lower_80"][::-1]),
        fill="toself",
        fillcolor="rgba(96,165,250,0.35)",
        line={"color": "rgba(0,0,0,0)"},
        showlegend=True, name="80% interval",
    ))

    # Point forecast
    fig.add_trace(go.Scatter(
        x=future_dates, y=forecast["point"],
        mode="lines", name="Point forecast",
        line={"color": "#f97316", "width": 2, "dash": "dot"},
    ))

    # Cutoff line
    fig.add_vline(
        x=str(hist.index[-1]),
        line_dash="dash", line_color="grey",
        annotation_text="Forecast start",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Units sold",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        height=420,
    )
    return fig


# ── Main app ───────────────────────────────────────────────────────────────────

st.title("📈 M5 Forecast Platform")
st.caption("Probabilistic hierarchical forecasting on M5 Walmart data")

df = load_data()

if df is None:
    st.warning(
        "⚠️ No processed data found at `data/processed/m5_long.parquet`. "
        "Run `make data` to download and process the M5 dataset."
    )
    st.stop()

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    stores = sorted(df["store_id"].unique())
    store = st.selectbox("Store", stores)

    depts = sorted(df[df["store_id"] == store]["dept_id"].unique())
    dept = st.selectbox("Department", ["(all)"] + depts)

    items = sorted(
        df[(df["store_id"] == store) & (df["dept_id"] == dept if dept != "(all)" else True)]["item_id"].unique()
    )
    item = st.selectbox("Item", ["(aggregate)"] + list(items))

    horizon = st.slider("Forecast horizon (days)", 7, 56, 28)
    n_history = st.slider("History to show (days)", 30, 365, 90)

    st.markdown("---")
    model_choice = st.selectbox(
        "Model",
        ["lgbm", "ets", "arima", "seasonal_naive", "chronos", "nbeats"],
    )
    show_llm = st.checkbox("Show LLM explanation", value=False)

# ── Filter series ──────────────────────────────────────────────────────────────
mask = df["store_id"] == store
if dept != "(all)":
    mask &= df["dept_id"] == dept
if item != "(aggregate)":
    mask &= df["item_id"] == item

series = df[mask].groupby("date")["sales"].sum().sort_index()

if series.empty:
    st.error("No data for the selected filters.")
    st.stop()

# ── Main forecast plot ─────────────────────────────────────────────────────────
forecast_data = make_stub_forecast(series, horizon=horizon)

col1, col2, col3 = st.columns(3)
col1.metric("Mean forecast (daily)", f"{np.mean(forecast_data['point']):.1f} units")
col2.metric(
    "80% interval width",
    f"{np.mean(forecast_data['upper_80'] - forecast_data['lower_80']):.1f} units",
)
col3.metric("History length", f"{len(series)} days")

title = f"{item if item != '(aggregate)' else dept if dept != '(all)' else store} — {horizon}-day forecast"
fig = plot_forecast_fan(series, forecast_data, title=title, n_history=n_history)
st.plotly_chart(fig, use_container_width=True)

# ── Tabbed detail section ─────────────────────────────────────────────────────
tab_metrics, tab_anomaly, tab_explain = st.tabs(["📊 Metrics", "🚨 Anomalies", "💬 Explain"])

with tab_metrics:
    st.subheader("Model comparison (placeholder — run backtests first)")
    placeholder_metrics = pd.DataFrame({
        "Model": ["seasonal_naive", "ets", "arima", "lgbm", "chronos"],
        "WAPE": [0.42, 0.35, 0.33, 0.24, 0.28],
        "RMSSE": [1.12, 0.95, 0.91, 0.71, 0.80],
        "MAE": [12.1, 9.8, 9.3, 6.9, 7.8],
    })
    st.dataframe(
        placeholder_metrics.style.highlight_min(["WAPE", "RMSSE", "MAE"], color="#d1fae5"),
        use_container_width=True,
    )

with tab_anomaly:
    st.subheader("Anomaly detection")
    try:
        from m5forecast.analysis.anomaly_impact import detect_anomalies_zscore
        flags = detect_anomalies_zscore(series)
        anomalies = series[flags]
        if anomalies.empty:
            st.success("No anomalies detected in the selected series.")
        else:
            fig_a = go.Figure()
            fig_a.add_trace(go.Scatter(
                x=series.index, y=series.values, mode="lines",
                name="Sales", line={"color": "#2563eb"},
            ))
            fig_a.add_trace(go.Scatter(
                x=anomalies.index, y=anomalies.values, mode="markers",
                name="Anomaly", marker={"color": "red", "size": 8, "symbol": "x"},
            ))
            fig_a.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig_a, use_container_width=True)
            st.write(f"**{len(anomalies)} anomalous days** detected (Z-score > 3)")
    except Exception as e:
        st.warning(f"Anomaly detection unavailable: {e}")

with tab_explain:
    if show_llm:
        st.subheader("Forecast explanation")
        with st.spinner("Generating explanation…"):
            try:
                from m5forecast.llm.explain import generate_explanation
                explanation = generate_explanation(
                    item_id=item if item != "(aggregate)" else "aggregate",
                    store_id=store,
                    horizon=horizon,
                    point_forecast=forecast_data["point"],
                    lower=forecast_data["lower_80"],
                    upper=forecast_data["upper_80"],
                    top_features=[("lag_7", 0.18), ("roll_mean_28", 0.14), ("sell_price", 0.10)],
                    context="",
                )
                st.info(explanation)
            except Exception as e:
                st.warning(f"Explanation unavailable: {e}")
    else:
        st.info("Enable 'Show LLM explanation' in the sidebar to generate a narrative.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("M5 Forecasting Platform · Built on M5 Walmart data · [GitHub](#)")
