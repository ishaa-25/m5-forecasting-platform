"""
Task 9 (optional) — LLM-powered forecast explanation.

Given a forecast, its confidence interval, and key feature importances,
generate a plain-English business narrative using an OpenAI-compatible
chat model.

Falls back to a rule-based template if the openai package is unavailable
or no API key is set — so the rest of the platform works without it.

Install with: pip install -e ".[llm]"
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from string import Template

import numpy as np

logger = logging.getLogger(__name__)


# ── Rule-based fallback template ──────────────────────────────────────────────

_TEMPLATE = Template(
    "Forecast summary for $item_id in $store_id:\n"
    "\n"
    "Over the next $horizon days, sales are expected to average $mean_forecast units/day "
    "(80% interval: $lower–$upper).\n"
    "\n"
    "Key drivers: $top_features.\n"
    "\n"
    "Notable context: $context.\n"
    "\n"
    "Confidence: $confidence_label. "
    "The interval width is $interval_width units, "
    "suggesting $uncertainty_comment."
)


def _rule_based_explanation(
    item_id: str,
    store_id: str,
    horizon: int,
    point_forecast: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    top_features: list[tuple[str, float]],
    context: str = "",
) -> str:
    mean_fc = round(float(np.mean(point_forecast)), 1)
    mean_lower = round(float(np.mean(lower)), 1)
    mean_upper = round(float(np.mean(upper)), 1)
    width = mean_upper - mean_lower

    feature_str = ", ".join(
        f"{name} (importance: {val:.2f})" for name, val in top_features[:3]
    )
    if width < 5:
        uncertainty = "low uncertainty — stable demand pattern"
        confidence = "High"
    elif width < 20:
        uncertainty = "moderate uncertainty — typical for this category"
        confidence = "Medium"
    else:
        uncertainty = "high uncertainty — consider wider safety stock"
        confidence = "Low"

    return _TEMPLATE.substitute(
        item_id=item_id,
        store_id=store_id,
        horizon=horizon,
        mean_forecast=mean_fc,
        lower=mean_lower,
        upper=mean_upper,
        top_features=feature_str or "no feature data available",
        context=context or "no special events in the forecast window",
        confidence_label=confidence,
        interval_width=round(width, 1),
        uncertainty_comment=uncertainty,
    )


# ── LLM-powered explanation ────────────────────────────────────────────────────

def generate_explanation(
    item_id: str,
    store_id: str,
    horizon: int,
    point_forecast: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    top_features: list[tuple[str, float]],
    context: str = "",
    model: str = "gpt-4o-mini",
    use_llm: bool = True,
) -> str:
    """
    Generate a plain-English forecast explanation.

    Attempts to use the OpenAI API if available and use_llm=True,
    falls back to the rule-based template otherwise.

    Parameters
    ----------
    item_id / store_id : identifiers shown in the narrative
    horizon            : forecast horizon (days)
    point_forecast     : 1-D array of point forecasts
    lower / upper      : 1-D arrays defining the 80% prediction interval
    top_features       : list of (feature_name, importance_value) tuples
    context            : any extra context (events, promotions)
    model              : OpenAI model name
    use_llm            : set False to force the rule-based fallback
    """
    if not use_llm:
        return _rule_based_explanation(
            item_id, store_id, horizon, point_forecast, lower, upper, top_features, context
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.info("OPENAI_API_KEY not set; using rule-based explanation.")
        return _rule_based_explanation(
            item_id, store_id, horizon, point_forecast, lower, upper, top_features, context
        )

    try:
        from openai import OpenAI  # type: ignore[import]
    except ImportError:
        logger.info("openai package not installed; using rule-based explanation.")
        return _rule_based_explanation(
            item_id, store_id, horizon, point_forecast, lower, upper, top_features, context
        )

    client = OpenAI(api_key=api_key)

    mean_fc = round(float(np.mean(point_forecast)), 1)
    mean_lower = round(float(np.mean(lower)), 1)
    mean_upper = round(float(np.mean(upper)), 1)
    feature_str = "; ".join(f"{n} ({v:.2f})" for n, v in top_features[:5])

    system_prompt = (
        "You are a data science assistant helping a retail analyst understand "
        "a sales forecast. Be concise, factual, and use plain business language. "
        "Avoid technical jargon. Stick to 3–5 sentences."
    )
    user_prompt = (
        f"Explain this forecast for item {item_id} at store {store_id}:\n"
        f"- Horizon: {horizon} days\n"
        f"- Mean forecast: {mean_fc} units/day\n"
        f"- 80% prediction interval: {mean_lower}–{mean_upper} units/day\n"
        f"- Top model features: {feature_str}\n"
        f"- Context: {context or 'none'}\n"
        "What should the store manager know?"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("OpenAI call failed (%s); using rule-based fallback.", e)
        return _rule_based_explanation(
            item_id, store_id, horizon, point_forecast, lower, upper, top_features, context
        )
