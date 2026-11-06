"""Adaptive demand forecasting with rolling-origin model selection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, data

_MAX_BACKTEST_POINTS = 12
_MIN_TRAINING_POINTS = 3
_SEASONAL_PERIODS = {
    "daily": 7,
    "weekly": 52,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1,
}
_MODELS = ("last_value", "moving_average", "linear_trend", "seasonal_naive")


@dataclass
class ForecastResult:
    """Forecast values plus model-selection evidence and history status."""

    forecast: pd.DataFrame
    selected_model: str | None
    validation_mae: float | None
    history_periods: int
    status: str


def _predict(history: np.ndarray, periods: int, model: str, seasonal_period: int) -> np.ndarray:
    if model == "last_value":
        values = np.repeat(history[-1], periods)
    elif model == "moving_average":
        values = np.repeat(history[-min(3, len(history)):].mean(), periods)
    elif model == "linear_trend":
        x = np.arange(len(history), dtype=float)
        slope, intercept = np.polyfit(x, history, 1)
        future_x = np.arange(len(history), len(history) + periods, dtype=float)
        values = intercept + slope * future_x
    elif model == "seasonal_naive":
        steps = np.arange(periods) % seasonal_period
        values = history[-seasonal_period:][steps]
    else:
        raise ValueError(f"Unknown forecasting model: {model}")
    return np.maximum(values, 0.0)


def _choose_model(history: np.ndarray, freq: str) -> tuple[str, float]:
    seasonal_period = _SEASONAL_PERIODS[freq]
    models = list(_MODELS[:-1])
    if seasonal_period > 1 and len(history) >= 2 * seasonal_period:
        models.append("seasonal_naive")

    first_origin = max(
        _MIN_TRAINING_POINTS,
        len(history) - _MAX_BACKTEST_POINTS,
        seasonal_period if "seasonal_naive" in models else 0,
    )
    errors = {model: [] for model in models}
    for origin in range(first_origin, len(history)):
        training = history[:origin]
        actual = history[origin]
        for model in models:
            predicted = _predict(training, 1, model, seasonal_period)[0]
            errors[model].append(abs(float(actual) - float(predicted)))

    scores = {model: float(np.mean(model_errors)) for model, model_errors in errors.items()}
    selected = min(models, key=lambda model: (scores[model], _MODELS.index(model)))
    return selected, scores[selected]


def forecast_demand(
    df: pd.DataFrame,
    periods: int = 3,
    freq: str = "monthly",
    product: str | None = None,
) -> ForecastResult:
    """Forecast total or per-product units sold for the requested future periods.

    Candidate models are compared using expanding-window, one-step-ahead MAE.
    Seasonal naive is eligible only after at least two complete seasons. The
    configured minimum history is required before producing any forecast.
    Negative predictions are clipped to zero; recommendations are not made.
    """
    if freq not in config.FREQUENCIES:
        raise ValueError(f"freq must be one of {list(config.FREQUENCIES)}")
    if isinstance(periods, bool) or not isinstance(periods, int) or periods <= 0:
        raise ValueError("periods must be a positive integer.")

    sales = df[df["record_type"] == "sale"]
    if product is not None:
        sales = sales[sales["product"] == product]
    history = data.aggregate(sales, freq, value="quantity")
    if history.empty:
        return ForecastResult(
            pd.DataFrame(columns=["forecast_demand"]).rename_axis("period"),
            None,
            None,
            0,
            "no_sales",
        )

    values = history["quantity"].to_numpy(dtype=float)
    minimum_history = config.RULES["min_history_points"]
    if len(values) < minimum_history:
        return ForecastResult(
            pd.DataFrame(columns=["forecast_demand"]).rename_axis("period"),
            None,
            None,
            len(values),
            "insufficient_history",
        )

    selected, validation_mae = _choose_model(values, freq)
    predictions = _predict(values, periods, selected, _SEASONAL_PERIODS[freq])
    alias = config.FREQUENCIES[freq]
    future_periods = pd.date_range(
        start=history.index[-1], periods=periods + 1, freq=alias
    )[1:]
    forecast = pd.DataFrame(
        {"forecast_demand": predictions}, index=future_periods.rename("period")
    )
    return ForecastResult(
        forecast=forecast,
        selected_model=selected,
        validation_mae=validation_mae,
        history_periods=len(values),
        status="ok",
    )
