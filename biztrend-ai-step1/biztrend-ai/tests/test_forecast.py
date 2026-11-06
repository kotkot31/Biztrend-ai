import numpy as np
import pandas as pd
import pytest

from biztrend import data, forecast


def _sales(values, freq="MS", product="A"):
    dates = pd.date_range("2024-01-01", periods=len(values), freq=freq)
    raw = pd.DataFrame({
        "date": dates,
        "record_type": "sale",
        "product": product,
        "quantity": values,
        "unit_price": 10.0,
        "amount": np.asarray(values, dtype=float) * 10,
    })
    cleaned, _ = data.clean_transactions(raw)
    return cleaned


def test_linear_trend_is_selected_from_rolling_validation():
    result = forecast.forecast_demand(_sales(np.arange(1, 13)), periods=3)

    assert result.status == "ok"
    assert result.selected_model == "linear_trend"
    assert result.validation_mae == pytest.approx(0)
    assert result.history_periods == 12
    assert result.forecast.index.tolist() == list(pd.date_range("2025-01-01", periods=3, freq="MS"))
    assert result.forecast["forecast_demand"].tolist() == pytest.approx([13, 14, 15])


def test_seasonal_naive_is_selected_when_it_backtests_best():
    seasonal_values = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2] * 2
    result = forecast.forecast_demand(_sales(seasonal_values), periods=3)

    assert result.selected_model == "seasonal_naive"
    assert result.validation_mae == pytest.approx(0)
    assert result.forecast["forecast_demand"].tolist() == [1, 2, 1]


def test_insufficient_history_returns_explicit_status():
    result = forecast.forecast_demand(_sales([2, 4, 3]), periods=2)

    assert result.status == "insufficient_history"
    assert result.history_periods == 3
    assert result.selected_model is None
    assert result.validation_mae is None
    assert result.forecast.empty


def test_product_filter_and_frequency_alignment():
    rows = pd.concat([
        _sales([2] * 8, freq="W-SUN", product="A"),
        _sales([5] * 8, freq="W-SUN", product="B"),
    ], ignore_index=True)
    result = forecast.forecast_demand(rows, periods=2, freq="weekly", product="B")

    assert result.status == "ok"
    assert result.forecast["forecast_demand"].tolist() == [5, 5]
    assert result.forecast.index.tolist() == list(pd.date_range("2024-03-03", periods=2, freq="W-SUN"))
    assert result.history_periods == 8


@pytest.mark.parametrize(
    ("freq", "date_frequency", "last_period", "expected_forecast_periods"),
    [
        ("quarterly", "QS", "2025-10-01", ["2026-01-01", "2026-04-01"]),
        ("yearly", "YS", "2031-01-01", ["2032-01-01", "2033-01-01"]),
    ],
)
def test_quarterly_and_yearly_forecasts_align_to_period_starts(
    freq, date_frequency, last_period, expected_forecast_periods
):
    rows = _sales(np.arange(1, 9), freq=date_frequency)

    result = forecast.forecast_demand(rows, periods=2, freq=freq)

    assert result.status == "ok"
    assert result.history_periods == 8
    assert result.forecast.index.tolist() == list(pd.to_datetime(expected_forecast_periods))
    assert rows["date"].max().to_period("D").start_time == pd.Timestamp(last_period)


def test_no_matching_sales_and_invalid_arguments():
    empty = pd.DataFrame(columns=data.TRANSACTION_COLUMNS)
    assert forecast.forecast_demand(empty).status == "no_sales"

    with pytest.raises(ValueError):
        forecast.forecast_demand(empty, periods=0)
    with pytest.raises(ValueError):
        forecast.forecast_demand(empty, freq="hourly")


def test_negative_trend_is_clipped_to_nonnegative_demand():
    result = forecast.forecast_demand(_sales([9, 8, 7, 6, 5, 4, 3, 2]), periods=3)

    assert (result.forecast["forecast_demand"] >= 0).all()
