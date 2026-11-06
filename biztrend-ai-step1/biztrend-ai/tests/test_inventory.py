import numpy as np
import pandas as pd
import pytest

from biztrend import config, data, inventory


def _transactions(units, product="A"):
    raw = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=len(units), freq="D"),
        "record_type": "sale",
        "product": product,
        "quantity": units,
        "unit_price": 10.0,
        "amount": np.asarray(units, dtype=float) * 10,
    })
    cleaned, _ = data.clean_transactions(raw)
    return cleaned


def _products(stock=20, lead_time=3, cost=2.0):
    return pd.DataFrame([{
        "product": "A",
        "unit_cost": cost,
        "stock_on_hand": stock,
        "lead_time_days": lead_time,
    }])


def test_reorder_quantity_and_cost_for_steady_demand():
    result = inventory.inventory_recommendations(_transactions([10] * 10), _products())
    item = result.iloc[0]

    assert item["reorder_status"] == "reorder"
    assert item["forecast_model"] == "last_value"
    assert item["expected_lead_time_demand"] == pytest.approx(30)
    assert item["safety_stock"] == pytest.approx(0)
    assert item["reorder_point"] == pytest.approx(30)
    assert item["recommended_order_qty"] == 20
    assert item["estimated_order_cost"] == pytest.approx(40)


def test_stock_above_reorder_point_does_not_recommend_order():
    result = inventory.inventory_recommendations(_transactions([10] * 10), _products(stock=50))

    assert result.iloc[0]["reorder_status"] == "sufficient"
    assert result.iloc[0]["recommended_order_qty"] == 0
    assert result.iloc[0]["estimated_order_cost"] == 0


def test_safety_stock_uses_daily_variability_and_service_level():
    units = [2, 8, 2, 8, 2, 8, 2, 8, 2, 8]
    result = inventory.inventory_recommendations(_transactions(units), _products(stock=0))
    expected_std = float(np.std(units, ddof=1))
    expected_safety_stock = (
        config.RULES["inventory_service_level_z"] * expected_std * np.sqrt(3)
    )

    assert result.iloc[0]["daily_demand_std"] == pytest.approx(expected_std)
    assert result.iloc[0]["safety_stock"] == pytest.approx(expected_safety_stock)
    assert result.iloc[0]["recommended_order_qty"] > 0


def test_products_without_sales_or_enough_history_are_not_recommended():
    no_history = pd.DataFrame(columns=data.TRANSACTION_COLUMNS)
    result = inventory.inventory_recommendations(no_history, _products())
    assert result.iloc[0]["reorder_status"] == "no_sales"
    assert result.iloc[0]["recommended_order_qty"] == 0

    short_history = inventory.inventory_recommendations(
        _transactions([10] * 5), _products()
    )
    assert short_history.iloc[0]["reorder_status"] == "insufficient_history"
    assert short_history.iloc[0]["recommended_order_qty"] == 0


@pytest.mark.parametrize(
    "products",
    [
        pd.DataFrame([{"product": "A", "stock_on_hand": -1, "unit_cost": 2, "lead_time_days": 3}]),
        pd.DataFrame([{"product": "A", "stock_on_hand": 1, "unit_cost": 2, "lead_time_days": 0}]),
        pd.DataFrame([{"product": "A", "stock_on_hand": 1, "unit_cost": float("inf"), "lead_time_days": 3}]),
        pd.DataFrame([
            {"product": "A", "stock_on_hand": 1, "unit_cost": 2, "lead_time_days": 3},
            {"product": "A", "stock_on_hand": 2, "unit_cost": 2, "lead_time_days": 3},
        ]),
    ],
)
def test_invalid_product_inventory_values_raise(products):
    with pytest.raises(ValueError):
        inventory.inventory_recommendations(_transactions([10] * 10), products)


def test_missing_product_columns_raise():
    with pytest.raises(ValueError, match="missing required columns"):
        inventory.inventory_recommendations(
            _transactions([10] * 10), pd.DataFrame({"product": ["A"]})
        )
