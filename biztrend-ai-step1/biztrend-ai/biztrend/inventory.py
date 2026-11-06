"""Inventory replenishment estimates based on demand forecasts and lead times."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config, data, forecast

INVENTORY_COLUMNS = [
    "product",
    "stock_on_hand",
    "lead_time_days",
    "average_daily_demand",
    "daily_demand_std",
    "expected_lead_time_demand",
    "safety_stock",
    "reorder_point",
    "recommended_order_qty",
    "reorder_status",
    "unit_cost",
    "estimated_order_cost",
    "forecast_model",
    "forecast_mae",
    "history_periods",
]


def _validate_products(products: pd.DataFrame) -> pd.DataFrame:
    required = set(data.PRODUCT_COLUMNS)
    missing = required - set(products.columns)
    if missing:
        raise ValueError(f"Products table is missing required columns: {sorted(missing)}")

    result = products[data.PRODUCT_COLUMNS].copy()
    result["product"] = result["product"].fillna("").astype(str).str.strip()
    if result["product"].eq("").any():
        raise ValueError("Product names must not be empty.")
    if result["product"].duplicated().any():
        raise ValueError("Product names must be unique in the products table.")

    for column in ("unit_cost", "stock_on_hand", "lead_time_days"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(result[column]).all():
            raise ValueError(f"{column} must contain finite numeric values.")
        if (result[column] < 0).any():
            raise ValueError(f"{column} must not be negative.")
    if result["lead_time_days"].eq(0).any():
        raise ValueError("lead_time_days must be greater than zero.")
    return result.reset_index(drop=True)


def inventory_recommendations(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate reorder points and advisory order quantities per product.

    Expected demand during supplier lead time comes from the adaptive daily
    demand forecast. Safety stock uses the configured service-level z-score
    times historical daily-demand standard deviation and sqrt(lead time).
    When stock is at or below the reorder point, the recommended quantity
    replenishes to the reorder point plus one average day of demand (a daily
    review-period allowance). Products without sufficient sales history are
    explicitly marked and receive no numeric order recommendation.
    """
    inventory = _validate_products(products)
    if inventory.empty:
        return pd.DataFrame(columns=INVENTORY_COLUMNS)
    if not set(data.TRANSACTION_COLUMNS).issubset(transactions.columns):
        raise ValueError("transactions must use the canonical transaction schema.")

    sales = transactions[transactions["record_type"] == "sale"]
    recommendations: list[dict[str, object]] = []
    z_score = float(config.RULES["inventory_service_level_z"])
    if not math.isfinite(z_score) or z_score < 0:
        raise ValueError("inventory_service_level_z must be finite and nonnegative.")

    for item in inventory.itertuples(index=False):
        lead_time = int(math.ceil(item.lead_time_days))
        product_sales = sales[sales["product"] == item.product]
        history = data.aggregate(product_sales, "daily", value="quantity")
        average_daily_demand = 0.0
        daily_demand_std = 0.0
        expected_lead_time_demand = 0.0
        safety_stock = 0.0
        reorder_point = 0.0
        order_quantity = 0
        reorder_status = "no_sales"
        selected_model = None
        validation_mae = None
        history_periods = 0

        if not history.empty:
            daily_values = history["quantity"].to_numpy(dtype=float)
            history_periods = len(daily_values)
            average_daily_demand = float(np.mean(daily_values))
            daily_demand_std = (
                float(np.std(daily_values, ddof=1)) if history_periods > 1 else 0.0
            )

        result = forecast.forecast_demand(
            transactions, periods=lead_time, freq="daily", product=item.product
        )
        selected_model = result.selected_model
        validation_mae = result.validation_mae
        if result.status == "ok":
            expected_lead_time_demand = float(result.forecast["forecast_demand"].sum())
            safety_stock = z_score * daily_demand_std * math.sqrt(lead_time)
            reorder_point = expected_lead_time_demand + safety_stock
            order_up_to_level = reorder_point + average_daily_demand
            if item.stock_on_hand <= reorder_point:
                order_quantity = max(
                    0, math.ceil(order_up_to_level - item.stock_on_hand)
                )
                reorder_status = "reorder"
            else:
                reorder_status = "sufficient"
        elif result.status == "insufficient_history":
            reorder_status = "insufficient_history"

        recommendations.append({
            "product": item.product,
            "stock_on_hand": float(item.stock_on_hand),
            "lead_time_days": lead_time,
            "average_daily_demand": average_daily_demand,
            "daily_demand_std": daily_demand_std,
            "expected_lead_time_demand": expected_lead_time_demand,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "recommended_order_qty": order_quantity,
            "reorder_status": reorder_status,
            "unit_cost": float(item.unit_cost),
            "estimated_order_cost": order_quantity * float(item.unit_cost),
            "forecast_model": selected_model,
            "forecast_mae": validation_mae,
            "history_periods": history_periods,
        })

    return pd.DataFrame(recommendations, columns=INVENTORY_COLUMNS)
