"""Explainable, rule-based business insights from BizTrend analysis outputs."""
from __future__ import annotations

import math

from dataclasses import dataclass

import pandas as pd

from . import anomaly, analytics, config, data, forecast, inventory

_PRIORITY_ORDER = {"urgent": 0, "review": 1, "watch": 2, "info": 3}
_DEMAND_CHANGE_SIGNAL_PCT = 10.0


@dataclass(frozen=True)
class BusinessInsight:
    """One evidence-backed, advisory business insight."""

    priority: str
    category: str
    title: str
    explanation: str
    recommended_action: str
    evidence: dict[str, object]


@dataclass
class AssistantReport:
    """Analysis outputs and generated insights for one reporting run."""

    sales_summary: dict[str, float | int]
    financial_performance: pd.DataFrame
    demand_forecast: forecast.ForecastResult
    inventory_recommendations: pd.DataFrame
    transaction_anomalies: pd.DataFrame
    demand_anomalies: pd.DataFrame
    insights: list[BusinessInsight]


def _validate_transactions(transactions: pd.DataFrame) -> None:
    missing = set(data.TRANSACTION_COLUMNS) - set(transactions.columns)
    if missing:
        raise ValueError(
            "transactions must use the canonical schema; "
            f"missing required columns: {sorted(missing)}"
        )
    if transactions.empty:
        return
    invalid_types = set(transactions["record_type"].dropna().unique()) - set(config.RECORD_TYPES)
    if invalid_types:
        raise ValueError(f"transactions contains unsupported record types: {sorted(invalid_types)}")


def _add_insight(
    insights: list[BusinessInsight],
    *,
    priority: str,
    category: str,
    title: str,
    explanation: str,
    recommended_action: str,
    evidence: dict[str, object],
) -> None:
    insights.append(BusinessInsight(
        priority=priority,
        category=category,
        title=title,
        explanation=explanation,
        recommended_action=recommended_action,
        evidence=evidence,
    ))


def _generate_insights(
    transactions: pd.DataFrame,
    sales: dict[str, float | int],
    financial: pd.DataFrame,
    demand: forecast.ForecastResult,
    stock: pd.DataFrame,
    transaction_alerts: pd.DataFrame,
    demand_alerts: pd.DataFrame,
    periods: int,
    freq: str,
) -> list[BusinessInsight]:
    insights: list[BusinessInsight] = []

    if not financial.empty:
        latest = financial.iloc[-1]
        period_label = str(financial.index[-1].date())
        if latest["net_profit"] < 0:
            _add_insight(
                insights,
                priority="urgent",
                category="financial",
                title="Latest period shows a net loss",
                explanation=(
                    f"Net profit for {period_label} was "
                    f"{latest['net_profit']:,.2f}."
                ),
                recommended_action=(
                    "Review the period's revenue and expense breakdown before "
                    "making financial decisions."
                ),
                evidence={
                    "period": period_label,
                    "revenue": float(latest["revenue"]),
                    "total_expenses": float(latest["total_expenses"]),
                    "net_profit": float(latest["net_profit"]),
                },
            )
        if bool(latest["expense_growth_warning"]):
            _add_insight(
                insights,
                priority="review",
                category="financial",
                title="Operating expenses are outpacing revenue",
                explanation=(
                    f"For {period_label}, operating expenses grew "
                    f"{latest['operating_expense_growth_pct']:.1f}% while revenue "
                    f"grew {latest['revenue_growth_pct']:.1f}%."
                ),
                recommended_action=(
                    "Review operating-expense categories and compare the change "
                    "with the business's budget and operating plans."
                ),
                evidence={
                    "period": period_label,
                    "revenue_growth_pct": float(latest["revenue_growth_pct"]),
                    "operating_expense_growth_pct": float(
                        latest["operating_expense_growth_pct"]
                    ),
                },
            )

    product_sales = analytics.sales_by_product(transactions)
    if not product_sales.empty:
        top = product_sales.iloc[0]
        _add_insight(
            insights,
            priority="info",
            category="sales",
            title=f"{top['product']} leads recorded product revenue",
            explanation=(
                f"It generated {top['revenue']:,.2f} in recorded revenue "
                f"across {int(top['transaction_count'])} transaction lines."
            ),
            recommended_action=(
                "Consider this product's performance when reviewing sales and "
                "inventory plans."
            ),
            evidence={
                "product": str(top["product"]),
                "revenue": float(top["revenue"]),
                "units_sold": float(top["units_sold"]),
            },
        )

    if demand.status == "ok" and len(demand.forecast) == periods:
        history = analytics.sales_trends(transactions, freq=freq)
        observed = history["units_sold"].tail(periods)
        if len(observed) == periods and observed.sum() > 0:
            forecast_units = float(demand.forecast["forecast_demand"].sum())
            observed_units = float(observed.sum())
            change_pct = (forecast_units / observed_units - 1) * 100
            if abs(change_pct) >= _DEMAND_CHANGE_SIGNAL_PCT:
                direction = "higher" if change_pct > 0 else "lower"
                _add_insight(
                    insights,
                    priority="watch",
                    category="demand",
                    title=f"Forecast demand is {direction} than recent sales",
                    explanation=(
                        f"The next {periods} {freq} periods forecast {forecast_units:,.1f} "
                        f"units, {change_pct:+.1f}% versus the preceding "
                        f"{periods} {freq} periods ({observed_units:,.1f} units)."
                    ),
                    recommended_action=(
                        "Review the forecast alongside recent business context; "
                        "it is an estimate, not a commitment."
                    ),
                    evidence={
                        "forecast_units": forecast_units,
                        "comparison_units": observed_units,
                        "change_pct": change_pct,
                        "model": demand.selected_model,
                        "validation_mae": demand.validation_mae,
                    },
                )

    if not stock.empty:
        for item in stock[stock["reorder_status"] == "reorder"].itertuples(index=False):
            _add_insight(
                insights,
                priority="review",
                category="inventory",
                title=f"Review replenishment for {item.product}",
                explanation=(
                    f"Recorded stock ({item.stock_on_hand:g}) is at or below "
                    f"the estimated reorder point ({math.ceil(item.reorder_point):,})."
                ),
                recommended_action=(
                    f"Review the suggested {item.recommended_order_qty} units "
                    "against actual stock and supplier constraints; no order "
                    "has been placed."
                ),
                evidence={
                    "product": item.product,
                    "stock_on_hand": float(item.stock_on_hand),
                    "reorder_point": float(item.reorder_point),
                    "recommended_order_qty": int(item.recommended_order_qty),
                    "estimated_order_cost": float(item.estimated_order_cost),
                    "forecast_model": item.forecast_model,
                },
            )
        for item in stock[stock["reorder_status"] == "insufficient_history"].itertuples(index=False):
            _add_insight(
                insights,
                priority="watch",
                category="inventory",
                title=f"Demand history is limited for {item.product}",
                explanation=(
                    f"Only {item.history_periods} daily observations are available; "
                    "the forecasting minimum has not been met."
                ),
                recommended_action=(
                    "Collect more sales history and verify stock manually before "
                    "deciding whether to replenish."
                ),
                evidence={
                    "product": item.product,
                    "history_periods": int(item.history_periods),
                    "status": item.reorder_status,
                },
            )

    for row in transaction_alerts.itertuples(index=False):
        _add_insight(
            insights,
            priority="review",
            category="anomaly",
            title=f"Review unusual transaction for {row.product or row.category}",
            explanation=(
                f"The {row.anomaly_metric} value {getattr(row, row.anomaly_metric):,.2f} "
                f"was unusual within {row.comparison_group} records."
            ),
            recommended_action=(
                "Verify the source record and business context; an anomaly is a "
                "statistical flag, not evidence of wrongdoing."
            ),
            evidence={
                "date": str(pd.Timestamp(row.date).date()),
                "record_type": row.record_type,
                "product": row.product,
                "category": row.category,
                "metric": row.anomaly_metric,
                "score": float(row.anomaly_score),
                "peer_count": int(row.peer_count),
            },
        )

    for row in demand_alerts.itertuples(index=False):
        _add_insight(
            insights,
            priority="watch",
            category="anomaly",
            title=f"Review unusual daily demand for {row.product}",
            explanation=(
                f"Sales were {row.units_sold:g} units versus a recent active-day "
                f"median of {row.expected_units:g}."
            ),
            recommended_action=(
                "Check for promotions, one-off orders, recording errors, or "
                "other context before interpreting this demand signal."
            ),
            evidence={
                "date": str(pd.Timestamp(row.date).date()),
                "product": row.product,
                "units_sold": float(row.units_sold),
                "expected_units": float(row.expected_units),
                "score": float(row.anomaly_score),
                "history_points": int(row.history_points),
            },
        )

    if not transactions.empty and not insights:
        _add_insight(
            insights,
            priority="info",
            category="overview",
            title="No configured review signals were triggered",
            explanation=(
                f"Analysis covered {int(sales['transaction_count'])} sales rows. "
                "This does not establish that all business activity is normal."
            ),
            recommended_action=(
                "Continue reviewing source records and business context; "
                "rule-based checks cover only their configured thresholds."
            ),
            evidence={
                "sales_rows": int(sales["transaction_count"]),
                "frequency": freq,
            },
        )

    return sorted(
        insights,
        key=lambda item: (
            _PRIORITY_ORDER[item.priority],
            item.category,
            item.title,
        ),
    )


def generate_business_report(
    transactions: pd.DataFrame,
    products: pd.DataFrame | None = None,
    freq: str = "monthly",
    forecast_periods: int = 3,
) -> AssistantReport:
    """Generate transparent insights and source analyses for one business.

    This assistant is deterministic and rule-based. It does not call an LLM,
    execute transactions, or place purchasing commitments. Input transactions
    must use the canonical schema returned by the data layer.
    """
    _validate_transactions(transactions)
    if freq not in config.FREQUENCIES:
        raise ValueError(f"freq must be one of {list(config.FREQUENCIES)}")
    if (
        isinstance(forecast_periods, bool)
        or not isinstance(forecast_periods, int)
        or forecast_periods <= 0
    ):
        raise ValueError("forecast_periods must be a positive integer.")

    sales_summary = analytics.sales_summary(transactions)
    financial = analytics.financial_performance(transactions, freq=freq)
    demand = forecast.forecast_demand(
        transactions, periods=forecast_periods, freq=freq
    )
    product_data = (
        products if products is not None
        else pd.DataFrame(columns=data.PRODUCT_COLUMNS)
    )
    stock = inventory.inventory_recommendations(transactions, product_data)
    transaction_alerts = anomaly.detect_transaction_anomalies(transactions)
    demand_alerts = anomaly.detect_daily_demand_anomalies(transactions)
    insights = _generate_insights(
        transactions,
        sales_summary,
        financial,
        demand,
        stock,
        transaction_alerts,
        demand_alerts,
        forecast_periods,
        freq,
    )
    return AssistantReport(
        sales_summary=sales_summary,
        financial_performance=financial,
        demand_forecast=demand,
        inventory_recommendations=stock,
        transaction_anomalies=transaction_alerts,
        demand_anomalies=demand_alerts,
        insights=insights,
    )
