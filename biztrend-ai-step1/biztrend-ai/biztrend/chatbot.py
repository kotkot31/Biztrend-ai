"""Deterministic, data-grounded business Q&A with graceful fallbacks."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import pandas as pd

from . import (
    anomaly,
    analytics,
    assistant as business_assistant,
    config,
    data,
    forecast,
    help_content,
    inventory,
)

_LOGGER = logging.getLogger(__name__)
MAX_QUESTION_LENGTH = 500
DEFAULT_FORECAST_PERIODS = 3
MAX_ANOMALY_TRANSACTIONS = 20_000
MAX_CHAT_TRANSACTIONS = 100_000
MAX_INVENTORY_PRODUCTS = 1_000
UNEXPECTED_ERROR_MESSAGE = (
    "I couldn't finish that answer just now. Your data hasn't been changed. "
    "Please try again, or ask about sales, expenses, forecasts, or inventory."
)
SUPPORTED_TOPICS = (
    "sales and product performance, expenses and profit, demand forecasts, "
    "inventory, unusual transactions, and rule-based business recommendations"
)

_INTENT_TERMS = {
    "inventory": (
        "inventory", "stock", "reorder", "replenish", "restock", "on hand",
        "on-hand", "lead time",
    ),
    "anomaly": (
        "anomal", "unusual", "outlier", "suspicious", "irregular",
    ),
    "forecast": (
        "forecast", "predict", "prediction", "future", "demand", "next month",
        "next quarter", "next year",
    ),
    "financial": (
        "profit", "profitability", "expense", "expenses", "cost", "costs",
        "margin", "loss", "p&l", "financial",
    ),
    "expense_analysis": (
        "expense categor", "expense breakdown", "expense analysis",
        "expense trend", "spending by categor", "spend by categor",
        "expenses by categor", "expense by categor",
        "review expense", "top expenses", "cost breakdown",
    ),
    "sales": (
        "sales", "sale", "revenue", "product", "top", "best-selling",
        "bestselling", "units sold", "selling",
    ),
    "recommendations": (
        "recommend", "recommendation", "what should i", "what can i do",
        "suggestion", "action", "actions", "advice",
    ),
}


@dataclass(frozen=True)
class EvidenceChart:
    """Small, serializable chart specification for an answer's evidence."""

    kind: str
    data: list[dict[str, object]]
    x: str
    y: tuple[str, ...]
    title: str


@dataclass(frozen=True)
class ChatResponse:
    """A bounded chatbot answer, clarification, or fallback."""

    status: str
    answer: str
    category: str
    evidence: dict[str, object] = field(default_factory=dict)
    chart: EvidenceChart | None = None
    suggestions: tuple[str, ...] = ()


def _fallback(status: str, message: str, category: str = "help") -> ChatResponse:
    return ChatResponse(
        status=status,
        answer=message,
        category=category,
        suggestions=(
            "Which products generated the most revenue?",
            "How did profit change in the latest periods?",
            "What does the demand forecast show?",
        ),
    )


def _date_range(transactions: pd.DataFrame) -> str:
    if transactions.empty:
        return "No transaction history is available."
    dates = pd.to_datetime(transactions["date"], errors="coerce").dropna()
    if dates.empty:
        return "Transaction dates are unavailable."
    return f"Historical data: {dates.min().date()} to {dates.max().date()}."


def _evidence_chart(
    frame: pd.DataFrame,
    *,
    kind: str,
    x: str,
    y: tuple[str, ...],
    title: str,
) -> EvidenceChart:
    records: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        safe_row: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                safe_row[key] = value.strftime("%Y-%m-%d")
            elif pd.isna(value):
                safe_row[key] = None
            elif hasattr(value, "item"):
                safe_row[key] = value.item()
            else:
                safe_row[key] = value
        records.append(safe_row)
    return EvidenceChart(kind=kind, data=records, x=x, y=y, title=title)


def _products_in_question(
    question: str,
    transactions: pd.DataFrame,
    products: pd.DataFrame,
) -> list[str]:
    names: set[str] = set()
    if "product" in transactions:
        names.update(
            str(name).strip()
            for name in transactions["product"].dropna().unique()
            if str(name).strip()
        )
    if "product" in products:
        names.update(
            str(name).strip()
            for name in products["product"].dropna().unique()
            if str(name).strip()
        )
    normalized_question = question.casefold()
    return sorted(
        (name for name in names if name.casefold() in normalized_question),
        key=lambda name: (-len(name), name.casefold()),
    )


def _product_scope_requested(question: str) -> bool:
    if re.search(r"\b(?:for|of|about|on)\s+(?:product|item)\b", question):
        return True
    if re.search(r"\b(?:did|does)\s+[a-z0-9][a-z0-9_-]*\s+(?:make|generate|sell)\b", question):
        return True
    match = re.search(r"\b(?:for|of|about|on|from)\s+([a-z0-9][a-z0-9_-]*)", question)
    if match is None:
        return False
    return match.group(1) not in {
        "the", "next", "last", "this", "all", "any", "these", "each",
        "three", "two", "one", "a", "an",
    }


def _detect_intent(question: str) -> str | None:
    matches = [
        intent
        for intent, terms in _INTENT_TERMS.items()
        if any(term in question for term in terms)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer specific data domains over generic recommendation wording.
        priority = (
            "inventory", "anomaly", "forecast", "expense_analysis",
            "financial", "sales",
        )
        for intent in priority:
            if intent in matches:
                return intent
    return None


def _period_label(period: object, freq: str) -> str:
    return analytics.period_label(period, freq)


def _answer_sales(
    transactions: pd.DataFrame,
    question: str,
    products: pd.DataFrame,
    freq: str,
    date_range: str,
) -> ChatResponse:
    sales = analytics.sales_by_product(transactions)
    if sales.empty:
        return _fallback(
            "insufficient_data",
            "I don't have named product sales to answer that yet. Add sales records "
            "with product names, then try again.",
            "sales",
        )
    mentioned = _products_in_question(question, transactions, products)
    if any(
        term in question
        for term in ("trend", "growth", "grow", "increase", "decrease", "changed", "compare")
    ):
        scoped_transactions = transactions[transactions["record_type"] == "sale"]
        if mentioned:
            scoped_transactions = scoped_transactions[
                scoped_transactions["product"].str.casefold() == mentioned[0].casefold()
            ]
        elif _product_scope_requested(question):
            return _fallback(
                "clarification_needed",
                "I couldn't match that product name to the available records. Check "
                "the product name or ask for an overall sales trend.",
                "sales",
            )
        if scoped_transactions.empty:
            return _fallback(
                "insufficient_data",
                "There are no matching sales records to compare.",
                "sales",
            )
        trends = analytics.sales_trends(scoped_transactions, freq=freq)
        if len(trends) < 2:
            return _fallback(
                "insufficient_data",
                f"I need at least two {freq} periods to compare sales trends.",
                "sales",
            )
        measure = (
            "units_sold"
            if any(term in question for term in ("unit", "quantity", "volume"))
            else "revenue"
        )
        previous, latest = trends.iloc[-2], trends.iloc[-1]
        previous_value, latest_value = float(previous[measure]), float(latest[measure])
        delta = latest_value - previous_value
        pct_change = (
            (delta / previous_value) * 100 if previous_value else None
        )
        product_label = f" for {mentioned[0]}" if mentioned else ""
        change_text = (
            f"{pct_change:+.1f}%"
            if pct_change is not None
            else "not defined because the earlier period was zero"
        )
        return ChatResponse(
            status="answered",
            category="sales",
            answer=(
                f"{measure.replace('_', ' ').title()}{product_label} changed "
                f"from {previous_value:,.1f} in {_period_label(trends.index[-2], freq)} "
                f"to {latest_value:,.1f} in {_period_label(trends.index[-1], freq)} "
                f"({delta:+,.1f}; {change_text}). {date_range}"
            ),
            evidence={
                "product": mentioned[0] if mentioned else None,
                "measure": measure,
                "previous_period": _period_label(trends.index[-2], freq),
                "previous_value": previous_value,
                "latest_period": _period_label(trends.index[-1], freq),
                "latest_value": latest_value,
                "change": delta,
                "change_pct": pct_change,
            },
            chart=_evidence_chart(
                pd.DataFrame({
                    "period": [
                        _period_label(trends.index[-2], freq),
                        _period_label(trends.index[-1], freq),
                    ],
                    measure: [previous_value, latest_value],
                }),
                kind="line",
                x="period",
                y=(measure,),
                title=f"{measure.replace('_', ' ').title()}{product_label}",
            ),
        )
    if mentioned:
        product = mentioned[0]
        matched = sales[sales["product"].str.casefold() == product.casefold()]
        if matched.empty:
            return _fallback(
                "insufficient_data",
                f"I found {product} in the product records, but there are no sales "
                "for it in the available transaction history.",
                "sales",
            )
        row = matched.iloc[0]
        return ChatResponse(
            status="answered",
            category="sales",
            answer=(
                f"{product} generated {row['revenue']:,.2f} in recorded revenue "
                f"from {row['units_sold']:,.0f} units across "
                f"{int(row['transaction_count'])} transaction lines. {date_range}"
            ),
            evidence={
                "product": product,
                "revenue": float(row["revenue"]),
                "units_sold": float(row["units_sold"]),
                "transaction_lines": int(row["transaction_count"]),
            },
            chart=_evidence_chart(
                matched[["product", "revenue"]],
                kind="bar",
                x="product",
                y=("revenue",),
                title="Recorded revenue",
            ),
        )
    if _product_scope_requested(question):
        return _fallback(
            "clarification_needed",
            "I couldn't match that product name to the available records. Check the "
            "product name or ask about all products.",
            "sales",
        )

    top = sales.iloc[0]
    if any(term in question for term in ("top", "best-selling", "bestselling", "most")):
        leaders = sales.head(3)
        details = "; ".join(
            f"{row.product}: {row.revenue:,.2f} revenue, {row.units_sold:,.0f} units"
            for row in leaders.itertuples(index=False)
        )
        return ChatResponse(
            status="answered",
            category="sales",
            answer=f"Top products by recorded revenue: {details}. {date_range}",
            evidence={
                "top_products": leaders[
                    ["product", "revenue", "units_sold", "transaction_count"]
                ].to_dict(orient="records"),
            },
            chart=_evidence_chart(
                leaders[["product", "revenue"]],
                kind="bar",
                x="product",
                y=("revenue",),
                title="Top products by revenue",
            ),
        )

    summary = analytics.sales_summary(transactions)
    return ChatResponse(
        status="answered",
        category="sales",
        answer=(
            f"Recorded sales revenue is {summary['revenue']:,.2f} from "
            f"{summary['units_sold']:,.0f} units across "
            f"{summary['transaction_count']} transaction lines. "
            f"The top product by revenue is {top['product']} "
            f"({top['revenue']:,.2f}). {date_range}"
        ),
        evidence={
            **summary,
            "top_product": str(top["product"]),
            "top_product_revenue": float(top["revenue"]),
        },
        chart=_evidence_chart(
            sales.head(5)[["product", "revenue"]],
            kind="bar",
            x="product",
            y=("revenue",),
            title="Top products by revenue",
        ),
    )


def _answer_financial(
    transactions: pd.DataFrame,
    question: str,
    products: pd.DataFrame,
    freq: str,
    date_range: str,
) -> ChatResponse:
    if _products_in_question(question, transactions, products) or _product_scope_requested(question):
        return _fallback(
            "unsupported",
            "This version reports business-wide profit and margins; product-level "
            "profitability isn't available yet.",
            "financial",
        )
    result = analytics.financial_performance(transactions, freq)
    if result.empty:
        return _fallback(
            "insufficient_data",
            "There isn't enough dated financial data to answer that. Add sales or "
            "expense records and try again.",
            "financial",
        )
    latest = result.iloc[-1]
    latest_label = _period_label(result.index[-1], freq)
    if len(result) >= 2:
        previous = result.iloc[-2]
        previous_label = _period_label(result.index[-2], freq)
        profit_change = float(latest["net_profit"] - previous["net_profit"])
        revenue_change = float(latest["revenue"] - previous["revenue"])
        expense_change = float(latest["total_expenses"] - previous["total_expenses"])
        answer = (
            f"Net profit was {latest['net_profit']:,.2f} in {latest_label}, "
            f"{profit_change:+,.2f} versus {previous_label}. Revenue changed "
            f"{revenue_change:+,.2f} and total expenses changed "
            f"{expense_change:+,.2f} between those periods. "
        )
    else:
        previous_label = None
        profit_change = None
        revenue_change = None
        expense_change = None
        answer = (
            f"Net profit was {latest['net_profit']:,.2f} in {latest_label}. "
            "There is only one available period, so I can't compare a trend yet. "
        )
    if "margin" in question:
        gross_margin = latest["gross_margin_pct"]
        net_margin = latest["net_margin_pct"]
        gross_margin_text = (
            f"{gross_margin:.1f}%" if pd.notna(gross_margin) else "unavailable"
        )
        net_margin_text = (
            f"{net_margin:.1f}%" if pd.notna(net_margin) else "unavailable"
        )
        answer += (
            f"Gross margin was {gross_margin_text} and net margin was "
            f"{net_margin_text} "
            f"in {latest_label}. "
        )
    if bool(latest["expense_growth_warning"]):
        answer += (
            f"Operating expenses grew {latest['operating_expense_growth_pct']:.1f}%, "
            f"compared with revenue growth of {latest['revenue_growth_pct']:.1f}%. "
        )
    answer += date_range
    chart_frame = result.tail(2).copy()
    chart_frame.index = [
        _period_label(period, freq) for period in chart_frame.index
    ]
    chart_frame.index.name = "period"
    chart_frame = chart_frame.reset_index()[
        ["period", "revenue", "total_expenses", "net_profit"]
    ]
    return ChatResponse(
        status="answered",
        category="financial",
        answer=answer,
        evidence={
            "period": latest_label,
            "previous_period": previous_label,
            "revenue": float(latest["revenue"]),
            "total_expenses": float(latest["total_expenses"]),
            "net_profit": float(latest["net_profit"]),
            "profit_change": profit_change,
            "revenue_change": revenue_change,
            "expense_change": expense_change,
            "expense_growth_warning": bool(latest["expense_growth_warning"]),
        },
        suggestions=(
            "Review the underlying expense categories and source records before "
            "making financial decisions.",
        ),
        chart=_evidence_chart(
            chart_frame,
            kind="bar",
            x="period",
            y=("revenue", "total_expenses", "net_profit"),
            title="Revenue, expenses, and net profit",
        ),
    )


def _answer_expenses(
    transactions: pd.DataFrame,
    freq: str,
    date_range: str,
) -> ChatResponse:
    breakdown = analytics.expense_category_breakdown(transactions, freq)
    if breakdown.empty:
        return _fallback(
            "insufficient_data",
            "I couldn't find recorded expenses to break down by category. "
            "Add expense records with categories and try again.",
            "expenses",
        )

    financial = analytics.financial_performance(transactions, freq)
    latest_period = financial.index[-1]
    previous_period = financial.index[-2] if len(financial) >= 2 else None
    latest = breakdown[breakdown["period"] == latest_period].copy()
    previous = (
        breakdown[breakdown["period"] == previous_period]
        if previous_period is not None
        else breakdown.iloc[0:0]
    )
    latest_total = float(latest["amount"].sum())
    previous_total = float(previous["amount"].sum())
    latest["share_pct"] = (
        latest["amount"] / latest_total * 100 if latest_total else 0.0
    )
    previous_by_category = previous.set_index("category")["amount"]
    latest["previous_amount"] = (
        latest["category"].map(previous_by_category).fillna(0.0)
    )
    latest["change_amount"] = latest["amount"] - latest["previous_amount"]
    current_by_category = latest.set_index("category")["amount"]
    comparison = pd.DataFrame([
        {
            "category": category,
            "expense_type": (
                latest.set_index("category")["expense_type"].get(category)
                or previous.set_index("category")["expense_type"].get(category)
            ),
            "amount": float(current_by_category.get(category, 0.0)),
            "previous_amount": float(previous_by_category.get(category, 0.0)),
        }
        for category in sorted(set(latest["category"]) | set(previous["category"]))
    ])
    if not comparison.empty:
        comparison["share_pct"] = (
            comparison["amount"] / latest_total * 100 if latest_total else 0.0
        )
        comparison["change_amount"] = (
            comparison["amount"] - comparison["previous_amount"]
        )
        comparison = comparison.sort_values(
            ["amount", "previous_amount"], ascending=False
        ).head(10)

    if latest.empty:
        answer = (
            f"No expenses were recorded in the latest {freq} period, "
            f"{_period_label(latest_period, freq)}. "
        )
    else:
        top_categories = latest.nlargest(10, "amount")
        categories = ", ".join(
            (
                f"{row.category}: {row.amount:,.2f} ({row.share_pct:.1f}%; "
                f"change {row.change_amount:+,.2f})"
                if row.previous_amount > 0
                else f"{row.category}: {row.amount:,.2f} ({row.share_pct:.1f}%; new)"
            )
            for row in top_categories.itertuples(index=False)
        )
        answer = (
            f"Expenses in {_period_label(latest_period, freq)} totaled "
            f"{latest_total:,.2f}. Top categories: {categories}. "
        )

    if previous_period is not None:
        answer += (
            f"Previous-period expenses "
            f"({_period_label(previous_period, freq)}) totaled "
            f"{previous_total:,.2f}. "
        )
    answer += date_range

    chart_frame = latest.nlargest(10, "amount")[["category", "amount"]].sort_values(
        "amount", ascending=True
    )
    return ChatResponse(
        status="answered",
        category="expenses",
        answer=answer,
        evidence={
            "period": _period_label(latest_period, freq),
            "total_expenses": latest_total,
            "previous_period": (
                _period_label(previous_period, freq)
                if previous_period is not None
                else None
            ),
            "previous_total_expenses": previous_total,
            "categories": comparison.to_dict(orient="records"),
        },
        suggestions=(
            "Compare category changes with the previous period and your budget.",
            "Review uncategorized expenses and confirm cost-of-goods entries are labeled consistently.",
        ),
        chart=(
            _evidence_chart(
                chart_frame,
                kind="bar",
                x="amount",
                y=("category",),
                title=f"Expense categories — {_period_label(latest_period, freq)}",
            )
            if not chart_frame.empty
            else None
        ),
    )


def _answer_forecast(
    transactions: pd.DataFrame,
    question: str,
    products: pd.DataFrame,
    freq: str,
    date_range: str,
) -> ChatResponse:
    if re.search(r"\b(revenue|profit|profitability|expense|expenses|margin)\b", question):
        return _fallback(
            "unsupported",
            "I can forecast unit demand, but this version does not forecast revenue "
            "or profit. I won't infer financial outcomes from a unit forecast.",
            "forecast",
        )
    mentioned = _products_in_question(question, transactions, products)
    if not mentioned and _product_scope_requested(question):
        return _fallback(
            "clarification_needed",
            "I couldn't match that product name to the available records. Check the "
            "product name or ask for an overall demand forecast.",
            "forecast",
        )
    product = mentioned[0] if mentioned else None
    result = forecast.forecast_demand(
        transactions,
        periods=DEFAULT_FORECAST_PERIODS,
        freq=freq,
        product=product,
    )
    if result.status == "no_sales":
        return _fallback(
            "insufficient_data",
            "There are no matching sales records to forecast. Add sales history "
            "and try again.",
            "forecast",
        )
    if result.status == "insufficient_history":
        return _fallback(
            "insufficient_data",
            f"I don't have enough history to make a forecast: "
            f"{result.history_periods} {freq} periods are available, while "
            f"{config.RULES['min_history_points']} are required.",
            "forecast",
        )
    total = float(result.forecast["forecast_demand"].sum())
    product_label = f" for {product}" if product else ""
    values = ", ".join(
        f"{_period_label(period, freq)}: {value:,.1f}"
        for period, value in result.forecast["forecast_demand"].items()
    )
    return ChatResponse(
        status="answered",
        category="forecast",
        answer=(
            f"Forecast demand{product_label} for the next {DEFAULT_FORECAST_PERIODS} "
            f"{freq} periods totals {total:,.1f} units ({values}). "
            f"Model: {result.selected_model}; validation MAE: "
            f"{result.validation_mae:.2f}. This is an estimate, not a commitment. "
            f"{date_range}"
        ),
        evidence={
            "product": product,
            "periods": [
                {
                    "period": _period_label(period, freq),
                    "forecast_demand": float(value),
                }
                for period, value in result.forecast["forecast_demand"].items()
            ],
            "total_forecast_units": total,
            "model": result.selected_model,
            "validation_mae": result.validation_mae,
            "history_periods": result.history_periods,
        },
        chart=_evidence_chart(
            pd.DataFrame({
                "period": [
                    _period_label(period, freq)
                    for period in result.forecast.index
                ],
                "forecast_demand": result.forecast["forecast_demand"].to_numpy(),
            }),
            kind="line",
            x="period",
            y=("forecast_demand",),
            title=f"Forecast demand{product_label}",
        ),
    )


def _answer_inventory(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    question: str,
    date_range: str,
) -> ChatResponse:
    if products.empty:
        return _fallback(
            "insufficient_data",
            "I don't have product stock and lead-time records to assess inventory. "
            "Add products with stock_on_hand and lead_time_days, then try again.",
            "inventory",
        )
    if len(products) > MAX_INVENTORY_PRODUCTS:
        return _fallback(
            "limit_exceeded",
            f"Inventory Q&A is limited to {MAX_INVENTORY_PRODUCTS:,} products in "
            "this version. Narrow the product list and try again.",
            "inventory",
        )
    result = inventory.inventory_recommendations(transactions, products)
    mentioned = _products_in_question(question, transactions, products)
    if not mentioned and _product_scope_requested(question):
        return _fallback(
            "clarification_needed",
            "I couldn't match that product name to the available inventory records. "
            "Check the product name or ask about all products.",
            "inventory",
        )
    if mentioned:
        result = result[result["product"].str.casefold() == mentioned[0].casefold()]
        if result.empty:
            return _fallback(
                "insufficient_data",
                f"I couldn't find inventory data for {mentioned[0]}.",
                "inventory",
            )
    rows = result[result["reorder_status"].isin(["reorder", "insufficient_history", "no_sales"])]
    if rows.empty:
        rows = result.head(3)
    messages = []
    evidence = []
    for row in rows.itertuples(index=False):
        evidence.append({
            "product": row.product,
            "status": row.reorder_status,
            "stock_on_hand": float(row.stock_on_hand),
            "reorder_point": float(row.reorder_point),
            "recommended_order_qty": int(row.recommended_order_qty),
        })
        if row.reorder_status == "reorder":
            messages.append(
                f"{row.product}: stock {row.stock_on_hand:g}, estimated reorder "
                f"point {row.reorder_point:.1f}; review the suggested "
                f"{row.recommended_order_qty} units"
            )
        elif row.reorder_status in {"no_sales", "insufficient_history"}:
            messages.append(
                f"{row.product}: {row.reorder_status.replace('_', ' ')}; "
                "no order quantity is suggested"
            )
        else:
            messages.append(
                f"{row.product}: stock {row.stock_on_hand:g} is above the "
                f"estimated reorder point {row.reorder_point:.1f}"
            )
    chart_frame = pd.DataFrame(evidence)
    return ChatResponse(
        status="answered",
        category="inventory",
        answer=(
            "; ".join(messages)
            + ". Review these estimates against actual stock and suppliers; "
            "no order is placed. "
            + date_range
        ),
        evidence={"inventory": evidence},
        chart=_evidence_chart(
            chart_frame,
            kind="bar",
            x="product",
            y=("stock_on_hand", "reorder_point", "recommended_order_qty"),
            title="Stock compared with estimated replenishment",
        ),
    )


def _answer_anomalies(
    transactions: pd.DataFrame,
    question: str,
    products: pd.DataFrame,
    date_range: str,
) -> ChatResponse:
    if len(transactions) > MAX_ANOMALY_TRANSACTIONS:
        return _fallback(
            "limit_exceeded",
            f"Anomaly review is limited to {MAX_ANOMALY_TRANSACTIONS:,} transaction "
            "rows in this version to keep responses responsive. Narrow the data "
            "to a smaller date range and try again.",
            "anomaly",
        )
    transaction_alerts = anomaly.detect_transaction_anomalies(transactions)
    demand_alerts = anomaly.detect_daily_demand_anomalies(transactions)
    mentioned = _products_in_question(question, transactions, products)
    if mentioned:
        product_name = mentioned[0].casefold()
        transaction_alerts = transaction_alerts[
            transaction_alerts["product"].str.casefold() == product_name
        ]
        demand_alerts = demand_alerts[
            demand_alerts["product"].str.casefold() == product_name
        ]
    alert_counts = pd.DataFrame([
        {"alert_type": "Unusual transactions", "count": len(transaction_alerts)},
        {"alert_type": "Unusual daily demand", "count": len(demand_alerts)},
    ])
    if transaction_alerts.empty and demand_alerts.empty:
        return ChatResponse(
            status="answered",
            category="anomaly",
            answer=(
                "No unusual records met the configured thresholds and minimum "
                "history requirements. That does not prove all activity is normal. "
                f"{date_range}"
            ),
            evidence={"transaction_alerts": 0, "demand_alerts": 0},
            suggestions=(
                "Review source records and business context; anomaly flags are "
                "investigation aids, not proof of wrongdoing.",
            ),
            chart=_evidence_chart(
                alert_counts,
                kind="bar",
                x="alert_type",
                y=("count",),
                title="Alerts found for review",
            ),
        )
    transaction_rows = transaction_alerts.head(5)
    demand_rows = demand_alerts.head(5)
    summary = []
    if not transaction_rows.empty:
        summary.append(
            f"{len(transaction_alerts)} unusual transaction(s), including "
            + ", ".join(
                f"{row.product or row.category} on "
                f"{pd.Timestamp(row.date).date()} ({row.anomaly_metric})"
                for row in transaction_rows.itertuples(index=False)
            )
        )
    if not demand_rows.empty:
        summary.append(
            f"{len(demand_alerts)} unusual daily-demand period(s), including "
            + ", ".join(
                f"{row.product} on {pd.Timestamp(row.date).date()}"
                for row in demand_rows.itertuples(index=False)
            )
        )
    return ChatResponse(
        status="answered",
        category="anomaly",
        answer=(
            "; ".join(summary)
            + ". These are statistical flags for review, not findings of wrongdoing. "
            + date_range
        ),
        evidence={
            "transaction_alert_count": len(transaction_alerts),
            "demand_alert_count": len(demand_alerts),
            "transaction_alerts": transaction_rows.to_dict(orient="records"),
            "demand_alerts": demand_rows.to_dict(orient="records"),
        },
        chart=_evidence_chart(
            alert_counts,
            kind="bar",
            x="alert_type",
            y=("count",),
            title="Alerts found for review",
        ),
    )


def _answer_recommendations(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    freq: str,
    date_range: str,
) -> ChatResponse:
    if transactions.empty:
        return _fallback(
            "insufficient_data",
            "I need transaction history before I can suggest data-based areas to "
            "review. Add transactions and try again.",
            "recommendations",
        )
    if len(transactions) > MAX_ANOMALY_TRANSACTIONS:
        return _fallback(
            "limit_exceeded",
            f"Combined recommendations are limited to {MAX_ANOMALY_TRANSACTIONS:,} "
            "transaction rows in this version to keep responses responsive. "
            "Narrow the data to a smaller date range and try again.",
            "recommendations",
        )
    business_report = business_assistant.generate_business_report(
        transactions,
        products=products,
        freq=freq,
        forecast_periods=DEFAULT_FORECAST_PERIODS,
    )
    if not business_report.insights:
        return _fallback(
            "insufficient_data",
            "The available data did not support a specific recommendation. Add "
            "more history or ask about a particular sales, financial, or inventory measure.",
            "recommendations",
        )
    selected = business_report.insights[:3]
    answer = "Areas to review based on the available data: " + "; ".join(
        f"{item.title}: {item.recommended_action}" for item in selected
    )
    return ChatResponse(
        status="answered",
        category="recommendations",
        answer=f"{answer} {date_range}",
        evidence={
            "insights": [
                {
                    "priority": item.priority,
                    "category": item.category,
                    "title": item.title,
                    "evidence": item.evidence,
                }
                for item in selected
            ]
        },
        suggestions=(
            "Recommendations are advisory and depend on the quality and coverage "
            "of the supplied data.",
        ),
    )


def _answer_question(
    question: str,
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    freq: str,
) -> ChatResponse:
    if not isinstance(question, str):
        return _fallback(
            "clarification_needed",
            "Please enter your question as text about one of the supported business topics.",
        )
    if not question.strip():
        return _fallback(
            "clarification_needed",
            "Ask a question about sales, expenses, profit, forecasts, inventory, "
            "or unusual activity.",
        )
    if len(transactions) > MAX_CHAT_TRANSACTIONS:
        return _fallback(
            "limit_exceeded",
            f"Chat analysis is limited to {MAX_CHAT_TRANSACTIONS:,} transaction rows "
            "in this version. Narrow the data and try again.",
        )
    if len(question) > MAX_QUESTION_LENGTH:
        return _fallback(
            "clarification_needed",
            f"Please shorten your question to {MAX_QUESTION_LENGTH} characters or fewer.",
        )
    if freq not in config.FREQUENCIES:
        return _fallback(
            "invalid_parameters",
            f"Choose a supported analysis period: {', '.join(config.FREQUENCIES)}.",
        )

    normalized = re.sub(r"\s+", " ", question.casefold()).strip()
    guide_answer = help_content.answer_for_question(normalized)
    if guide_answer is not None:
        return ChatResponse(
            status="answered",
            answer=guide_answer,
            category="help",
            suggestions=(
                "How do I choose an analysis period?",
                "How do I safely delete test data?",
                "What does the forecasting page show?",
            ),
        )

    if transactions.empty:
        return _fallback(
            "insufficient_data",
            "I don't have transaction data to answer that yet. Add or upload "
            "business records, then ask about sales, expenses, forecasts, inventory, "
            "or unusual activity.",
        )

    if re.search(
        r"\b(?:last|past|previous)\s+(?:\d+\s+)?"
        r"(?:day|week|month|quarter|year)s?\b|\bbetween\b|\bin\s+20\d{2}\b",
        normalized,
    ):
        return _fallback(
            "unsupported",
            "This version analyzes the full available transaction history; custom "
            "date-range filtering in chat isn't available yet. Try asking about "
            "overall performance or the latest completed periods.",
            "date_range",
        )
    if normalized in {"?", "help", "hi", "hello", "hey"}:
        return _fallback(
            "clarification_needed",
            f"Hi! I can answer questions about {SUPPORTED_TOPICS}. Ask a specific "
            "business question to get started.",
        )

    intent = _detect_intent(normalized)
    if intent is None:
        return _fallback(
            "out_of_scope",
            f"I can answer questions about {SUPPORTED_TOPICS}, but I couldn't "
            "match this question to one of those topics. Try naming a business "
            "measure and, if relevant, a product.",
        )

    date_range = _date_range(transactions)
    if intent == "sales":
        return _answer_sales(transactions, normalized, products, freq, date_range)
    if intent == "financial":
        return _answer_financial(transactions, normalized, products, freq, date_range)
    if intent == "expense_analysis":
        return _answer_expenses(transactions, freq, date_range)
    if intent == "forecast":
        return _answer_forecast(transactions, normalized, products, freq, date_range)
    if intent == "inventory":
        return _answer_inventory(transactions, products, normalized, date_range)
    if intent == "anomaly":
        return _answer_anomalies(transactions, normalized, products, date_range)
    if intent == "recommendations":
        return _answer_recommendations(transactions, products, freq, date_range)
    return _fallback(
        "out_of_scope",
        f"I can answer questions about {SUPPORTED_TOPICS}. Please try one of those topics.",
    )


def answer_question(
    question: str,
    transactions: pd.DataFrame,
    products: pd.DataFrame | None = None,
    freq: str = "monthly",
) -> ChatResponse:
    """Answer one business question from provided data, or return a safe fallback.

    Expected out-of-scope, ambiguous, and insufficient-data cases are returned
    explicitly. Unexpected processing errors are logged with a traceback and
    converted to a brief retry message rather than crashing the chat session.
    """
    product_data = (
        products if products is not None
        else pd.DataFrame(columns=data.PRODUCT_COLUMNS)
    )
    try:
        business_assistant._validate_transactions(transactions)
    except (TypeError, ValueError, KeyError) as exc:
        _LOGGER.warning("Invalid data supplied to BizTrend chatbot: %s", exc)
        return _fallback(
            "invalid_data",
            "I can't analyze these records because they don't match the expected "
            "validated transaction format. Check the imported data and try again.",
            "data",
        )
    try:
        return _answer_question(question, transactions, product_data, freq)
    except Exception:
        _LOGGER.exception("Unexpected error while answering a BizTrend question.")
        return ChatResponse(
            status="error",
            category="error",
            answer=UNEXPECTED_ERROR_MESSAGE,
        )
