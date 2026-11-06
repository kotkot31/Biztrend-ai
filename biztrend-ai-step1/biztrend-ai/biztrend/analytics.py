"""Sales and financial analysis built on the canonical transaction schema."""
from __future__ import annotations

import pandas as pd

from . import config, data


def _validate_frequency(freq: str) -> str:
    if freq not in config.FREQUENCIES:
        raise ValueError(f"freq must be one of {list(config.FREQUENCIES)}")
    return config.FREQUENCIES[freq]


def period_label(period: object, freq: str) -> str:
    """Format an aggregation period for display in charts and summaries."""
    value = pd.Timestamp(period)
    if freq == "monthly":
        return value.strftime("%B %Y")
    if freq == "quarterly":
        return f"Q{value.quarter} {value.year}"
    if freq == "yearly":
        return value.strftime("%Y")
    return value.strftime("%Y-%m-%d")


def sales_summary(df: pd.DataFrame) -> dict[str, float | int]:
    """Return all-time sales KPIs; transaction count means sales rows, not orders."""
    sales = df[df["record_type"] == "sale"]
    revenue = float(sales["amount"].sum())
    transaction_count = len(sales)
    return {
        "revenue": revenue,
        "units_sold": float(sales["quantity"].sum()),
        "transaction_count": transaction_count,
        "products_sold": int(sales.loc[sales["product"] != "", "product"].nunique()),
        "average_transaction_value": (
            revenue / transaction_count if transaction_count else 0.0
        ),
    }


def sales_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """Rank named products by revenue, with units and transaction-line counts."""
    sales = df[(df["record_type"] == "sale") & df["product"].ne("")]
    columns = ["product", "revenue", "units_sold", "transaction_count"]
    if sales.empty:
        return pd.DataFrame(columns=columns)
    return (
        sales.groupby("product", as_index=False)
        .agg(
            revenue=("amount", "sum"),
            units_sold=("quantity", "sum"),
            transaction_count=("amount", "size"),
        )
        .sort_values(["revenue", "product"], ascending=[False, True])
        .reset_index(drop=True)
    )[columns]


def sales_trends(df: pd.DataFrame, freq: str = "monthly") -> pd.DataFrame:
    """Return revenue, units, and transaction-line counts for each period."""
    alias = _validate_frequency(freq)
    columns = ["revenue", "units_sold", "transaction_count"]
    periods = data.aggregate(df, freq)
    if periods.empty:
        return pd.DataFrame(columns=columns).rename_axis("period")

    sales = df[df["record_type"] == "sale"]
    if sales.empty:
        return pd.DataFrame(0.0, index=periods.index, columns=columns).rename_axis("period")

    grouped = sales.groupby(pd.Grouper(key="date", freq=alias))
    result = pd.DataFrame(index=periods.index)
    result["revenue"] = grouped["amount"].sum().reindex(periods.index, fill_value=0.0)
    result["units_sold"] = grouped["quantity"].sum().reindex(periods.index, fill_value=0.0)
    result["transaction_count"] = grouped.size().reindex(periods.index, fill_value=0)
    return result.rename_axis("period")


def financial_performance(df: pd.DataFrame, freq: str = "monthly") -> pd.DataFrame:
    """Return period P&L, margins, and operating-expense growth warnings.

    Cost of goods is identified by the configured expense category; all other
    expense categories are treated as operating expenses.
    """
    _validate_frequency(freq)
    columns = [
        "revenue", "cost_of_goods", "operating_expenses", "total_expenses",
        "gross_profit", "operating_profit", "net_profit", "gross_margin_pct",
        "net_margin_pct", "revenue_growth_pct", "operating_expense_growth_pct",
        "expense_growth_warning",
    ]
    periods = data.aggregate(df, freq)
    if periods.empty:
        return pd.DataFrame(columns=columns).rename_axis("period")

    revenue_by_period = data.aggregate(df, freq, "sale")
    revenue = (
        revenue_by_period["amount"]
        if not revenue_by_period.empty
        else pd.Series(dtype=float)
    )
    expenses = df[df["record_type"] == "expense"]
    is_cogs = expenses["category"].str.strip().str.casefold() == config.COGS_CATEGORY.casefold()
    cogs = data.aggregate(expenses[is_cogs], freq, "expense")
    operating = data.aggregate(expenses[~is_cogs], freq, "expense")

    result = pd.DataFrame(index=periods.index)
    result["revenue"] = revenue.reindex(result.index, fill_value=0.0)
    result["cost_of_goods"] = (
        cogs["amount"].reindex(result.index, fill_value=0.0) if not cogs.empty else 0.0
    )
    result["operating_expenses"] = (
        operating["amount"].reindex(result.index, fill_value=0.0)
        if not operating.empty else 0.0
    )
    result["total_expenses"] = result["cost_of_goods"] + result["operating_expenses"]
    result["gross_profit"] = result["revenue"] - result["cost_of_goods"]
    result["operating_profit"] = result["gross_profit"] - result["operating_expenses"]
    result["net_profit"] = result["revenue"] - result["total_expenses"]
    result["gross_margin_pct"] = (
        result["gross_profit"].div(result["revenue"].where(result["revenue"] != 0)) * 100
    )
    result["net_margin_pct"] = (
        result["net_profit"].div(result["revenue"].where(result["revenue"] != 0)) * 100
    )
    result["revenue_growth_pct"] = (
        result["revenue"].pct_change(fill_method=None) * 100
    ).replace([float("inf"), float("-inf")], float("nan"))
    result["operating_expense_growth_pct"] = (
        result["operating_expenses"].pct_change(fill_method=None) * 100
    ).replace([float("inf"), float("-inf")], float("nan"))
    warning_gap = config.RULES["expense_growth_warning_gap"] * 100
    result["expense_growth_warning"] = (
        result["operating_expense_growth_pct"] - result["revenue_growth_pct"]
    ).gt(warning_gap).fillna(False)
    return result[columns].rename_axis("period")
