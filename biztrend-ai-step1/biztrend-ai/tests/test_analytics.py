import pandas as pd
import pytest

from biztrend import analytics, data


@pytest.fixture
def transactions():
    raw = pd.DataFrame([
        ("2026-01-05", "sale", "A", "Hardware", 2, 300, 600),
        ("2026-01-20", "sale", "B", "Hardware", 4, 100, 400),
        ("2026-01-25", "sale", "A", "Hardware", 1, 300, 300),
        ("2026-01-31", "expense", "", "Cost of Goods", 1, 400, 400),
        ("2026-01-31", "expense", "", "Rent", 1, 100, 100),
        ("2026-02-05", "sale", "A", "Hardware", 3, 300, 900),
        ("2026-02-20", "sale", "B", "Hardware", 6, 100, 600),
        ("2026-02-28", "expense", "", "Cost of Goods", 1, 600, 600),
        ("2026-02-28", "expense", "", "Rent", 1, 200, 200),
        ("2026-03-10", "expense", "", "Rent", 1, 200, 200),
    ], columns=[
        "date", "record_type", "product", "category", "quantity", "unit_price", "amount",
    ])
    cleaned, _ = data.clean_transactions(raw)
    return cleaned


def test_sales_summary_and_product_ranking(transactions):
    summary = analytics.sales_summary(transactions)
    assert summary == {
        "revenue": 2800.0,
        "units_sold": 16.0,
        "transaction_count": 5,
        "products_sold": 2,
        "average_transaction_value": 560.0,
    }

    products = analytics.sales_by_product(transactions)
    assert products["product"].tolist() == ["A", "B"]
    assert products["revenue"].tolist() == [1800, 1000]
    assert products["units_sold"].tolist() == [6, 10]
    assert products["transaction_count"].tolist() == [3, 2]


def test_sales_trends_support_periods_and_fill_missing_sales(transactions):
    monthly = analytics.sales_trends(transactions, "monthly")
    assert monthly["revenue"].tolist() == [1300, 1500, 0]
    assert monthly["units_sold"].tolist() == [7, 9, 0]
    assert monthly["transaction_count"].tolist() == [3, 2, 0]

    weekly = analytics.sales_trends(transactions, "weekly")
    assert weekly["revenue"].sum() == pytest.approx(2800)
    with pytest.raises(ValueError):
        analytics.sales_trends(transactions, "hourly")


def test_financial_performance_calculates_profit_margins_and_warning(transactions):
    result = analytics.financial_performance(transactions, "monthly")
    january, february, march = result.iloc[0], result.iloc[1], result.iloc[2]

    assert january["gross_profit"] == 900
    assert january["operating_profit"] == 800
    assert january["net_profit"] == 800
    assert january["gross_margin_pct"] == pytest.approx(900 / 1300 * 100)
    assert january["net_margin_pct"] == pytest.approx(800 / 1300 * 100)
    assert february["revenue_growth_pct"] == pytest.approx((1500 / 1300 - 1) * 100)
    assert february["operating_expense_growth_pct"] == pytest.approx(100)
    assert february["expense_growth_warning"]
    assert pd.isna(march["gross_margin_pct"])
    assert march["revenue_growth_pct"] == -100
    assert march["operating_expense_growth_pct"] == 0
    assert march["expense_growth_warning"]


def test_financial_performance_supports_quarterly_and_yearly_periods():
    rows = []
    for date, record_type, amount in [
        ("2024-01-15", "sale", 100),
        ("2024-03-20", "expense", 40),
        ("2024-04-15", "sale", 200),
        ("2025-02-15", "expense", 30),
    ]:
        rows.append({
            "date": date,
            "record_type": record_type,
            "product": "Keyboard" if record_type == "sale" else "",
            "category": "Peripherals" if record_type == "sale" else "Rent",
            "quantity": 1,
            "unit_price": amount,
            "amount": amount,
        })
    cleaned, _ = data.clean_transactions(pd.DataFrame(rows))

    quarterly = analytics.financial_performance(cleaned, "quarterly")
    yearly = analytics.financial_performance(cleaned, "yearly")

    assert quarterly["revenue"].tolist() == [100, 200, 0, 0, 0]
    assert quarterly["total_expenses"].tolist() == [40, 0, 0, 0, 30]
    assert yearly["net_profit"].tolist() == [260, -30]
    assert analytics.period_label(pd.Timestamp("2024-04-01"), "quarterly") == "Q2 2024"
    assert analytics.period_label(pd.Timestamp("2024-01-01"), "yearly") == "2024"


def test_empty_sales_and_expense_only_data_are_handled():
    empty = pd.DataFrame(columns=data.TRANSACTION_COLUMNS)
    assert analytics.sales_summary(empty)["revenue"] == 0
    assert analytics.sales_by_product(empty).empty
    assert analytics.sales_trends(empty).empty
    assert analytics.financial_performance(empty).empty

    expenses, _ = data.clean_transactions(pd.DataFrame({
        "date": ["2026-01-01"], "record_type": ["expense"],
        "category": ["Rent"], "amount": [500],
    }))
    trends = analytics.sales_trends(expenses)
    assert trends["revenue"].tolist() == [0]
    performance = analytics.financial_performance(expenses)
    assert performance.iloc[0]["operating_expenses"] == 500
    assert performance.iloc[0]["net_profit"] == -500
