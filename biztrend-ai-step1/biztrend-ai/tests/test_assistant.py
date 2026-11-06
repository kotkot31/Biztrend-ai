import pandas as pd
import pytest

from biztrend import assistant, data


def _business_data():
    rows = []
    for month in range(1, 10):
        start = pd.Timestamp(2026, month, 1)
        for day in range(1, 11):
            units = 20 if month == 9 and day == 10 else 1
            rows.append((
                start + pd.Timedelta(days=day - 1),
                "sale",
                "A",
                "Hardware",
                units,
                100,
                units * 100,
                "",
            ))
        rows.append((
            start + pd.offsets.MonthEnd(0),
            "expense",
            "",
            "Rent",
            1,
            None,
            5000 if month == 9 else 100 + month * 20,
            "",
        ))
    transactions, _ = data.clean_transactions(pd.DataFrame(rows, columns=data.TRANSACTION_COLUMNS))
    products = pd.DataFrame([{
        "product": "A",
        "unit_cost": 50,
        "stock_on_hand": 0,
        "lead_time_days": 3,
    }])
    return transactions, products


def test_report_generates_explainable_insights_from_analysis_outputs():
    transactions, products = _business_data()
    report = assistant.generate_business_report(transactions, products)

    categories = {insight.category for insight in report.insights}
    assert {"sales", "financial", "inventory", "anomaly"} <= categories
    assert report.sales_summary["transaction_count"] == 90
    assert report.demand_forecast.status == "ok"
    assert not report.financial_performance.empty
    assert not report.inventory_recommendations.empty
    assert not report.transaction_anomalies.empty
    assert not report.demand_anomalies.empty

    for insight in report.insights:
        assert insight.priority in {"urgent", "review", "watch", "info"}
        assert insight.explanation
        assert insight.recommended_action
        assert insight.evidence
        assert "outpaces revenue" not in insight.title or insight.category == "financial"

    priorities = [insight.priority for insight in report.insights]
    assert priorities == sorted(
        priorities,
        key=lambda priority: {"urgent": 0, "review": 1, "watch": 2, "info": 3}[priority],
    )


def test_report_handles_empty_data_and_no_inventory():
    empty = pd.DataFrame(columns=data.TRANSACTION_COLUMNS)
    report = assistant.generate_business_report(empty)

    assert report.sales_summary["revenue"] == 0
    assert report.financial_performance.empty
    assert report.demand_forecast.status == "no_sales"
    assert report.inventory_recommendations.empty
    assert report.transaction_anomalies.empty
    assert report.demand_anomalies.empty
    assert report.insights == []


def test_report_preserves_insufficient_forecast_history():
    raw = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=4, freq="MS"),
        "record_type": "sale",
        "product": "A",
        "quantity": [1, 2, 3, 4],
        "unit_price": 10,
        "amount": [10, 20, 30, 40],
    })
    transactions, _ = data.clean_transactions(raw)
    report = assistant.generate_business_report(transactions)

    assert report.demand_forecast.status == "insufficient_history"
    assert not any(insight.category == "demand" for insight in report.insights)


def test_report_rejects_invalid_input_parameters():
    empty = pd.DataFrame(columns=data.TRANSACTION_COLUMNS)
    with pytest.raises(ValueError, match="freq"):
        assistant.generate_business_report(empty, freq="hourly")
    with pytest.raises(ValueError, match="forecast_periods"):
        assistant.generate_business_report(empty, forecast_periods=0)
    with pytest.raises(ValueError, match="canonical schema"):
        assistant.generate_business_report(pd.DataFrame({"product": ["A"]}))
