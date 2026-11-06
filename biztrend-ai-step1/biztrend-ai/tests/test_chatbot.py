import logging
import math

import pandas as pd
import pytest

from biztrend import chatbot, data, help_content, inventory


def _monthly_sales(values, product="Keyboard"):
    rows = []
    for month, units in enumerate(values, start=1):
        date = pd.Timestamp(2025, month, 1)
        rows.append({
            "date": date,
            "record_type": "sale",
            "product": product,
            "category": "Peripherals",
            "quantity": units,
            "unit_price": 10,
            "amount": units * 10,
            "notes": "",
        })
    cleaned, _ = data.clean_transactions(pd.DataFrame(rows))
    return cleaned


def _daily_sales(values, product="Keyboard"):
    raw = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=len(values), freq="D"),
        "record_type": "sale",
        "product": product,
        "category": "Peripherals",
        "quantity": values,
        "unit_price": 10,
        "amount": [value * 10 for value in values],
        "notes": "",
    })
    cleaned, _ = data.clean_transactions(raw)
    return cleaned


def test_answers_top_product_with_historical_evidence():
    transactions = _monthly_sales([2, 4, 3], "Keyboard")
    other, _ = data.clean_transactions(pd.DataFrame([{
        "date": "2025-02-01",
        "record_type": "sale",
        "product": "Mouse",
        "quantity": 10,
        "unit_price": 5,
        "amount": 50,
    }]))
    transactions = pd.concat([transactions, other], ignore_index=True)

    response = chatbot.answer_question("Which products generated the most revenue?", transactions)

    assert response.status == "answered"
    assert response.category == "sales"
    assert "Keyboard" in response.answer
    assert "2025-01-01 to 2025-03-01" in response.answer
    assert response.evidence["top_products"][0]["product"] == "Keyboard"
    assert response.chart is not None
    assert response.chart.kind == "bar"
    assert response.chart.data[0]["product"] == "Keyboard"
    assert response.chart.y == ("revenue",)


def test_answers_specific_product_or_clarifies_unknown_product():
    transactions = _monthly_sales([2, 4, 3])
    response = chatbot.answer_question("How much revenue did Keyboard make?", transactions)
    assert response.status == "answered"
    assert response.evidence["product"] == "Keyboard"
    assert response.evidence["revenue"] == 90

    unknown = chatbot.answer_question("How much revenue did Unknown make?", transactions)
    assert unknown.status == "clarification_needed"
    assert "couldn't match" in unknown.answer


def test_answers_revenue_trend_for_a_specific_product():
    transactions = _monthly_sales([2, 4, 3])

    response = chatbot.answer_question("Is Keyboard revenue growing?", transactions)

    assert response.status == "answered"
    assert response.evidence["product"] == "Keyboard"
    assert response.evidence["previous_value"] == 40
    assert response.evidence["latest_value"] == 30
    assert response.evidence["change_pct"] == -25
    assert response.chart is not None
    assert response.chart.kind == "line"
    assert response.chart.data[-1]["revenue"] == 30


def test_answers_profit_change_from_last_two_periods():
    rows = []
    for date, record_type, amount, category in [
        ("2025-01-15", "sale", 100, "Hardware"),
        ("2025-01-31", "expense", 40, "Rent"),
        ("2025-02-15", "sale", 120, "Hardware"),
        ("2025-02-28", "expense", 80, "Rent"),
    ]:
        rows.append({
            "date": date,
            "record_type": record_type,
            "product": "Keyboard" if record_type == "sale" else "",
            "category": category,
            "quantity": 1,
            "unit_price": amount,
            "amount": amount,
            "notes": "",
        })
    transactions, _ = data.clean_transactions(pd.DataFrame(rows))

    response = chatbot.answer_question("Why did profit change?", transactions, freq="monthly")

    assert response.status == "answered"
    assert response.category == "financial"
    assert response.evidence["profit_change"] == -20
    assert response.evidence["revenue_change"] == 20
    assert response.evidence["expense_change"] == 40


def test_expense_category_question_returns_period_breakdown_and_chart():
    rows = [{
        "date": date,
        "record_type": "expense",
        "product": "",
        "category": category,
        "quantity": 1,
        "unit_price": amount,
        "amount": amount,
        "notes": "",
    } for date, category, amount in [
        ("2025-01-15", "Rent", 100),
        ("2025-01-15", "Cost of Goods", 50),
        ("2025-02-15", "Rent", 200),
        ("2025-02-15", "Cost of Goods", 50),
        ("2025-02-15", "Utilities", 50),
    ]]
    transactions, _ = data.clean_transactions(pd.DataFrame(rows))

    response = chatbot.answer_question("Review expense categories", transactions)

    assert response.status == "answered"
    assert response.category == "expenses"
    assert "February 2025" in response.answer
    assert "Rent: 200.00 (66.7%; change +100.00)" in response.answer
    assert response.evidence["total_expenses"] == 300
    rent = next(row for row in response.evidence["categories"] if row["category"] == "Rent")
    assert rent["expense_type"] == "Operating expense"
    assert rent["previous_amount"] == 100
    assert rent["change_amount"] == 100
    assert response.chart is not None
    assert response.chart.kind == "bar"


def test_expense_category_chatbot_handles_missing_and_zero_previous_expenses():
    rows = [{
        "date": date,
        "record_type": "expense",
        "product": "",
        "category": category,
        "quantity": 1,
        "unit_price": amount,
        "amount": amount,
        "notes": "",
    } for date, category, amount in [
        ("2025-01-15", "Rent", 100),
        ("2025-02-15", "Utilities", 50),
    ]]
    transactions, _ = data.clean_transactions(pd.DataFrame(rows))
    response = chatbot.answer_question("Show expenses by category", transactions)
    utilities = next(
        row for row in response.evidence["categories"]
        if row["category"] == "Utilities"
    )
    assert utilities["previous_amount"] == 0
    assert utilities["share_pct"] == 100
    assert "Utilities: 50.00 (100.0%; new)" in response.answer
    assert response.status == "answered"

    sales_only = _monthly_sales([2, 4])
    missing = chatbot.answer_question("Review expense categories", sales_only)
    assert missing.status == "insufficient_data"
    assert "couldn't find recorded expenses" in missing.answer


def test_answers_profit_margins_and_refuses_unsupported_date_filter():
    rows = [{
        "date": date,
        "record_type": record_type,
        "product": "Keyboard" if record_type == "sale" else "",
        "category": "Peripherals" if record_type == "sale" else "Rent",
        "quantity": 1,
        "unit_price": amount,
        "amount": amount,
        "notes": "",
    } for date, record_type, amount in [
        ("2025-01-15", "sale", 100),
        ("2025-01-31", "expense", 40),
        ("2025-02-15", "sale", 120),
        ("2025-02-28", "expense", 80),
    ]]
    transactions, _ = data.clean_transactions(pd.DataFrame(rows))
    margin = chatbot.answer_question("What is my profit margin?", transactions)

    assert margin.status == "answered"
    assert "Gross margin was" in margin.answer
    assert "net margin was" in margin.answer

    period = chatbot.answer_question("What were my sales last month?", transactions)
    assert period.status == "unsupported"
    assert "full available transaction history" in period.answer


def test_product_level_profitability_is_not_reported_as_business_wide():
    transactions = _monthly_sales([2, 4, 3])

    response = chatbot.answer_question("What is Keyboard's profit?", transactions)

    assert response.status == "unsupported"
    assert "product-level profitability isn't available" in response.answer


def test_forecast_answers_and_insufficient_history_fallback():
    enough = _monthly_sales(list(range(1, 10)))
    response = chatbot.answer_question("What is the demand forecast?", enough)
    assert response.status == "answered"
    assert response.category == "forecast"
    assert "estimate" in response.answer
    assert response.evidence["history_periods"] == 9
    assert response.chart is not None
    assert response.chart.data[0]["forecast_demand"] >= 0

    too_short = _monthly_sales([1, 2, 3])
    insufficient = chatbot.answer_question("Forecast demand", too_short)
    assert insufficient.status == "insufficient_data"
    assert "don't have enough history" in insufficient.answer


def test_inventory_recommendation_is_advisory():
    transactions = _daily_sales([10] * 10)
    products = pd.DataFrame([{
        "product": "Keyboard",
        "unit_cost": 3,
        "stock_on_hand": 0,
        "lead_time_days": 3,
    }])

    response = chatbot.answer_question("Should I reorder Keyboard?", transactions, products)

    assert response.status == "answered"
    assert response.category == "inventory"
    assert "no order is placed" in response.answer
    assert response.evidence["inventory"][0]["recommended_order_qty"] > 0
    assert response.chart is not None
    assert response.chart.kind == "bar"


def test_inventory_chatbot_rounds_displayed_reorder_point_up():
    transactions = _daily_sales([2, 8] * 5)
    products = pd.DataFrame([{
        "product": "Keyboard",
        "unit_cost": 3,
        "stock_on_hand": 0,
        "lead_time_days": 3,
    }])
    expected = inventory.inventory_recommendations(
        transactions, products
    ).iloc[0]["reorder_point"]

    response = chatbot.answer_question("Should I reorder Keyboard?", transactions, products)

    assert response.evidence["inventory"][0]["reorder_point"] == math.ceil(expected)
    assert response.chart.data[0]["reorder_point"] == math.ceil(expected)
    assert f"reorder point {math.ceil(expected):,}" in response.answer


def test_out_of_scope_ambiguous_and_empty_data_get_friendly_fallbacks():
    transactions = _monthly_sales([1, 2, 3])
    unsupported = chatbot.answer_question("Who won the football game?", transactions)
    assert unsupported.status == "out_of_scope"
    assert "sales and product performance" in unsupported.answer
    assert unsupported.suggestions

    unclear = chatbot.answer_question("?", transactions)
    assert unclear.status == "clarification_needed"

    empty = pd.DataFrame(columns=data.TRANSACTION_COLUMNS)
    missing = chatbot.answer_question("What were my sales?", empty)
    assert missing.status == "insufficient_data"
    assert "don't have transaction data" in missing.answer

    blank = chatbot.answer_question("  ", transactions)
    assert blank.status == "clarification_needed"
    too_long = chatbot.answer_question("sales " + ("x" * chatbot.MAX_QUESTION_LENGTH), transactions)
    assert too_long.status == "clarification_needed"


def test_chatbot_answers_app_help_without_transaction_data():
    empty = pd.DataFrame(columns=data.TRANSACTION_COLUMNS)

    response = chatbot.answer_question("How do I upload a CSV?", empty)

    assert response.status == "answered"
    assert response.category == "help"
    assert "CSV upload (session only)" in response.answer
    assert "does not overwrite the saved CSV files" in response.answer


def test_chatbot_answers_general_help_from_guide():
    transactions = _monthly_sales([1, 2, 3])

    response = chatbot.answer_question("What is BizTrend?", transactions)

    assert response.status == "answered"
    assert response.category == "help"
    assert response.answer.startswith("BizTrend AI is a business analytics")
    assert "forecasts product demand" in response.answer
    assert "inventory replenishment" in response.answer
    assert "unusual activity" in response.answer
    assert "does not place orders" in response.answer
    assert len(response.answer) < 400


@pytest.mark.parametrize(
    "question",
    [
        "BizTrend",
        "BizTrend AI",
        "What is BizTrend?",
        "What is the BizTrend AI app?",
        "Tell me about BizTrend",
        "Explain BizTrend AI",
        "What does BizTrend do?",
        "What can BizTrend AI help me with?",
        "BizTrend AI features",
        "What are the capabilities of BizTrend?",
        "What is BizTrend used for?",
    ],
)
def test_biztrend_name_and_common_description_questions_return_app_summary(question):
    empty = pd.DataFrame(columns=data.TRANSACTION_COLUMNS)
    response = chatbot.answer_question(question, empty)

    assert response.status == "answered"
    assert response.category == "help"
    assert response.answer.startswith("BizTrend AI is a business analytics")
    assert "forecasts product demand" in response.answer


def test_biztrend_business_data_question_is_not_routed_to_app_description():
    response = help_content.answer_for_question("What is my revenue in BizTrend?")

    assert response is None


def test_chatbot_generic_help_still_shows_guide_intro():
    transactions = _monthly_sales([1, 2, 3])

    response = chatbot.answer_question("help", transactions)

    assert response.status == "answered"
    assert response.category == "help"
    assert "Getting started" in response.answer
    assert "Analysis period" in response.answer


def test_chatbot_answers_feature_how_to_from_guide():
    transactions = _monthly_sales([1, 2, 3])

    response = chatbot.answer_question("How does forecasting work?", transactions)

    assert response.status == "answered"
    assert response.category == "help"
    assert "Forecasting" in response.answer
    assert "8 history periods" in response.answer


def test_chatbot_answers_safe_delete_questions_with_help_section():
    transactions = _monthly_sales([1, 2, 3])

    response = chatbot.answer_question("How do I safely delete test data?", transactions)

    assert response.status == "answered"
    assert response.category == "help"
    assert "Type `DELETE` exactly" in response.answer
    assert "Sample demo" in response.answer


def test_forecast_does_not_infer_financial_predictions():
    transactions = _monthly_sales(list(range(1, 10)))
    response = chatbot.answer_question("Forecast my revenue", transactions)

    assert response.status == "unsupported"
    assert "does not forecast revenue or profit" in response.answer


def test_anomaly_workload_limit_returns_prompt_fallback(monkeypatch):
    transactions = _monthly_sales([1, 2, 3])
    monkeypatch.setattr(chatbot, "MAX_ANOMALY_TRANSACTIONS", 1)

    response = chatbot.answer_question("Are there unusual transactions?", transactions)

    assert response.status == "limit_exceeded"
    assert "Narrow the data" in response.answer


def test_unexpected_error_is_logged_and_returns_retry_message(monkeypatch, caplog):
    transactions = _monthly_sales([1, 2, 3])

    def fail(_):
        raise RuntimeError("internal detail")

    monkeypatch.setattr(chatbot.analytics, "sales_by_product", fail)
    with caplog.at_level(logging.ERROR, logger="biztrend.chatbot"):
        response = chatbot.answer_question("Which products sold most?", transactions)

    assert response.status == "error"
    assert "couldn't finish that answer" in response.answer
    assert "Your data hasn't been changed" in response.answer
    assert response.answer == chatbot.UNEXPECTED_ERROR_MESSAGE
    assert not response.suggestions
    assert "internal detail" in caplog.text


def test_streamlit_exception_is_converted_to_the_neutral_chat_message(monkeypatch, caplog):
    from streamlit.errors import StreamlitAPIException

    transactions = _monthly_sales([1, 2, 3])
    expense, _ = data.clean_transactions(pd.DataFrame([{
        "date": "2025-03-15",
        "record_type": "expense",
        "category": "Rent",
        "amount": 100,
    }]))
    transactions = pd.concat([transactions, expense], ignore_index=True)

    def fail(_transactions, _freq):
        raise StreamlitAPIException("internal Streamlit detail")

    monkeypatch.setattr(chatbot.analytics, "financial_performance", fail)
    with caplog.at_level(logging.ERROR, logger="biztrend.chatbot"):
        response = chatbot.answer_question("Review expense categories", transactions)

    assert response.status == "error"
    assert response.answer == chatbot.UNEXPECTED_ERROR_MESSAGE
    assert "internal Streamlit detail" in caplog.text
