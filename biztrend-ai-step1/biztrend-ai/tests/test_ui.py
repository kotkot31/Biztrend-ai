import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException
from streamlit.testing.v1 import AppTest

from biztrend import chatbot
from biztrend import data, ui


def test_branding_is_in_sidebar_not_repeated_in_main_content():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run()

    assert not app.exception
    assert any(element.value == "BizTrend AI" for element in app.sidebar.title)
    assert any(
        element.value == "Business intelligence and predictive decision support"
        for element in app.sidebar.caption
    )
    assert sum(element.value == "BizTrend AI" for element in app.title) == 1
    assert any(element.value == "Business overview" for element in app.title)


def test_chatbot_display_error_shows_neutral_message(monkeypatch, caplog):
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run()
    app.radio[0].set_value("Sample demo")
    app.radio[1].set_value("AI Support")
    app.run()

    def fail_plot_render(*_args, **_kwargs):
        raise StreamlitAPIException("private renderer detail")

    monkeypatch.setattr(st, "plotly_chart", fail_plot_render)
    with caplog.at_level(logging.ERROR):
        app.chat_input[0].set_value("review expense categories").run()

    assert not app.exception
    assert not app.error
    assert chatbot.UNEXPECTED_ERROR_MESSAGE in [message.value for message in app.info]
    assert "private renderer detail" not in str(app)
    assert "private renderer detail" in caplog.text


def test_chatbot_displays_muted_disclaimer_at_bottom_of_chat():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run()
    app.radio[0].set_value("Sample demo")
    app.radio[1].set_value("AI Support")
    app.run()

    assert not app.exception
    assert any(element.value == "AI Support" for element in app.title)
    assert any(
        element.value
        == (
            "Ask the chatbot about sales, expenses, profit, forecasts, "
            "inventory, or unusual activity."
        )
        for element in app.caption
    )
    assert any(
        "Chat analyzes the full available transaction history"
        in element.value
        and "it never places orders or executes financial actions."
        in element.value
        for element in app.caption
    )
    assert any(
        "AI-generated insights are estimates based on available data"
        in element.value
        and "Verify results before making business decisions." in element.value
        for element in app.caption
    )


def test_unusual_daily_demand_displays_expected_units_as_whole_numbers():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()
    app.radio[0].set_value("Sample demo")
    app.radio[1].set_value("Anomaly review")
    app.run(timeout=20)

    assert not app.exception
    assert any(
        element.value
        == (
            "An infinite score means historical demand had no variation, "
            "and the current value differs from that baseline."
        )
        for element in app.caption
    )
    demand_table = app.dataframe[1]
    columns = json.loads(demand_table.proto.columns)
    assert (
        columns["expected_units"]["type_config"]["format"] == "%.0f"
    )


def test_executive_summary_page_renders_sample_business_overview():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()
    app.radio[0].set_value("Sample demo")
    app.radio[1].set_value("Executive summary")
    app.run(timeout=20)

    assert not app.exception
    assert any(element.value == "Executive summary" for element in app.title)
    assert "Net profit" in [element.label for element in app.metric]
    assert "Revenue change" in [element.label for element in app.metric]
    assert len(app.get("plotly_chart")) == 1
    assert any(
        element.value == "Priority insights and recommended actions"
        for element in app.subheader
    )


def test_sales_and_finance_page_renders_expense_analysis_tab():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()
    app.radio[0].set_value("Sample demo")
    app.radio[1].set_value("Sales & finance")
    app.run(timeout=20)

    assert not app.exception
    assert any(element.value == "Expense analysis" for element in app.subheader)
    charts = app.get("plotly_chart")
    financial_chart = json.loads(charts[1].proto.spec)
    category_chart = json.loads(charts[2].proto.spec)
    expense_trend = json.loads(charts[3].proto.spec)

    financial_colors = [
        trace["marker"]["color"] for trace in financial_chart["data"]
    ]
    assert financial_colors[0] == "#8B5CF6"
    assert financial_colors[1] in {"#F0B66A", "#C17A2E"}
    assert financial_colors[2] == "#A7E8A5"
    assert category_chart["layout"]["title"]["text"] == "Expense by category"
    assert category_chart["data"][0]["marker"]["color"] == financial_colors[1]
    assert expense_trend["data"][0]["marker"]["color"] == financial_colors[1]


def test_inventory_page_displays_whole_unit_estimates_and_formatted_costs():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()
    app.radio[0].set_value("Sample demo")
    app.radio[1].set_value("Inventory")
    app.run(timeout=20)

    assert not app.exception
    displayed = app.dataframe[0].value
    for column in (
        "average_daily_demand",
        "daily_demand_std",
        "expected_lead_time_demand",
        "safety_stock",
        "reorder_point",
    ):
        assert displayed[column].mod(1).eq(0).all()

    column_config = json.loads(app.dataframe[0].proto.columns)
    assert column_config["unit_cost"]["type_config"]["format"] == "%,.2f"
    assert column_config["estimated_order_cost"]["type_config"]["format"] == "%,.2f"


def test_merge_transactions_adds_new_rows_and_skips_duplicates():
    existing, _ = data.clean_transactions(pd.DataFrame([{
        "date": "2026-01-01",
        "record_type": "sale",
        "product": "A",
        "quantity": 1,
        "unit_price": 5,
    }]))
    incoming, _ = data.clean_transactions(pd.DataFrame([
        {
            "date": "2026-01-01",
            "record_type": "sale",
            "product": "A",
            "quantity": 1,
            "unit_price": 5,
        },
        {
            "date": "2026-01-02",
            "record_type": "sale",
            "product": "B",
            "quantity": 2,
            "unit_price": 10,
        },
    ]))

    merged, added = ui.merge_transactions(existing, incoming)

    assert added == 1
    assert merged["product"].tolist() == ["A", "B"]
    assert list(merged.columns) == data.TRANSACTION_COLUMNS


def test_merge_products_adds_and_updates_by_product_name():
    existing = pd.DataFrame([{
        "product": "A",
        "unit_cost": 2,
        "stock_on_hand": 5,
        "lead_time_days": 3,
    }])
    incoming = pd.DataFrame([
        {
            "product": "A",
            "unit_cost": 4,
            "stock_on_hand": 7,
            "lead_time_days": 5,
        },
        {
            "product": "B",
            "unit_cost": 1,
            "stock_on_hand": 2,
            "lead_time_days": 2,
        },
    ])

    merged, applied = ui.merge_products(existing, incoming)

    assert applied == 2
    assert merged["product"].tolist() == ["A", "B"]
    assert merged.loc[merged["product"] == "A", "unit_cost"].iloc[0] == 4
    assert merged.loc[merged["product"] == "A", "stock_on_hand"].iloc[0] == 7
