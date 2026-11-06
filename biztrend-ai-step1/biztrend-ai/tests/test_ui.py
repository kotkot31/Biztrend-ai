import pandas as pd

from biztrend import data, ui


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
