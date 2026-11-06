import io

import pandas as pd
import pytest

from biztrend import analytics, data, sample_data


def test_alias_headers_and_derived_amount():
    csv = io.StringIO("Date,Type,Item,Qty,Price\n2026-01-05,sales,Latte,2,4.5\n")
    df, report = data.load_transactions_csv(csv)
    assert len(df) == 1
    assert df.loc[0, "record_type"] == "sale"
    assert df.loc[0, "product"] == "Latte"
    assert df.loc[0, "amount"] == pytest.approx(9.0)
    assert report.rows_dropped == 0


def test_bad_rows_are_dropped_with_reasons():
    csv = io.StringIO(
        "date,record_type,product,quantity,unit_price,amount\n"
        "2026-01-05,sale,A,1,5,5\n"
        "not-a-date,sale,A,1,5,5\n"
        "2026-01-06,refund,A,1,5,5\n"
        "2026-01-07,sale,A,1,5,-5\n"
    )
    df, report = data.load_transactions_csv(csv)
    assert len(df) == 1
    assert report.rows_read == 4 and report.rows_dropped == 3
    assert len(report.issues) == 3


def test_missing_required_columns_raise():
    with pytest.raises(ValueError):
        data.clean_transactions(pd.DataFrame({"product": ["A"], "amount": [1]}))
    with pytest.raises(ValueError):
        data.clean_transactions(pd.DataFrame({"date": ["2026-01-01"]}))


def test_duplicates_removed():
    csv = io.StringIO("date,type,product,amount\n2026-01-05,sale,A,5\n2026-01-05,sale,A,5\n")
    df, report = data.load_transactions_csv(csv)
    assert len(df) == 1
    assert any("duplicate" in i for i in report.issues)


def test_store_manual_entry_and_import_dedup(tmp_path):
    store = data.DataStore(tmp_path / "t.csv", tmp_path / "p.csv")
    assert store.load_transactions().empty
    assert store.add_record(date="2026-02-01", record_type="sale", product="A",
                            quantity=3, unit_price=2.0) == 1
    assert store.add_record(date="2026-02-01", record_type="sale", product="A",
                            quantity=3, unit_price=2.0) == 0  # duplicate ignored
    assert store.add_record(date="2026-02-02", record_type="expense",
                            category="Rent", amount=1000) == 1
    assert len(store.load_transactions()) == 2
    with pytest.raises(ValueError):
        store.add_record(date="2026-02-03", record_type="sale", amount=-1)


def test_store_clear_all_removes_transactions_and_products(tmp_path):
    store = data.DataStore(tmp_path / "transactions.csv", tmp_path / "products.csv")
    store.add_record(
        date="2026-02-01",
        record_type="sale",
        product="Test product",
        quantity=1,
        unit_price=10,
    )
    store.save_products(pd.DataFrame([{
        "product": "Test product",
        "unit_cost": 5,
        "stock_on_hand": 2,
        "lead_time_days": 3,
    }]))

    store.clear_all()

    assert store.load_transactions().empty
    assert store.load_products().empty
    assert list(pd.read_csv(tmp_path / "transactions.csv").columns) == data.TRANSACTION_COLUMNS
    assert list(pd.read_csv(tmp_path / "products.csv").columns) == data.PRODUCT_COLUMNS


def test_aggregate_fills_gaps_and_frequencies():
    df, _ = data.clean_transactions(pd.DataFrame({
        "date": ["2026-01-01", "2026-01-04"], "type": ["sale", "sale"],
        "product": ["A", "A"], "amount": [10, 20]}))
    daily = data.aggregate(df, "daily", "sale")
    assert len(daily) == 4 and daily["amount"].tolist() == [10, 0, 0, 20]
    monthly = data.aggregate(df, "monthly", "sale")
    assert monthly["amount"].iloc[0] == 30
    with pytest.raises(ValueError):
        data.aggregate(df, "hourly")


def test_aggregate_supports_quarterly_and_yearly_periods():
    df, _ = data.clean_transactions(pd.DataFrame({
        "date": ["2024-01-05", "2024-03-20", "2024-04-02", "2025-02-01"],
        "type": ["sale"] * 4,
        "product": ["A"] * 4,
        "amount": [10, 20, 30, 40],
    }))

    quarterly = data.aggregate(df, "quarterly", "sale")
    yearly = data.aggregate(df, "yearly", "sale")

    assert quarterly.index.tolist() == list(pd.date_range("2024-01-01", "2025-01-01", freq="QS"))
    assert quarterly["amount"].tolist() == [30, 30, 0, 0, 40]
    assert yearly.index.tolist() == list(pd.date_range("2024-01-01", "2025-01-01", freq="YS"))
    assert yearly["amount"].tolist() == [60, 40]


def test_aggregate_by_product_and_pnl():
    df, _ = sample_data.generate("2026-01-01", "2026-03-31")
    by_product = data.aggregate(df, "monthly", "sale", by="product")
    assert set(by_product.columns) == {p[0] for p in sample_data.PRODUCTS}
    pnl = data.profit_and_loss(df, "monthly")
    assert list(pnl.columns) == ["revenue", "expenses", "profit"]
    assert len(pnl) == 3
    assert (pnl["profit"] == pnl["revenue"] - pnl["expenses"]).all()


def test_sample_data_roundtrip_is_clean(tmp_path):
    tx, products = sample_data.generate("2026-01-01", "2026-02-28")
    path = tmp_path / "tx.csv"
    tx.to_csv(path, index=False)
    cleaned, report = data.load_transactions_csv(path)
    assert report.rows_dropped == 0
    assert len(cleaned) == len(tx)
    assert len(data.clean_products(products)) == len(products)


def test_sample_expenses_are_accrued_daily_and_roll_up_by_period():
    transactions, _ = sample_data.generate("2026-10-01", "2026-10-31")
    daily = analytics.financial_performance(transactions, "daily")
    monthly = analytics.financial_performance(transactions, "monthly")

    assert len(daily) == 31
    assert daily.loc[pd.Timestamp("2026-10-31"), "total_expenses"] < 10_000
    assert daily["total_expenses"].sum() == pytest.approx(
        monthly.iloc[0]["total_expenses"]
    )
    operating_expenses = transactions[
        (transactions["record_type"] == "expense")
        & (transactions["category"] == "Rent")
    ]
    assert len(operating_expenses) == 31
