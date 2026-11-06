import pandas as pd
import pytest

from biztrend import anomaly, data, sample_data


def _records(rows):
    raw = pd.DataFrame(rows, columns=[
        "date", "record_type", "product", "category", "quantity", "amount",
    ])
    raw["unit_price"] = raw["amount"] / raw["quantity"]
    cleaned, _ = data.clean_transactions(raw)
    return cleaned


def test_transaction_detection_flags_bulk_sale_in_same_product_cohort():
    rows = [
        (f"2026-01-{day:02}", "sale", "A", "Hardware", 1, 10)
        for day in range(1, 11)
    ]
    rows.append(("2026-01-11", "sale", "A", "Hardware", 40, 400))

    result = anomaly.detect_transaction_anomalies(_records(rows))

    assert len(result) == 1
    assert result.iloc[0]["quantity"] == 40
    assert result.iloc[0]["anomaly_metric"] == "quantity"
    assert result.iloc[0]["anomaly_score"] == float("inf")
    assert result.iloc[0]["peer_count"] == 10
    assert result.iloc[0]["comparison_group"] == "sale:A"
    assert "unusually high quantity" in result.iloc[0]["anomaly_reason"]


def test_transaction_detection_uses_expense_category_and_handles_low_outliers():
    rows = [
        (f"2026-01-{day:02}", "expense", "", "Rent", 1, 500)
        for day in range(1, 11)
    ]
    rows.append(("2026-01-11", "expense", "", "Rent", 1, 50))

    result = anomaly.detect_transaction_anomalies(_records(rows))

    assert len(result) == 1
    assert result.iloc[0]["amount"] == 50
    assert result.iloc[0]["comparison_group"] == "expense:Rent"
    assert result.iloc[0]["anomaly_score"] == float("-inf")
    assert "unusually low amount" in result.iloc[0]["anomaly_reason"]


def test_transaction_detector_skips_groups_without_enough_peers():
    rows = [
        ("2026-01-01", "sale", "A", "Hardware", 1, 10),
        ("2026-01-02", "sale", "A", "Hardware", 100, 1000),
    ]

    result = anomaly.detect_transaction_anomalies(
        _records(rows), minimum_history=3
    )

    assert result.empty
    assert list(result.columns) == anomaly.TRANSACTION_ANOMALY_COLUMNS


def test_daily_demand_detection_uses_only_prior_active_days():
    rows = [
        (f"2026-01-{day:02}", "sale", "A", "Hardware", 2, 20)
        for day in range(1, 11)
    ]
    rows.append(("2026-01-11", "sale", "A", "Hardware", 30, 300))
    result = anomaly.detect_daily_demand_anomalies(_records(rows))

    assert len(result) == 1
    assert result.iloc[0]["date"] == pd.Timestamp("2026-01-11")
    assert result.iloc[0]["units_sold"] == 30
    assert result.iloc[0]["expected_units"] == 2
    assert result.iloc[0]["direction"] == "unusually_high"
    assert result.iloc[0]["history_points"] == 10


def test_daily_demand_detector_does_not_flag_when_history_is_short():
    rows = [
        (f"2026-01-{day:02}", "sale", "A", "Hardware", 2, 20)
        for day in range(1, 6)
    ]
    rows.append(("2026-01-06", "sale", "A", "Hardware", 30, 300))
    result = anomaly.detect_daily_demand_anomalies(
        _records(rows), minimum_history=6
    )
    assert result.empty


def test_anomaly_detectors_validate_parameters_and_empty_input():
    empty = pd.DataFrame(columns=data.TRANSACTION_COLUMNS)
    assert anomaly.detect_transaction_anomalies(empty).empty
    assert anomaly.detect_daily_demand_anomalies(empty).empty

    with pytest.raises(ValueError, match="threshold"):
        anomaly.detect_transaction_anomalies(empty, threshold=0)
    with pytest.raises(ValueError, match="minimum_history"):
        anomaly.detect_transaction_anomalies(empty, minimum_history=2)
    with pytest.raises(ValueError, match="minimum_history"):
        anomaly.detect_daily_demand_anomalies(empty, minimum_history=3.5)
    with pytest.raises(ValueError, match="window"):
        anomaly.detect_daily_demand_anomalies(empty, minimum_history=5, window=4)


def test_detectors_find_injected_sample_business_anomalies():
    transactions, _ = sample_data.generate("2025-01-01", "2026-10-31")

    transaction_alerts = anomaly.detect_transaction_anomalies(transactions)
    assert transaction_alerts["notes"].str.contains("bulk order").any()
    assert transaction_alerts["notes"].str.contains("unplanned equipment purchase").any()

    demand_alerts = anomaly.detect_daily_demand_anomalies(transactions)
    assert (
        (demand_alerts["product"] == "Laptop 15in Core i5")
        & (demand_alerts["date"] == pd.Timestamp("2026-03-14"))
    ).any()
