"""Robust outlier detection for transactions and daily product demand."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config, data

TRANSACTION_ANOMALY_COLUMNS = [
    *data.TRANSACTION_COLUMNS,
    "comparison_group",
    "anomaly_metric",
    "anomaly_score",
    "anomaly_reason",
    "peer_count",
]
DEMAND_ANOMALY_COLUMNS = [
    "date",
    "product",
    "units_sold",
    "expected_units",
    "anomaly_score",
    "direction",
    "history_points",
]


def _validate_settings(threshold: float, minimum_history: int) -> None:
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be a positive finite number.")
    if (
        isinstance(minimum_history, bool)
        or not isinstance(minimum_history, (int, np.integer))
        or minimum_history < 3
    ):
        raise ValueError("minimum_history must be an integer of at least 3.")


def _robust_zscore(values: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    median = float(np.median(baseline))
    deviations = np.abs(baseline - median)
    mad = float(np.median(deviations))
    if mad > 0:
        return 0.6745 * (values - median) / mad

    q25, q75 = np.percentile(baseline, [25, 75])
    iqr = float(q75 - q25)
    if iqr > 0:
        return (values - median) / (iqr / 1.349)

    scores = np.zeros_like(values, dtype=float)
    scores[values > median] = np.inf
    scores[values < median] = -np.inf
    return scores


def _canonical_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    missing = set(data.TRANSACTION_COLUMNS) - set(transactions.columns)
    if missing:
        raise ValueError(f"transactions is missing required columns: {sorted(missing)}")
    return transactions[data.TRANSACTION_COLUMNS].reset_index(drop=True).copy()


def detect_transaction_anomalies(
    transactions: pd.DataFrame,
    threshold: float | None = None,
    minimum_history: int | None = None,
) -> pd.DataFrame:
    """Flag unusual transaction amounts or quantities versus comparable records.

    Sales are compared within product; expenses are compared within category.
    A robust modified z-score is calculated against all other records in the
    group, so the candidate transaction cannot distort its own baseline.
    Groups smaller than ``minimum_history`` produce no alerts.
    """
    limit = float(
        config.RULES["anomaly_zscore_threshold"] if threshold is None else threshold
    )
    min_records = (
        config.RULES["min_history_points"]
        if minimum_history is None else minimum_history
    )
    _validate_settings(limit, min_records)
    df = _canonical_transactions(transactions)
    if df.empty:
        return pd.DataFrame(columns=TRANSACTION_ANOMALY_COLUMNS)

    entity = df["product"].where(
        (df["record_type"] == "sale") & df["product"].ne(""),
        df["category"].where(df["category"].ne(""), "Uncategorized"),
    )
    df["_comparison_group"] = df["record_type"].astype(str) + ":" + entity.astype(str)
    alerts: list[dict[str, object]] = []

    for group_name, group in df.groupby("_comparison_group", sort=False):
        if len(group) <= min_records:
            continue
        indices = group.index.to_numpy()
        for metric in ("quantity", "amount"):
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            for position, row_index in enumerate(indices):
                baseline = np.delete(values, position)
                if len(baseline) < min_records:
                    continue
                score = float(_robust_zscore(values[position:position + 1], baseline)[0])
                if abs(score) <= limit:
                    continue
                alerts.append({
                    "row_index": int(row_index),
                    "comparison_group": str(group_name),
                    "anomaly_metric": metric,
                    "anomaly_score": score,
                    "anomaly_reason": (
                        f"unusually {'high' if score > 0 else 'low'} {metric} "
                        f"compared with {len(baseline)} similar records"
                    ),
                    "peer_count": len(baseline),
                })

    if not alerts:
        return pd.DataFrame(columns=TRANSACTION_ANOMALY_COLUMNS)

    alert_frame = pd.DataFrame(alerts)
    alert_frame["_absolute_score"] = alert_frame["anomaly_score"].abs()
    alert_frame = alert_frame.sort_values(
        ["row_index", "_absolute_score"], ascending=[True, False]
    ).drop_duplicates("row_index", keep="first").reset_index(drop=True)
    result = df.iloc[alert_frame["row_index"].to_numpy()][data.TRANSACTION_COLUMNS]
    result = result.reset_index(drop=True)
    result["comparison_group"] = alert_frame["comparison_group"].to_numpy()
    result["anomaly_metric"] = alert_frame["anomaly_metric"].to_numpy()
    result["anomaly_score"] = alert_frame["anomaly_score"].to_numpy()
    result["anomaly_reason"] = alert_frame["anomaly_reason"].to_numpy()
    result["peer_count"] = alert_frame["peer_count"].to_numpy()
    return result[TRANSACTION_ANOMALY_COLUMNS]


def detect_daily_demand_anomalies(
    transactions: pd.DataFrame,
    threshold: float | None = None,
    minimum_history: int | None = None,
    window: int = 28,
) -> pd.DataFrame:
    """Flag unusual daily units sold versus prior active selling days per product.

    Zero-sale calendar days are omitted from the baseline so intermittent
    products are compared with their prior selling days rather than an
    artificial run of zeros. The current day is never included in its own
    trailing baseline.
    """
    limit = float(
        config.RULES["anomaly_zscore_threshold"] if threshold is None else threshold
    )
    min_records = (
        config.RULES["min_history_points"]
        if minimum_history is None else minimum_history
    )
    _validate_settings(limit, min_records)
    if isinstance(window, bool) or not isinstance(window, int) or window < min_records:
        raise ValueError("window must be an integer greater than or equal to minimum_history.")
    df = _canonical_transactions(transactions)
    if df.empty:
        return pd.DataFrame(columns=DEMAND_ANOMALY_COLUMNS)

    sales = df[(df["record_type"] == "sale") & df["product"].ne("")]
    alerts: list[dict[str, object]] = []
    for product, product_sales in sales.groupby("product", sort=False):
        daily = data.aggregate(product_sales, "daily", value="quantity")
        if daily.empty:
            continue
        active = daily.loc[daily["quantity"] > 0, "quantity"]
        values = active.to_numpy(dtype=float)
        dates = active.index
        for position, current in enumerate(values):
            start = max(0, position - window)
            baseline = values[start:position]
            if len(baseline) < min_records:
                continue
            score = float(_robust_zscore(values[position:position + 1], baseline)[0])
            if abs(score) <= limit:
                continue
            alerts.append({
                "date": dates[position],
                "product": product,
                "units_sold": float(current),
                "expected_units": float(np.median(baseline)),
                "anomaly_score": score,
                "direction": "unusually_high" if score > 0 else "unusually_low",
                "history_points": len(baseline),
            })

    if not alerts:
        return pd.DataFrame(columns=DEMAND_ANOMALY_COLUMNS)
    return pd.DataFrame(alerts, columns=DEMAND_ANOMALY_COLUMNS).sort_values(
        ["date", "product"]
    ).reset_index(drop=True)
