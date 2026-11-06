"""Generate a realistic synthetic dataset for an IT / computer store.

Run:  python -m biztrend.sample_data
Writes data/sample_transactions.csv and data/sample_products.csv.

The data contains deliberate patterns the later modules should discover:
  * weekly seasonality (busier weekends) and a Nov-Dec holiday lift
  * back-to-school peak for laptops, holiday peak for gaming laptops
  * "1TB NVMe SSD" and "Mechanical Keyboard" trending up, "External DVD Drive" declining
  * operating expenses (payroll, marketing) growing faster than revenue
  * daily cost-of-goods and accrued operating expenses for period-aligned P&L
  * a few injected anomalies (a bulk order, an unplanned purchase, a test entry)

Amounts are currency-neutral.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

PRODUCTS = [
    # name, category, price, unit_cost, base daily units, trend/yr, seasonality, stock, lead_time
    ("Laptop 15in Core i5",   "Computers",   749.0, 590.0, 1.8,  0.05, "school",  25, 10),
    ("Gaming Laptop RTX",     "Computers",  1299.0, 1020.0, 0.6,  0.35, "holiday", 8, 14),
    ("27in IPS Monitor",      "Peripherals", 229.0, 160.0, 2.2,  0.15, "none",    30,  7),
    ("Mechanical Keyboard",   "Peripherals",  89.0,  52.0, 4.0,  0.20, "none",    50,  5),
    ("Wireless Mouse",        "Peripherals",  29.0,  14.0, 9.0, -0.05, "none",   120,  5),
    ("USB-C Cable 2m",        "Accessories",  12.0,   3.5, 15.0, 0.00, "none",   300,  4),
    ("1TB NVMe SSD",          "Components",   79.0,  52.0, 3.5,  0.40, "none",    40,  8),
    ("External DVD Drive",    "Peripherals",  35.0,  22.0, 2.5, -0.40, "none",    70,  7),
]

COGS_CATEGORY = config.COGS_CATEGORY

EXPENSE_CATEGORIES = {
    # category: (monthly base, annual growth)
    "Rent": (4200.0, 0.00),
    "Payroll": (9500.0, 0.35),
    "Utilities": (780.0, 0.25),
    "Marketing": (1400.0, 0.50),
    "Shipping & Supplies": (650.0, 0.20),
    "Software Subscriptions": (380.0, 0.30),
}

# kind -> (amplitude, peak position as fraction of the year)
_SEASONS = {
    "school":  (0.45, 0.62),   # peaks around August
    "holiday": (0.50, 0.93),   # peaks around late December
}


def _seasonal(kind: str, day: pd.Timestamp) -> float:
    if kind not in _SEASONS:
        return 1.0
    amp, peak = _SEASONS[kind]
    doy = day.dayofyear / 365.25
    return 1.0 + amp * np.cos(2 * np.pi * (doy - peak))


def generate(start: str = "2025-01-01", end: str = "2026-10-31", seed: int = 42):
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    t0 = days[0]
    unit_cost = {p[0]: p[3] for p in PRODUCTS}
    rows = []

    for day in days:
        years = (day - t0).days / 365.25
        weekend = 1.25 if day.dayofweek >= 5 else 1.0
        holiday = 1.15 if day.month in (11, 12) else 1.0
        for name, cat, price, _cost, base, trend, season, _s, _l in PRODUCTS:
            mean = max(base * (1 + trend * years) * _seasonal(season, day) * weekend * holiday, 0.05)
            units = int(rng.poisson(mean))
            if units == 0:
                continue
            rows.append((day, "sale", name, cat, units, price, round(units * price, 2), ""))

    sales = pd.DataFrame(rows, columns=[
        "date", "record_type", "product", "category", "quantity", "unit_price", "amount", "notes"])

    # Match each day's cost of goods to the sales that incurred it.
    daily_cogs = (
        sales.assign(cogs=sales["quantity"] * sales["product"].map(unit_cost))
        .groupby("date")["cogs"]
        .sum()
    )
    for day, amount in daily_cogs.items():
        if amount > 0:
            cost = round(float(amount), 2)
            rows.append((day, "expense", "", COGS_CATEGORY, 1, cost, cost, "daily cost of goods"))

    # Accrue each monthly operating-cost estimate evenly across its calendar days.
    for month in pd.period_range(start, end, freq="M"):
        month_start = month.start_time
        month_end = month.end_time.normalize()
        month_days = pd.date_range(month_start, month_end, freq="D")
        active_days = month_days[(month_days >= days[0]) & (month_days <= days[-1])]
        years = (month_start - t0).days / 365.25
        for cat, (base, growth) in EXPENSE_CATEGORIES.items():
            monthly_amount = round(
                base * (1 + growth * years) * rng.normal(1.0, 0.02), 2
            )
            total_cents = round(monthly_amount * 100)
            daily_cents, remainder = divmod(total_cents, len(month_days))
            for day in active_days:
                cents = daily_cents + int(day.day <= remainder)
                amount = cents / 100
                rows.append((
                    day, "expense", "", cat, 1, amount, amount,
                    "daily accrued operating expense",
                ))

    df = pd.DataFrame(rows, columns=sales.columns)

    # Inject anomalies the detector should flag
    anomalies = pd.DataFrame([
        (pd.Timestamp("2026-03-14"), "sale", "Laptop 15in Core i5", "Computers", 40, 749.0, 29960.00,
         "bulk order, verify buyer"),
        (pd.Timestamp("2026-06-09"), "expense", "", "Shipping & Supplies", 1, 9800.00, 9800.00,
         "unplanned equipment purchase"),
        (pd.Timestamp("2026-08-21"), "sale", "USB-C Cable 2m", "Accessories", 1, 12.0, 12.00,
         "test entry"),
    ], columns=df.columns)
    anomalies = anomalies[(anomalies["date"] >= days[0]) & (anomalies["date"] <= days[-1])]
    df = pd.concat([df, anomalies], ignore_index=True).sort_values("date").reset_index(drop=True)

    products = pd.DataFrame(
        [(n, c_, st, lt) for n, _cat, _p, c_, _b, _t, _s, st, lt in PRODUCTS],
        columns=["product", "unit_cost", "stock_on_hand", "lead_time_days"])
    return df, products


def main() -> None:
    tx, products = generate()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tx.to_csv(config.SAMPLE_TRANSACTIONS_PATH, index=False)
    products.to_csv(config.SAMPLE_PRODUCTS_PATH, index=False)
    print(f"Wrote {len(tx):,} transactions -> {config.SAMPLE_TRANSACTIONS_PATH}")
    print(f"Wrote {len(products)} products -> {config.SAMPLE_PRODUCTS_PATH}")


if __name__ == "__main__":
    main()
