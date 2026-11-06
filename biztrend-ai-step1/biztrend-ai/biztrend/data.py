"""Data layer: load, validate, store and aggregate business records.

Canonical transaction schema (one row per sale line or expense):

    date          datetime   when it happened
    record_type   str        "sale" or "expense"
    product       str        product name (sales); may be blank for expenses
    category      str        product category (sales) or expense category
    quantity      float      units sold (sales); 1 for expenses
    unit_price    float      price per unit (sales)
    amount        float      total value, always positive
    notes         str        free text

Optional products table (for inventory intelligence):

    product, unit_cost, stock_on_hand, lead_time_days
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config

TRANSACTION_COLUMNS = [
    "date", "record_type", "product", "category",
    "quantity", "unit_price", "amount", "notes",
]
PRODUCT_COLUMNS = ["product", "unit_cost", "stock_on_hand", "lead_time_days"]

# Accepted header spellings (case-insensitive) -> canonical name
_ALIASES = {
    "date": "date", "transaction_date": "date", "day": "date", "timestamp": "date",
    "record_type": "record_type", "type": "record_type", "kind": "record_type",
    "product": "product", "item": "product", "product_name": "product", "sku": "product",
    "category": "category", "expense_category": "category", "product_category": "category",
    "quantity": "quantity", "qty": "quantity", "units": "quantity",
    "unit_price": "unit_price", "price": "unit_price",
    "amount": "amount", "total": "amount", "revenue": "amount", "value": "amount",
    "notes": "notes", "note": "notes", "description": "notes", "memo": "notes",
    "unit_cost": "unit_cost", "cost": "unit_cost",
    "stock_on_hand": "stock_on_hand", "stock": "stock_on_hand", "on_hand": "stock_on_hand",
    "lead_time_days": "lead_time_days", "lead_time": "lead_time_days",
}

_TYPE_ALIASES = {
    "sale": "sale", "sales": "sale", "income": "sale", "revenue": "sale",
    "expense": "expense", "expenses": "expense", "cost": "expense", "purchase": "expense",
}


@dataclass
class LoadReport:
    """What happened while loading a file: rows kept, rows rejected and why."""
    rows_read: int = 0
    rows_kept: int = 0
    issues: list[str] = field(default_factory=list)

    @property
    def rows_dropped(self) -> int:
        return self.rows_read - self.rows_kept

    def add(self, message: str) -> None:
        self.issues.append(message)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in _ALIASES:
            renamed[col] = _ALIASES[key]
    df = df.rename(columns=renamed)
    # If two source columns map to the same canonical name, keep the first.
    return df.loc[:, ~df.columns.duplicated()]


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, LoadReport]:
    """Validate and standardize a raw transactions table."""
    report = LoadReport(rows_read=len(raw))
    df = _normalize_columns(raw.copy())

    if "date" not in df.columns:
        raise ValueError("Missing a date column (accepted: date, transaction_date, day, timestamp).")
    if "amount" not in df.columns and not {"quantity", "unit_price"} <= set(df.columns):
        raise ValueError("Need an amount column, or both quantity and unit_price.")

    # Defaults for optional columns
    if "record_type" not in df.columns:
        df["record_type"] = "sale"
        report.add("No record_type column found; assuming every row is a sale.")
    for col, default in (("product", ""), ("category", ""), ("notes", "")):
        if col not in df.columns:
            df[col] = default
    if "quantity" not in df.columns:
        df["quantity"] = 1.0
    if "unit_price" not in df.columns:
        df["unit_price"] = float("nan")
    if "amount" not in df.columns:
        df["amount"] = float("nan")

    # Types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("quantity", "unit_price", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["record_type"] = (
        df["record_type"].astype(str).str.strip().str.lower().map(_TYPE_ALIASES)
    )
    for col in ("product", "category", "notes"):
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Derive amount when missing
    missing_amount = df["amount"].isna() & df["quantity"].notna() & df["unit_price"].notna()
    df.loc[missing_amount, "amount"] = df["quantity"] * df["unit_price"]
    if "unit_price" in df and df["unit_price"].isna().any():
        fill = df["unit_price"].isna() & df["quantity"].gt(0) & df["amount"].notna()
        df.loc[fill, "unit_price"] = df["amount"] / df["quantity"]

    # Reject bad rows, with reasons
    checks = [
        (df["date"].isna(), "unreadable or missing date"),
        (df["record_type"].isna(), "record_type is not sale/expense"),
        (df["amount"].isna(), "missing or non-numeric amount"),
        (df["amount"].notna() & (df["amount"] <= 0), "amount must be positive"),
        (df["quantity"].isna() | (df["quantity"] <= 0), "quantity must be positive"),
    ]
    bad = pd.Series(False, index=df.index)
    for mask, reason in checks:
        new = mask & ~bad
        if new.any():
            report.add(f"Dropped {int(new.sum())} row(s): {reason}.")
        bad |= mask
    df = df.loc[~bad]

    dupes = df.duplicated(subset=TRANSACTION_COLUMNS[:-1], keep="first")
    if dupes.any():
        report.add(f"Removed {int(dupes.sum())} exact duplicate row(s).")
        df = df.loc[~dupes]

    df = df[TRANSACTION_COLUMNS].sort_values("date").reset_index(drop=True)
    report.rows_kept = len(df)
    return df, report


def clean_products(raw: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(raw.copy())
    if "product" not in df.columns:
        raise ValueError("Products file needs a product column.")
    for col, default in (("unit_cost", 0.0), ("stock_on_hand", 0.0),
                         ("lead_time_days", config.RULES["default_lead_time_days"])):
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    df["product"] = df["product"].astype(str).str.strip()
    df = df[df["product"] != ""].drop_duplicates("product", keep="last")
    return df[PRODUCT_COLUMNS].reset_index(drop=True)


def load_transactions_csv(path_or_buffer) -> tuple[pd.DataFrame, LoadReport]:
    """Load a transactions CSV (path or uploaded file object)."""
    return clean_transactions(pd.read_csv(path_or_buffer))


def load_products_csv(path_or_buffer) -> pd.DataFrame:
    return clean_products(pd.read_csv(path_or_buffer))


class DataStore:
    """Simple CSV-backed store for one business (no database needed)."""

    def __init__(self, transactions_path: Path = config.TRANSACTIONS_PATH,
                 products_path: Path = config.PRODUCTS_PATH):
        self.transactions_path = Path(transactions_path)
        self.products_path = Path(products_path)

    # --- transactions ---
    def load_transactions(self) -> pd.DataFrame:
        if not self.transactions_path.exists():
            return pd.DataFrame(columns=TRANSACTION_COLUMNS).astype(
                {"date": "datetime64[ns]", "quantity": float, "unit_price": float, "amount": float})
        df, _ = load_transactions_csv(self.transactions_path)
        return df

    def save_transactions(self, df: pd.DataFrame) -> None:
        self.transactions_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.transactions_path, index=False)

    def import_transactions(self, new_df: pd.DataFrame) -> int:
        """Merge cleaned rows into the store, skipping duplicates. Returns rows added."""
        existing = self.load_transactions()
        before = len(existing)
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=TRANSACTION_COLUMNS[:-1], keep="first")
        merged = merged.sort_values("date").reset_index(drop=True)
        self.save_transactions(merged)
        return len(merged) - before

    def add_record(self, *, date, record_type: str, amount: float | None = None,
                   product: str = "", category: str = "", quantity: float = 1.0,
                   unit_price: float | None = None, notes: str = "") -> int:
        """Manual entry of one sale or expense. Returns rows added (0 or 1)."""
        row = pd.DataFrame([{
            "date": date, "record_type": record_type, "product": product,
            "category": category, "quantity": quantity, "unit_price": unit_price,
            "amount": amount, "notes": notes,
        }])
        cleaned, report = clean_transactions(row)
        if cleaned.empty:
            raise ValueError("; ".join(report.issues) or "Record could not be validated.")
        return self.import_transactions(cleaned)

    # --- products ---
    def load_products(self) -> pd.DataFrame:
        if not self.products_path.exists():
            return pd.DataFrame(columns=PRODUCT_COLUMNS)
        return load_products_csv(self.products_path)

    def save_products(self, df: pd.DataFrame) -> None:
        self.products_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.products_path, index=False)

    def clear_all(self) -> None:
        """Replace saved transactions and products with empty canonical tables."""
        self.save_transactions(pd.DataFrame(columns=TRANSACTION_COLUMNS))
        self.save_products(pd.DataFrame(columns=PRODUCT_COLUMNS))


# --- aggregation ---------------------------------------------------------

def aggregate(df: pd.DataFrame, freq: str = "monthly", record_type: str | None = None,
              by: str | None = None, value: str = "amount") -> pd.DataFrame:
    """Aggregate to daily, weekly, monthly, quarterly, or yearly periods.

    Returns a wide table indexed by period start. With `by` (e.g. "product"),
    columns are the groups; otherwise a single column named after `value`.
    Missing periods are filled with 0 so time series have no gaps.
    """
    if freq not in config.FREQUENCIES:
        raise ValueError(f"freq must be one of {list(config.FREQUENCIES)}")
    alias = config.FREQUENCIES[freq]

    data = df if record_type is None else df[df["record_type"] == record_type]
    if data.empty:
        return pd.DataFrame()

    grouper = pd.Grouper(key="date", freq=alias)
    if by:
        out = data.groupby([grouper, by])[value].sum().unstack(by, fill_value=0.0)
    else:
        out = data.groupby(grouper)[value].sum().to_frame(value)
    full = pd.date_range(out.index.min(), out.index.max(), freq=alias)
    return out.reindex(full, fill_value=0.0).rename_axis("period")


def profit_and_loss(df: pd.DataFrame, freq: str = "monthly") -> pd.DataFrame:
    """Revenue, expenses and profit per period."""
    rev = aggregate(df, freq, "sale")
    exp = aggregate(df, freq, "expense")
    idx = rev.index.union(exp.index)
    out = pd.DataFrame(index=idx)
    out["revenue"] = rev["amount"].reindex(idx, fill_value=0.0) if not rev.empty else 0.0
    out["expenses"] = exp["amount"].reindex(idx, fill_value=0.0) if not exp.empty else 0.0
    out["profit"] = out["revenue"] - out["expenses"]
    return out.rename_axis("period")
