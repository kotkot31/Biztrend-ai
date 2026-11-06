"""Reusable data-management helpers for the Streamlit interface."""
from __future__ import annotations

import pandas as pd

from . import data


def merge_transactions(existing: pd.DataFrame, incoming: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Merge canonical transaction rows, ignoring duplicates like DataStore."""
    merged = pd.concat([existing, incoming], ignore_index=True)
    merged = merged.drop_duplicates(
        subset=data.TRANSACTION_COLUMNS[:-1], keep="first"
    ).sort_values("date").reset_index(drop=True)
    return merged[data.TRANSACTION_COLUMNS], len(merged) - len(existing)


def merge_products(existing: pd.DataFrame, incoming: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Merge products by name, with incoming values taking precedence.

    The returned count is the number of valid incoming product records applied,
    including records that update an existing product.
    """
    incoming = data.clean_products(incoming)
    merged = pd.concat([existing, incoming], ignore_index=True)
    cleaned = data.clean_products(merged)
    return cleaned, len(incoming)
