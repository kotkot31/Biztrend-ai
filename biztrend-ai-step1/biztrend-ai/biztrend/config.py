"""Central configuration and business-rule defaults.

Single-business deployment: one organization, many products.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

TRANSACTIONS_PATH = DATA_DIR / "transactions.csv"
PRODUCTS_PATH = DATA_DIR / "products.csv"
SAMPLE_TRANSACTIONS_PATH = DATA_DIR / "sample_transactions.csv"
SAMPLE_PRODUCTS_PATH = DATA_DIR / "sample_products.csv"

# Supported analysis granularities -> pandas offset aliases
FREQUENCIES = {
    "daily": "D",
    "weekly": "W-SUN",
    "monthly": "MS",
    "quarterly": "QS",
    "yearly": "YS",
}

RECORD_TYPES = ("sale", "expense")

# Expense category treated as cost of goods sold (excluded from operating expenses)
COGS_CATEGORY = "Cost of Goods"

# Business rules used by later modules (tunable by the owner)
RULES = {
    "anomaly_zscore_threshold": 3.0,
    "inventory_service_level_z": 1.65,   # ~95% service level
    "default_lead_time_days": 7,
    "expense_growth_warning_gap": 0.05,  # expenses growing >5 pts faster than revenue
    "min_history_points": 8,             # minimum periods needed to fit a forecast
}
