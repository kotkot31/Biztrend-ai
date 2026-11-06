"""Load and select user-guide sections for the dashboard and chatbot."""
from __future__ import annotations

import re
from pathlib import Path

_GUIDE_PATH = Path(__file__).resolve().parents[1] / "HELP.md"
_APP_DESCRIPTION_REQUEST = re.compile(
    r"^\s*(?:"
    r"(?:what is|what's|who is)\s+(?:the\s+)?biztrend(?:\s+ai)?"
    r"(?:\s+(?:app|platform|system))?"
    r"|(?:tell me about|describe|explain|give me an overview of)\s+"
    r"(?:the\s+)?biztrend(?:\s+ai)?(?:\s+(?:app|platform|system))?"
    r"|what does\s+(?:the\s+)?biztrend(?:\s+ai)?(?:\s+app)?\s+do"
    r"|what can\s+(?:the\s+)?biztrend(?:\s+ai)?(?:\s+app)?\s+"
    r"(?:do|help me with)"
    r"|biztrend(?:\s+ai)?\s+(?:app\s+)?(?:features?|capabilities|"
    r"core functions|overview|purpose)"
    r"|(?:features?|capabilities|core functions|overview)\s+of\s+"
    r"(?:the\s+)?biztrend(?:\s+ai)?"
    r"|what are (?:the\s+)?(?:features?|capabilities|core functions)\s+of\s+"
    r"(?:the\s+)?biztrend(?:\s+ai)?"
    r"|what is\s+(?:the\s+)?biztrend(?:\s+ai)?\s+"
    r"(?:used for|designed for)"
    r"|biztrend(?:\s+ai)?"
    r")\s*[?.!]*$",
    re.IGNORECASE,
)
_APP_DESCRIPTION = (
    "BizTrend AI is a business analytics and decision-support app. It analyzes "
    "sales, expenses, and profit; forecasts product demand; estimates inventory "
    "replenishment needs; flags unusual activity; and provides evidence-backed "
    "business insights. Its results are advisory, and it does not place orders or "
    "execute financial transactions."
)
_HELP_REQUEST = re.compile(
    r"\bhelp\b|\buser guide\b|\bhow[- ]to\b|\bhow to\b|\bhow do i\b|"
    r"\bhow can i\b|\bwhere can i\b|\bwhere do i\b|\bhow does (?:the )?app\b|"
    r"\bwhat can (?:the|this) (?:app|chatbot|business assistant)\b|"
    r"\bwhat features (?:does|do) (?:biztrend(?: ai)?|the app|this app)\b|"
    r"\bwhat does (?:the )?"
    r"(?:app|dashboard|chatbot|forecasting|inventory|anomaly review|data management|"
    r"business assistant|sales and finance)\b|"
    r"\bhow does (?:the )?(?:dashboard|chatbot|forecasting|inventory|anomaly review|"
    r"data management|business assistant|sales and finance)\b|"
    r"\bwhat is (?:biztrend(?: ai)?|(?:this|the) app|(?:this|the) dashboard)\b|"
    r"\btell me about (?:biztrend(?: ai)?|(?:this|the) app)\b",
    re.IGNORECASE,
)

_SECTION_TOPICS = (
    (
        "Data management and safe deletion",
        ("delete", "clear all", "remove all", "safe delete", "test data"),
    ),
    (
        "Data sources and uploads",
        ("csv", "upload", "import", "data source", "add transaction", "add product", "manual entry"),
    ),
    (
        "Analysis period",
        ("analysis period", "frequency", "daily", "weekly", "monthly", "quarterly", "yearly"),
    ),
    ("Forecasting", ("forecast", "predict", "prediction")),
    ("Inventory", ("inventory", "stock", "reorder", "replenish", "lead time")),
    ("Anomaly review", ("anomaly", "unusual", "outlier", "suspicious")),
    (
        "Business assistant and chatbot",
        ("chatbot", "chat", "business assistant", "recommendation", "recommend"),
    ),
    (
        "Sales and financial analysis",
        ("sales", "revenue", "expense", "profit", "financial", "product performance"),
    ),
    (
        "Themes and app controls",
        ("theme", "dark mode", "light mode", "rerun", "auto rerun", "clear cache"),
    ),
    ("Dashboard pages", ("navigate", "navigation", "dashboard page", "overview page")),
)


def load_guide() -> str:
    """Read the canonical Markdown guide."""
    return _GUIDE_PATH.read_text(encoding="utf-8")


def _section(guide: str, title: str) -> str:
    heading = re.compile(
        rf"^## {re.escape(title)}\s*$([\s\S]*?)(?=^## |\Z)",
        re.MULTILINE,
    )
    match = heading.search(guide)
    if match is None:
        raise ValueError(f"Help guide section is missing: {title}")
    return f"## {title}\n{match.group(1).strip()}"


def answer_for_question(question: str) -> str | None:
    """Return a relevant guide section for an explicit app-help question."""
    if _APP_DESCRIPTION_REQUEST.search(question) is not None:
        return _APP_DESCRIPTION

    if _HELP_REQUEST.search(question) is None:
        return None

    normalized = question.casefold()
    for title, terms in _SECTION_TOPICS:
        if any(term in normalized for term in terms):
            return _section(load_guide(), title)

    return _section(load_guide(), "Getting started")
