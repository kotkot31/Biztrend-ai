"""Streamlit dashboard for BizTrend AI.

Run from the project directory with: streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from biztrend import (
    anomaly,
    analytics,
    assistant,
    chatbot,
    config,
    data,
    forecast,
    help_content,
    inventory,
    sample_data,
    ui,
)

st.set_page_config(
    page_title="BizTrend AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
_LOGGER = logging.getLogger(__name__)
streamlit_theme = getattr(getattr(getattr(st, "context", None), "theme", None), "type", None)
IS_DARK_THEME = streamlit_theme == "dark"
THEME_COLORS = (
    {
        "accent": "#59CBB7",
        "blue": "#83ADFF",
        "purple": "#B69AE8",
        "orange": "#F0B66A",
        "coral": "#F18E9C",
        "text": "#E7EEF5",
        "grid": "rgba(224, 235, 246, 0.12)",
        "surface": "#192737",
    }
    if IS_DARK_THEME
    else {
        "accent": "#168B83",
        "blue": "#3478C8",
        "purple": "#8063B5",
        "orange": "#C17A2E",
        "coral": "#C95669",
        "text": "#20344A",
        "grid": "rgba(42, 69, 98, 0.12)",
        "surface": "#FFFFFF",
    }
)
px.defaults.template = "plotly_dark" if streamlit_theme == "dark" else "plotly_white"
px.defaults.color_discrete_sequence = [
    THEME_COLORS["blue"],
    THEME_COLORS["accent"],
    THEME_COLORS["orange"],
    THEME_COLORS["purple"],
    THEME_COLORS["coral"],
]
FINANCIAL_CHART_COLORS = {
    "revenue": "#8B5CF6",
    "total_expenses": THEME_COLORS["orange"],
    "net_profit": "#A7E8A5",
}

st.markdown(
    f"""
    <style>
    :root {{
        --biztrend-accent: {THEME_COLORS["accent"]};
        --biztrend-blue: {THEME_COLORS["blue"]};
        --biztrend-purple: {THEME_COLORS["purple"]};
        --biztrend-orange: {THEME_COLORS["orange"]};
        --biztrend-coral: {THEME_COLORS["coral"]};
        --biztrend-text: {THEME_COLORS["text"]};
        --biztrend-grid: {THEME_COLORS["grid"]};
        --biztrend-surface: {THEME_COLORS["surface"]};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"] {
        padding-top: 2.25rem;
        padding-bottom: 3.5rem;
    }

    [data-testid="stMainMenuPopover"] [data-testid="stMainMenuItem-theme-System"] {
        display: none !important;
    }

    [data-testid="stMain"] h1 {
        letter-spacing: -0.035em;
        font-weight: 750;
        background: linear-gradient(105deg, var(--biztrend-accent), var(--biztrend-blue) 58%, var(--biztrend-purple));
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3 {
        letter-spacing: -0.02em;
    }

    [data-testid="stMetric"] {
        border: 1px solid color-mix(in srgb, var(--biztrend-text) 12%, transparent);
        border-top: 3px solid var(--metric-accent, var(--biztrend-accent));
        border-radius: 16px;
        padding: 1.05rem 1.15rem;
        background: linear-gradient(145deg, color-mix(in srgb, var(--metric-accent, var(--biztrend-accent)) 7%, var(--biztrend-surface)), var(--biztrend-surface) 72%);
        box-shadow: 0 6px 20px color-mix(in srgb, var(--biztrend-text) 7%, transparent);
        transition: transform 160ms ease, box-shadow 160ms ease;
    }

    [data-testid="stHorizontalBlock"] > div:nth-child(1) [data-testid="stMetric"] {
        --metric-accent: var(--biztrend-accent);
    }

    [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stMetric"] {
        --metric-accent: var(--biztrend-blue);
    }

    [data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetric"] {
        --metric-accent: var(--biztrend-purple);
    }

    [data-testid="stHorizontalBlock"] > div:nth-child(4) [data-testid="stMetric"] {
        --metric-accent: var(--biztrend-orange);
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px color-mix(in srgb, var(--metric-accent, var(--biztrend-accent)) 18%, transparent);
    }

    [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"],
    [data-testid="stMetricValue"] [data-testid="stMarkdownContainer"] {
        overflow: visible;
        white-space: normal;
        text-overflow: clip;
    }

    [data-testid="stMetricLabel"] p {
        display: block;
        overflow: visible;
        white-space: normal;
        text-overflow: clip;
        line-height: 1.25;
    }

    [data-testid="stMetricValue"] p {
        display: block;
        overflow: visible;
        overflow-wrap: normal;
        white-space: normal;
        text-overflow: clip;
        font-size: clamp(1rem, 1.35vw, 1.3rem);
        line-height: 1.2;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid color-mix(in srgb, var(--biztrend-accent) 24%, transparent);
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 10px;
        padding: 0.35rem 0.55rem;
        transition: background 140ms ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: color-mix(in srgb, var(--biztrend-accent) 14%, transparent);
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: color-mix(in srgb, var(--biztrend-accent) 20%, transparent);
        border: 1px solid color-mix(in srgb, var(--biztrend-accent) 38%, transparent);
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        border-color: color-mix(in srgb, var(--biztrend-accent) 45%, transparent);
        border-radius: 10px;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        border-color: color-mix(in srgb, var(--biztrend-accent) 42%, transparent);
        background: color-mix(in srgb, var(--biztrend-accent) 8%, transparent);
    }

    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        border-radius: 10px;
        font-weight: 650;
        transition: transform 140ms ease, box-shadow 140ms ease;
    }

    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        border-color: var(--biztrend-accent);
        box-shadow: 0 5px 14px color-mix(in srgb, var(--biztrend-accent) 20%, transparent);
        transform: translateY(-1px);
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--biztrend-accent);
        border-bottom-color: var(--biztrend-accent);
    }

    [data-testid="stTabs"] [role="tab"]:hover {
        color: var(--biztrend-blue);
    }

    [data-testid="stExpander"] {
        border-color: color-mix(in srgb, var(--biztrend-blue) 24%, transparent);
        border-radius: 12px;
        background: color-mix(in srgb, var(--biztrend-blue) 3%, var(--biztrend-surface));
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid color-mix(in srgb, var(--biztrend-blue) 20%, transparent);
    }

    [data-testid="stDataFrame"],
    [data-testid="stPlotlyChart"] {
        border-radius: 12px;
        border: 1px solid color-mix(in srgb, var(--biztrend-blue) 15%, transparent);
        box-shadow: 0 5px 18px color-mix(in srgb, var(--biztrend-text) 5%, transparent);
    }

    [data-testid="stChatMessage"] {
        border: 1px solid color-mix(in srgb, var(--biztrend-accent) 16%, transparent);
        border-radius: 14px;
        background: color-mix(in srgb, var(--biztrend-accent) 4%, var(--biztrend-surface));
    }

    @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
            transition-duration: 0.01ms !important;
            animation-duration: 0.01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _style_figure(figure: go.Figure) -> go.Figure:
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": THEME_COLORS["text"], "size": 13},
        title={"font": {"size": 19}},
        hoverlabel={
            "bgcolor": THEME_COLORS["surface"],
            "font": {"color": THEME_COLORS["text"]},
            "bordercolor": THEME_COLORS["accent"],
        },
        legend={"bgcolor": "rgba(0,0,0,0)"},
        margin={"l": 18, "r": 18, "t": 58, "b": 18},
    )
    figure.update_xaxes(
        showgrid=False,
        zerolinecolor=THEME_COLORS["grid"],
        automargin=True,
    )
    figure.update_yaxes(
        gridcolor=THEME_COLORS["grid"],
        zerolinecolor=THEME_COLORS["grid"],
        automargin=True,
    )
    for trace in figure.data:
        if trace.type == "scatter" and trace.mode and "lines" in trace.mode:
            trace.line.width = 3
            if "markers" in trace.mode:
                trace.marker.size = 7
        elif trace.type == "bar":
            trace.marker.line.width = 0
    return figure


@st.cache_data(show_spinner="Preparing sample business data...")
def _sample_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    return sample_data.generate()


def _empty_transactions() -> pd.DataFrame:
    return pd.DataFrame(columns=data.TRANSACTION_COLUMNS)


def _empty_products() -> pd.DataFrame:
    return pd.DataFrame(columns=data.PRODUCT_COLUMNS)


def _read_transactions(upload) -> tuple[pd.DataFrame | None, data.LoadReport | None]:
    if upload is None:
        return None, None
    try:
        return data.load_transactions_csv(upload)
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        st.sidebar.error(f"Could not read transactions CSV: {exc}")
        return None, None


def _read_products(upload) -> pd.DataFrame | None:
    if upload is None:
        return None
    try:
        return data.load_products_csv(upload)
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        st.sidebar.error(f"Could not read products CSV: {exc}")
        return None


def _get_dataset(source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if source == "Sample demo":
        return _sample_dataset()
    if source == "CSV upload (session only)":
        tx_upload = st.session_state.get("dataset_transactions_upload")
        if tx_upload is not None:
            tx_bytes = tx_upload.getvalue()
            tx_fingerprint = hashlib.sha256(tx_bytes).hexdigest()
            if tx_fingerprint != st.session_state.get("session_transactions_fingerprint"):
                transactions, _ = _read_transactions(io.BytesIO(tx_bytes))
                st.session_state["session_transactions"] = (
                    transactions if transactions is not None else _empty_transactions()
                )
                st.session_state["session_transactions_fingerprint"] = tx_fingerprint
        elif "session_transactions" not in st.session_state:
            st.session_state["session_transactions"] = _empty_transactions()

        product_upload = st.session_state.get("dataset_products_upload")
        if product_upload is not None:
            product_bytes = product_upload.getvalue()
            product_fingerprint = hashlib.sha256(product_bytes).hexdigest()
            if product_fingerprint != st.session_state.get("session_products_fingerprint"):
                products = _read_products(io.BytesIO(product_bytes))
                st.session_state["session_products"] = (
                    products if products is not None else _empty_products()
                )
                st.session_state["session_products_fingerprint"] = product_fingerprint
        elif "session_products" not in st.session_state:
            st.session_state["session_products"] = _empty_products()
        return (
            st.session_state["session_transactions"],
            st.session_state["session_products"],
        )
    try:
        store = data.DataStore()
        return store.load_transactions(), store.load_products()
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        st.error(f"Could not load saved business data: {exc}")
        st.stop()


def _money(value: float) -> str:
    return f"{value:,.2f}"


def _period_date_range(transactions: pd.DataFrame, period: pd.Timestamp, freq: str) -> str:
    if freq == "daily":
        start = end = period
    elif freq == "weekly":
        start, end = period - pd.Timedelta(days=6), period
    elif freq == "monthly":
        start, end = period, period + pd.offsets.MonthEnd(0)
    elif freq == "quarterly":
        start, end = period, period + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    else:
        start, end = period, period + pd.DateOffset(years=1) - pd.Timedelta(days=1)
    end = min(end, transactions["date"].max().normalize())
    return f"{start:%b %d, %Y} – {end:%b %d, %Y}"


def _expense_period_comparison(
    breakdown: pd.DataFrame,
    period: pd.Timestamp,
    previous_period: pd.Timestamp | None,
) -> pd.DataFrame:
    current = breakdown[breakdown["period"] == period].set_index("category")
    previous = (
        breakdown[breakdown["period"] == previous_period].set_index("category")
        if previous_period is not None
        else breakdown.iloc[0:0].set_index("category")
    )
    categories = sorted(set(current.index) | set(previous.index))
    current_total = float(current["amount"].sum())
    rows = []
    for category in categories:
        amount = float(current["amount"].get(category, 0.0))
        previous_amount = float(previous["amount"].get(category, 0.0))
        expense_type = (
            current["expense_type"].get(category)
            or previous["expense_type"].get(category)
            or "Operating expense"
        )
        rows.append({
            "Category": category,
            "Type": expense_type,
            "Amount": amount,
            "Share of expenses": amount / current_total * 100 if current_total else 0.0,
            "Previous period": previous_amount,
            "Change": amount - previous_amount,
            "Change (%)": (
                (amount / previous_amount - 1) * 100
                if previous_amount > 0
                else None
            ),
        })
    return pd.DataFrame(rows)


def _render_overview(transactions: pd.DataFrame, products: pd.DataFrame, freq: str) -> None:
    st.title("Business overview")
    st.caption("A clear view of sales, financial performance, and areas to review.")
    if transactions.empty:
        st.info("No transactions are available. Add records, import a CSV, or choose Sample demo in the sidebar.")
        return

    summary = analytics.sales_summary(transactions)
    pnl = analytics.financial_performance(transactions, freq)
    latest = pnl.iloc[-1] if not pnl.empty else None
    period_name = {
        "daily": "day",
        "weekly": "week",
        "monthly": "month",
        "quarterly": "quarter",
        "yearly": "year",
    }[freq]
    metrics = st.columns(4)
    coverage = (
        f"{transactions['date'].min():%b %d, %Y} – "
        f"{transactions['date'].max():%b %d, %Y}"
    )
    metrics[0].metric(
        "All-time revenue",
        _money(summary["revenue"]),
        help=f"Total recorded sales revenue across all available history ({coverage}).",
    )
    metrics[1].metric(
        "Units sold",
        f"{summary['units_sold']:,.0f}",
        help=f"Total units across sale records in the available history ({coverage}).",
    )
    latest_help = "No financial period is available."
    if latest is not None:
        period_label = analytics.period_label(latest.name, freq)
        period_dates = _period_date_range(transactions, latest.name, freq)
        latest_help = (
            f"{period_label} ({period_dates}). "
            f"Revenue: {_money(latest['revenue'])}; "
            f"expenses: {_money(latest['total_expenses'])}; "
            f"net margin: {latest['net_margin_pct']:.1f}%."
            if pd.notna(latest["net_margin_pct"])
            else (
                f"{period_label} ({period_dates}). "
                f"Revenue: {_money(latest['revenue'])}; "
                f"expenses: {_money(latest['total_expenses'])}; "
                "net margin is unavailable because revenue is zero."
            )
        )
    metrics[2].metric(
        f"Latest {period_name} net profit",
        _money(latest["net_profit"]) if latest is not None else "—",
        help=latest_help,
    )
    products_count = products["product"].nunique() if not products.empty else summary["products_sold"]
    metrics[3].metric(
        "Products tracked",
        f"{products_count}",
        help=(
            "Distinct products in the product catalog."
            if not products.empty
            else "Distinct named products found in sales records; no product catalog is loaded."
        ),
    )

    trends = analytics.sales_trends(transactions, freq).reset_index()
    if not trends.empty and freq in {"quarterly", "yearly"}:
        trends["period"] = trends["period"].map(
            lambda period: analytics.period_label(period, freq)
        )
    if not trends.empty:
        fig = px.line(
            trends,
            x="period",
            y="revenue",
            markers=True,
            title=f"Revenue by {freq} period",
            labels={"period": "Period", "revenue": "Revenue"},
        )
        st.plotly_chart(_style_figure(fig), use_container_width=True)

    left, right = st.columns(2)
    products_by_sales = analytics.sales_by_product(transactions).head(5)
    with left:
        st.subheader("Top products")
        if products_by_sales.empty:
            st.caption("No named product sales are available.")
        else:
            st.dataframe(products_by_sales, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Latest period at a glance")
        if latest is None:
            st.caption("No financial periods are available.")
        else:
            st.metric("Expenses", _money(latest["total_expenses"]))
            st.metric("Net margin", f"{latest['net_margin_pct']:.1f}%" if pd.notna(latest["net_margin_pct"]) else "—")
            if bool(latest["expense_growth_warning"]):
                st.warning("Operating expenses grew faster than revenue in the latest period.")


def _render_executive_summary(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    freq: str,
) -> None:
    st.title("Executive summary")
    st.caption(
        "A concise, evidence-backed view of business performance and items to review."
    )
    if transactions.empty:
        st.info(
            "Add or upload business records, or choose Sample demo, to build an "
            "executive summary."
        )
        return

    with st.spinner("Preparing your executive summary..."):
        report = assistant.generate_business_report(
            transactions, products=products, freq=freq, forecast_periods=3
        )

    financial = report.financial_performance
    latest = financial.iloc[-1] if not financial.empty else None
    if latest is not None:
        latest_period = financial.index[-1]
        period_label = analytics.period_label(latest_period, freq)
        coverage = _period_date_range(transactions, latest_period, freq)
        period_help = f"Selected period: {period_label} ({coverage})."
        revenue = float(latest["revenue"])
        net_profit = float(latest["net_profit"])
        net_margin = (
            f"{latest['net_margin_pct']:.1f}%"
            if pd.notna(latest["net_margin_pct"])
            else "Unavailable"
        )
        revenue_change = (
            f"{latest['revenue_growth_pct']:+.1f}%"
            if pd.notna(latest["revenue_growth_pct"])
            else "—"
        )
        revenue_change_help = (
            f"Change from the previous {freq} period."
            if pd.notna(latest["revenue_growth_pct"])
            else f"Not available: there is no comparable previous {freq} period."
        )
    else:
        period_label = "No financial period"
        coverage = "Financial period coverage is unavailable."
        period_help = coverage
        revenue = 0.0
        net_profit = 0.0
        net_margin = "—"
        revenue_change = "—"
        revenue_change_help = "No financial period is available."

    stock = report.inventory_recommendations
    reorder_count = (
        int(stock["reorder_status"].eq("reorder").sum())
        if not stock.empty and "reorder_status" in stock
        else 0
    )
    st.caption(
        f"Analysis period: **{freq.title()}** · Latest period: **{period_label}** · "
        f"Data coverage: {transactions['date'].min():%b %d, %Y} – "
        f"{transactions['date'].max():%b %d, %Y}"
    )

    priorities = {insight.priority for insight in report.insights}
    if "urgent" in priorities:
        st.warning(
            "An urgent financial insight was flagged. Review the evidence and "
            "underlying records before making decisions."
        )
    elif "review" in priorities:
        st.warning(
            "Some items were flagged for review. Check the recommendations and "
            "supporting evidence below."
        )
    elif "watch" in priorities:
        st.info(
            "A demand trend is worth monitoring. Forecasts are estimates, not "
            "purchasing instructions."
        )
    else:
        st.success(
            "No high-priority items were flagged in the available data. This does "
            "not guarantee that all business activity is normal."
        )

    primary_metrics = st.columns(3)
    primary_metrics[0].metric(
        f"{freq.title()} revenue", _money(revenue), help=period_help
    )
    primary_metrics[1].metric(
        "Net profit",
        _money(net_profit),
        help=f"{period_help} Profit is revenue minus expenses recorded in this period.",
    )
    primary_metrics[2].metric(
        "Net margin",
        net_margin,
        help="Net profit as a percentage of revenue for the selected period.",
    )
    secondary_metrics = st.columns(2)
    secondary_metrics[0].metric(
        "Revenue change", revenue_change, help=revenue_change_help
    )
    secondary_metrics[1].metric(
        "Inventory reviews",
        f"{reorder_count:,}",
        help="Products at or below their estimated reorder point. No orders are placed.",
    )

    expense_breakdown = analytics.expense_category_breakdown(transactions, freq)
    if latest is not None:
        latest_expenses = expense_breakdown[
            expense_breakdown["period"] == financial.index[-1]
        ]
        if not latest_expenses.empty:
            category_totals = latest_expenses.groupby("category")["amount"].sum()
            top_category = category_totals.idxmax()
            expense_total = float(category_totals.sum())
            category_share = float(category_totals[top_category] / expense_total * 100)
            st.caption(
                f"Largest expense category in the latest period: "
                f"**{top_category}** — {_money(float(category_totals[top_category]))} "
                f"({category_share:.1f}% of expenses)."
            )
        else:
            st.caption("No expenses were recorded in the latest selected period.")

    if not financial.empty:
        trend = financial.tail(8).reset_index()
        if freq in {"quarterly", "yearly"}:
            trend["period"] = trend["period"].map(
                lambda period: analytics.period_label(period, freq)
            )
        figure = px.line(
            trend,
            x="period",
            y=["revenue", "net_profit"],
            markers=True,
            labels={"period": "Period", "value": "Amount", "variable": "Measure"},
            title="Revenue and net profit trend",
            color_discrete_map=FINANCIAL_CHART_COLORS,
        )
        st.plotly_chart(_style_figure(figure), use_container_width=True)

    st.subheader("Priority insights and recommended actions")
    if not report.insights:
        st.info(
            "No rule-based insights were generated. Check that the selected data "
            "contains enough relevant history and product details."
        )
    else:
        for insight in report.insights[:4]:
            with st.container(border=True):
                st.caption(f"{insight.priority.upper()} · {insight.category.title()}")
                st.markdown(f"#### {insight.title}")
                st.write(insight.explanation)
                st.markdown(f"**Suggested review:** {insight.recommended_action}")
                with st.expander("Supporting evidence"):
                    st.json(insight.evidence)

    st.caption(
        f"Inventory review items: {reorder_count:,} · "
        f"Transaction rows analyzed: {len(transactions):,} · {coverage}"
    )
    st.caption(
        "Insights and recommendations are advisory and depend on the quality and "
        "history of the selected data. No purchases or financial actions are executed."
    )


def _render_sales(transactions: pd.DataFrame, freq: str) -> None:
    st.title("Sales and financial intelligence")
    if transactions.empty:
        st.info("Add or upload transactions to see sales and financial analysis.")
        return
    trends = analytics.sales_trends(transactions, freq).reset_index()
    financial = analytics.financial_performance(transactions, freq).reset_index()
    if freq in {"quarterly", "yearly"}:
        financial["period"] = financial["period"].map(
            lambda period: analytics.period_label(period, freq)
        )
    sales, finance, expenses = st.tabs(
        ["Sales", "Financial performance", "Expense analysis"]
    )
    with sales:
        st.subheader("Sales trends")
        fig = px.line(
            trends,
            x="period",
            y=["revenue", "units_sold"],
            markers=True,
            labels={"value": "Amount / units", "period": "Period", "variable": "Measure"},
            color_discrete_map={"revenue": FINANCIAL_CHART_COLORS["revenue"]},
        )
        st.plotly_chart(_style_figure(fig), use_container_width=True)
        st.subheader("Product performance")
        product_sales = analytics.sales_by_product(transactions)
        st.dataframe(product_sales, use_container_width=True, hide_index=True)
    with finance:
        st.subheader("Profit and loss")
        fig = px.bar(
            financial,
            x="period",
            y=["revenue", "total_expenses", "net_profit"],
            barmode="group",
            labels={"value": "Amount", "period": "Period", "variable": "Measure"},
            color_discrete_map=FINANCIAL_CHART_COLORS,
        )
        st.plotly_chart(_style_figure(fig), use_container_width=True)
        st.dataframe(financial, use_container_width=True, hide_index=True)
        warnings = financial[financial["expense_growth_warning"]]
        if not warnings.empty:
            st.warning(f"{len(warnings)} period(s) show operating expenses growing faster than revenue.")
    with expenses:
        _render_expense_analysis(transactions, freq)


def _render_expense_analysis(transactions: pd.DataFrame, freq: str) -> None:
    st.subheader("Expense analysis")
    financial = analytics.financial_performance(transactions, freq)
    if financial.empty:
        st.info("No financial periods are available for expense analysis.")
        return

    latest_period = financial.index[-1]
    previous_period = financial.index[-2] if len(financial) > 1 else None
    latest = financial.iloc[-1]
    period_name = analytics.period_label(latest_period, freq)
    metrics = st.columns(3)
    metrics[0].metric("Total expenses", _money(float(latest["total_expenses"])))
    metrics[1].metric("Cost of goods", _money(float(latest["cost_of_goods"])))
    metrics[2].metric("Operating expenses", _money(float(latest["operating_expenses"])))
    st.caption(f"Selected period: **{period_name}** · Compared with the previous {freq} period.")

    breakdown = analytics.expense_category_breakdown(transactions, freq)
    current = breakdown[breakdown["period"] == latest_period]
    if current.empty:
        st.info("No expenses were recorded in this period.")
    else:
        category_totals = (
            current.groupby("category", as_index=False)["amount"]
            .sum()
            .sort_values("amount", ascending=True)
        )
        figure = px.bar(
            category_totals,
            x="amount",
            y="category",
            orientation="h",
            labels={"amount": "Amount", "category": "Category"},
            title="Expense by category",
            color_discrete_sequence=[FINANCIAL_CHART_COLORS["total_expenses"]],
        )
        figure.update_layout(
            height=max(360, 44 * len(category_totals) + 130),
            margin={"l": 32, "r": 24, "t": 58, "b": 36},
        )
        st.plotly_chart(_style_figure(figure), use_container_width=True)

    trend = financial.tail(12).reset_index()
    trend["period"] = trend["period"].map(
        lambda period: analytics.period_label(period, freq)
    )
    figure = px.bar(
        trend,
        x="period",
        y="total_expenses",
        labels={"total_expenses": "Expenses", "period": "Period"},
        title="Expense trend",
        color_discrete_sequence=[FINANCIAL_CHART_COLORS["total_expenses"]],
    )
    figure.update_layout(
        height=360,
        margin={"l": 32, "r": 24, "t": 58, "b": 72},
    )
    figure.update_xaxes(tickangle=-35)
    st.plotly_chart(_style_figure(figure), use_container_width=True)

    comparison = _expense_period_comparison(
        breakdown, latest_period, previous_period
    )
    if comparison.empty:
        st.info("Add expense records to see category shares and period comparisons.")
    else:
        st.subheader("Category amounts and period comparison")
        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Amount": st.column_config.NumberColumn(format="%.2f"),
                "Previous period": st.column_config.NumberColumn(format="%.2f"),
                "Change": st.column_config.NumberColumn(format="%+.2f"),
                "Share of expenses": st.column_config.NumberColumn(format="%.1f%%"),
                "Change (%)": st.column_config.NumberColumn(format="%+.1f%%"),
            },
        )


def _render_forecast(transactions: pd.DataFrame, freq: str) -> None:
    st.title("Demand forecasting")
    if transactions.empty:
        st.info("Add or upload transactions before forecasting demand.")
        return
    available_products = sorted(
        transactions.loc[
            (transactions["record_type"] == "sale") & transactions["product"].ne(""),
            "product",
        ].unique()
    )
    options = ["All products"] + available_products
    selected = st.selectbox("Forecast scope", options)
    horizon = st.number_input("Periods to forecast", min_value=1, max_value=24, value=3)
    product = None if selected == "All products" else selected
    result = forecast.forecast_demand(
        transactions, periods=int(horizon), freq=freq, product=product
    )
    if result.status != "ok":
        st.warning(
            "No forecast was produced: "
            + ("no matching sales records." if result.status == "no_sales"
               else f"only {result.history_periods} history periods are available; "
                    f"{config.RULES['min_history_points']} are required.")
        )
        return

    metrics = st.columns(3)
    metrics[0].metric("Selected model", result.selected_model or "—")
    metrics[1].metric("Validation MAE", f"{result.validation_mae:.2f}" if result.validation_mae is not None else "—")
    metrics[2].metric("History periods", result.history_periods)
    history = data.aggregate(
        transactions[transactions["record_type"] == "sale"]
        if product is None
        else transactions[
            (transactions["record_type"] == "sale")
            & (transactions["product"] == product)
        ],
        freq,
        value="quantity",
    ).rename(columns={"quantity": "units_sold"})
    chart = pd.DataFrame(index=history.index)
    chart["units_sold"] = history["units_sold"]
    chart["forecast_demand"] = float("nan")
    chart.loc[result.forecast.index[0], "forecast_demand"] = history["units_sold"].iloc[-1]
    for period, row in result.forecast.iterrows():
        chart.loc[period, "forecast_demand"] = row["forecast_demand"]
    chart = chart.reset_index()
    if freq in {"quarterly", "yearly"}:
        chart["period"] = chart["period"].map(
            lambda period: analytics.period_label(period, freq)
        )
    fig = px.line(
        chart,
        x="period",
        y=["units_sold", "forecast_demand"],
        markers=True,
        labels={"period": "Period", "value": "Units", "variable": "Series"},
        title=f"Observed and forecast demand — {selected}",
    )
    st.plotly_chart(_style_figure(fig), use_container_width=True)
    st.dataframe(result.forecast, use_container_width=True)
    st.caption("Forecasts are estimates for planning; they are not purchase commitments.")


def _render_inventory(transactions: pd.DataFrame, products: pd.DataFrame) -> None:
    st.title("Inventory intelligence")
    if products.empty:
        st.info("No product stock records are available. Add or import a products CSV in Data management.")
        return
    result = inventory.inventory_recommendations(transactions, products)
    if result.empty:
        st.info("No products are available for inventory analysis.")
        return
    reorder = result[result["reorder_status"] == "reorder"]
    insufficient = result[result["reorder_status"].isin(["no_sales", "insufficient_history"])]
    cols = st.columns(3)
    cols[0].metric("Products analyzed", len(result))
    cols[1].metric("At/below reorder point", len(reorder))
    cols[2].metric("Insufficient demand history", len(insufficient))
    st.dataframe(result, use_container_width=True, hide_index=True)
    if not reorder.empty:
        st.warning("Replenishment quantities are estimates for review only. No orders have been placed.")


def _render_anomalies(transactions: pd.DataFrame) -> None:
    st.title("Anomaly review")
    if transactions.empty:
        st.info("Add or upload transactions to review unusual activity.")
        return
    transaction_alerts = anomaly.detect_transaction_anomalies(transactions)
    demand_alerts = anomaly.detect_daily_demand_anomalies(transactions)
    st.caption("Statistical flags help prioritize review; they are not findings of fraud or wrongdoing.")
    transaction_tab, demand_tab = st.tabs(["Unusual transactions", "Unusual daily demand"])
    with transaction_tab:
        if transaction_alerts.empty:
            st.info("No transaction alerts met the configured threshold and history requirements.")
        else:
            st.dataframe(transaction_alerts, use_container_width=True, hide_index=True)
    with demand_tab:
        if demand_alerts.empty:
            st.info("No demand alerts met the configured threshold and history requirements.")
        else:
            st.dataframe(demand_alerts, use_container_width=True, hide_index=True)


def _render_assistant(transactions: pd.DataFrame, products: pd.DataFrame, freq: str) -> None:
    st.title("Business assistant")
    st.caption("Rule-based, evidence-backed business insights for review.")
    if transactions.empty:
        st.info("Add or upload transactions to generate business insights.")
        return
    with st.spinner("Analyzing business data..."):
        report = assistant.generate_business_report(
            transactions, products=products, freq=freq, forecast_periods=3
        )
    if not report.insights:
        st.info("Not enough data or no sales were available to generate insights.")
        return
    for insight in report.insights:
        label = f"{insight.priority.upper()} · {insight.category.title()} · {insight.title}"
        with st.expander(label, expanded=insight.priority in {"urgent", "review"}):
            st.write(insight.explanation)
            st.markdown(f"**Suggested review:** {insight.recommended_action}")
            st.json(insight.evidence)


def _render_chatbot(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    freq: str,
    source: str,
) -> None:
    st.title("BizTrend Q&A")
    st.caption(
        "Ask the chatbot about sales, expenses, profit, forecasts, inventory, "
        "or unusual activity. Answers are based on available business data."
    )
    st.info(
        "Chat analyzes the full available transaction history. It cannot answer "
        "custom date ranges or forecast revenue/profit yet. Recommendations are "
        "advisory; it never places orders or executes financial actions."
    )
    state_suffix = f"{source}_{freq}".replace(" ", "_").replace("&", "and")
    history_key = f"biztrend_chat_history_{state_suffix}"
    history = st.session_state.setdefault(history_key, [])

    def render_chart(chart: dict) -> None:
        frame = pd.DataFrame(chart["data"])
        if frame.empty:
            return
        if chart["kind"] == "line":
            figure = px.line(
                frame,
                x=chart["x"],
                y=list(chart["y"]),
                markers=True,
                title=chart["title"],
            )
        else:
            figure = px.bar(
                frame,
                x=chart["x"],
                y=list(chart["y"]),
                barmode="group",
                title=chart["title"],
            )
        st.plotly_chart(_style_figure(figure), use_container_width=True)

    for index, message in enumerate(history):
        try:
            with st.chat_message(message["role"]):
                if message.get("status") == "error":
                    st.info(chatbot.UNEXPECTED_ERROR_MESSAGE)
                    continue
                st.markdown(message["content"])
                if message.get("chart"):
                    render_chart(message["chart"])
                if message.get("evidence"):
                    with st.expander("Supporting evidence"):
                        st.write(message["evidence"])
                if message.get("suggestions"):
                    st.caption("You could also ask:")
                    for suggestion in message["suggestions"]:
                        st.markdown(f"- {suggestion}")
        except Exception:
            _LOGGER.exception("Unexpected error while rendering chatbot history.")
            history[index] = {
                "role": "assistant",
                "content": chatbot.UNEXPECTED_ERROR_MESSAGE,
                "status": "error",
            }
            with st.chat_message("assistant"):
                st.info(chatbot.UNEXPECTED_ERROR_MESSAGE)

    question = st.chat_input(
        "Ask a question about your business data...",
        max_chars=chatbot.MAX_QUESTION_LENGTH,
        key=f"biztrend_chat_input_{state_suffix}",
    )
    if question:
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        response = chatbot.ChatResponse(
            status="error",
            category="error",
            answer=chatbot.UNEXPECTED_ERROR_MESSAGE,
        )
        chart_data = None
        try:
            with st.chat_message("assistant"):
                with st.spinner("Checking available business data..."):
                    response = chatbot.answer_question(
                        question,
                        transactions,
                        products=products,
                        freq=freq,
                    )
                if response.status == "error":
                    st.info(chatbot.UNEXPECTED_ERROR_MESSAGE)
                elif response.status == "answered":
                    st.markdown(response.answer)
                else:
                    st.info(response.answer)
                if response.status != "error" and response.chart is not None:
                    chart_data = {
                        "kind": response.chart.kind,
                        "data": response.chart.data,
                        "x": response.chart.x,
                        "y": response.chart.y,
                        "title": response.chart.title,
                    }
                    render_chart(chart_data)
                if response.status != "error" and response.evidence:
                    with st.expander("Supporting evidence"):
                        st.write(response.evidence)
                if response.status != "error" and response.suggestions:
                    st.caption("You could also ask:")
                    for suggestion in response.suggestions:
                        st.markdown(f"- {suggestion}")
        except Exception:
            _LOGGER.exception("Unexpected error while rendering chatbot response.")
            response = chatbot.ChatResponse(
                status="error",
                category="error",
                answer=chatbot.UNEXPECTED_ERROR_MESSAGE,
            )
            chart_data = None
            with st.chat_message("assistant"):
                st.info(chatbot.UNEXPECTED_ERROR_MESSAGE)
        history.append({
            "role": "assistant",
            "content": (
                chatbot.UNEXPECTED_ERROR_MESSAGE
                if response.status == "error"
                else response.answer
            ),
            "status": response.status,
            "evidence": response.evidence if response.status != "error" else {},
            "chart": chart_data,
            "suggestions": (
                response.suggestions if response.status != "error" else ()
            ),
        })


def _save_transactions(
    incoming: pd.DataFrame,
    source: str,
    transactions: pd.DataFrame,
) -> int:
    if source == "Saved business data":
        return data.DataStore().import_transactions(incoming)
    if source == "CSV upload (session only)":
        merged, added = ui.merge_transactions(transactions, incoming)
        st.session_state["session_transactions"] = merged
        return added
    raise ValueError("Sample demo data is read-only.")


def _save_products(
    incoming: pd.DataFrame,
    source: str,
    products: pd.DataFrame,
) -> int:
    if source == "Saved business data":
        store = data.DataStore()
        merged, added = ui.merge_products(store.load_products(), incoming)
        store.save_products(merged)
        return added
    if source == "CSV upload (session only)":
        merged, added = ui.merge_products(products, incoming)
        st.session_state["session_products"] = merged
        return added
    raise ValueError("Sample demo data is read-only.")


def _delete_all_data(source: str) -> None:
    if st.session_state.get("delete_all_confirmation") != "DELETE":
        st.session_state["delete_all_status"] = (
            "error",
            "Type DELETE exactly to confirm removal.",
        )
        return

    try:
        if source == "Saved business data":
            data.DataStore().clear_all()
        elif source == "CSV upload (session only)":
            st.session_state["session_transactions"] = _empty_transactions()
            st.session_state["session_products"] = _empty_products()
            for key in (
                "dataset_transactions_upload",
                "dataset_products_upload",
                "manage_transactions_csv",
                "manage_products_csv",
                "session_transactions_fingerprint",
                "session_products_fingerprint",
            ):
                st.session_state.pop(key, None)
        else:
            raise ValueError("Sample demo data is read-only.")

        source_key = source.replace(" ", "_").replace("&", "and")
        history_prefix = f"biztrend_chat_history_{source_key}_"
        for key in list(st.session_state.keys()):
            if key.startswith(history_prefix):
                st.session_state.pop(key, None)
        st.session_state["delete_all_status"] = (
            "success",
            f"All transactions and products were deleted from {source}.",
        )
    except (OSError, ValueError) as exc:
        st.session_state["delete_all_status"] = (
            "error",
            f"Could not delete all data: {exc}",
        )
    finally:
        st.session_state["delete_all_confirmation"] = ""


def _render_data_management(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    source: str,
) -> None:
    st.title("Data management")
    st.caption("Imported records are validated before use. Saved business data is written only after you submit an import or entry.")
    if source == "Sample demo":
        st.info("Sample demo data is read-only. Switch to Saved business data or CSV upload to add records.")
        st.subheader("Current sample")
        st.caption(f"{len(transactions):,} transactions · {len(products):,} products")
        return

    with st.expander("Delete all test data", expanded=False):
        st.warning(
            "This permanently removes all transactions and products from the selected "
            "writable data source. This action cannot be undone."
        )
        st.caption(
            f"Selected source: {source} · "
            f"{len(transactions):,} transaction(s) · {len(products):,} product(s)"
        )
        delete_status = st.session_state.pop("delete_all_status", None)
        if delete_status:
            status_type, message = delete_status
            if status_type == "success":
                st.success(message)
            else:
                st.error(message)
        with st.form("delete_all_data_form"):
            st.text_input("Type DELETE to confirm", key="delete_all_confirmation")
            st.form_submit_button(
                "Permanently delete all data",
                type="secondary",
                on_click=_delete_all_data,
                args=(source,),
            )

    imports, manual, product_tab = st.tabs(["Import CSV", "Add transaction", "Manage products"])
    with imports:
        tx_file = st.file_uploader("Transactions CSV", type=["csv"], key="manage_transactions_csv")
        if tx_file is not None:
            loaded, report = _read_transactions(tx_file)
            if loaded is not None and report is not None:
                st.success(
                    f"Validated {report.rows_kept:,} transaction row(s); "
                    f"{report.rows_dropped:,} row(s) dropped."
                )
                for issue in report.issues:
                    st.warning(issue)
                if st.button("Import transactions", key="import_transactions_button"):
                    try:
                        added = _save_transactions(loaded, source, transactions)
                        st.success(f"Imported {added:,} new transaction row(s).")
                        st.rerun()
                    except (OSError, ValueError) as exc:
                        st.error(f"Could not save transactions: {exc}")

        product_file = st.file_uploader("Products CSV", type=["csv"], key="manage_products_csv")
        if product_file is not None:
            loaded_products = _read_products(product_file)
            if loaded_products is not None:
                st.success(f"Validated {len(loaded_products):,} product record(s).")
                if st.button("Import products", key="import_products_button"):
                    try:
                        added = _save_products(loaded_products, source, products)
                        st.success(f"Imported {added:,} product record(s).")
                        st.rerun()
                    except (OSError, ValueError) as exc:
                        st.error(f"Could not save products: {exc}")

    with manual:
        with st.form("manual_transaction"):
            record_type = st.selectbox("Record type", ["sale", "expense"])
            date = st.date_input("Date")
            product = st.text_input("Product (for sales)")
            category = st.text_input("Category")
            quantity = st.number_input("Quantity", min_value=0.01, value=1.0, step=1.0)
            unit_price = st.number_input(
                "Unit price (optional)", min_value=0.0, value=0.0, step=1.0
            )
            amount = st.number_input(
                "Total amount (optional if quantity and unit price are entered)",
                min_value=0.0,
                value=0.0,
                step=1.0,
            )
            notes = st.text_input("Notes")
            submitted = st.form_submit_button("Save transaction")
        if submitted:
            if record_type == "sale" and not product.strip():
                st.error("Enter a product name for a sale.")
            else:
                try:
                    price_value = unit_price if unit_price > 0 else None
                    amount_value = amount if amount > 0 else None
                    if source == "Saved business data":
                        added = data.DataStore().add_record(
                            date=date,
                            record_type=record_type,
                            product=product,
                            category=category,
                            quantity=quantity,
                            unit_price=price_value,
                            amount=amount_value,
                            notes=notes,
                        )
                    else:
                        row, report = data.clean_transactions(pd.DataFrame([{
                            "date": date,
                            "record_type": record_type,
                            "product": product,
                            "category": category,
                            "quantity": quantity,
                            "unit_price": price_value,
                            "amount": amount_value,
                            "notes": notes,
                        }]))
                        if row.empty:
                            raise ValueError("; ".join(report.issues) or "Transaction is invalid.")
                        added = _save_transactions(row, source, transactions)
                    st.success("Transaction saved." if added else "Duplicate transaction was not added.")
                    st.rerun()
                except (OSError, ValueError) as exc:
                    st.error(f"Could not save transaction: {exc}")

    with product_tab:
        st.subheader("Current products")
        if products.empty:
            st.caption("No product inventory records are available.")
        else:
            st.dataframe(products, use_container_width=True, hide_index=True)
        with st.form("manual_product"):
            name = st.text_input("Product name")
            unit_cost = st.number_input("Unit cost", min_value=0.0, value=0.0, step=1.0)
            stock = st.number_input("Stock on hand", min_value=0.0, value=0.0, step=1.0)
            lead_time = st.number_input(
                "Supplier lead time (days)", min_value=1.0,
                value=float(config.RULES["default_lead_time_days"]), step=1.0,
            )
            submitted_product = st.form_submit_button("Save product")
        if submitted_product:
            if not name.strip():
                st.error("Enter a product name.")
            else:
                try:
                    row = data.clean_products(pd.DataFrame([{
                        "product": name,
                        "unit_cost": unit_cost,
                        "stock_on_hand": stock,
                        "lead_time_days": lead_time,
                    }]))
                    added = _save_products(row, source, products)
                    st.success("Product saved." if added else "Product was updated.")
                    st.rerun()
                except (OSError, ValueError) as exc:
                    st.error(f"Could not save product: {exc}")


def _render_help() -> None:
    st.title("Help & guide")
    st.caption("How to use BizTrend AI, understand its analysis, and manage your data safely.")
    st.markdown(help_content.load_guide())


st.title("BizTrend AI")
st.caption("Business intelligence and predictive decision support")
with st.sidebar:
    st.header("Workspace")
    source = st.radio(
        "Data source",
        ["Saved business data", "Sample demo", "CSV upload (session only)"],
        help=(
            "Saved business data uses local CSV files. Sample demo is read-only. "
            "CSV upload is kept in this browser session and does not overwrite saved files."
        ),
    )
    if source == "CSV upload (session only)":
        st.file_uploader(
            "Upload transactions CSV",
            type=["csv"],
            key="dataset_transactions_upload",
            help="Headers are matched flexibly; invalid rows are reported.",
        )
        st.file_uploader(
            "Upload products CSV (optional)",
            type=["csv"],
            key="dataset_products_upload",
        )
    frequency = st.selectbox("Analysis period", list(config.FREQUENCIES))
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Executive summary",
            "Sales & finance",
            "Forecasting",
            "Inventory",
            "Anomaly review",
            "Business assistant",
            "Chatbot",
            "Data management",
            "Help & guide",
        ],
    )
    with st.expander("App controls", expanded=False):
        st.caption("These controls are in the three-dot menu at the top-right.")
        st.markdown(
            "- **Rerun:** Re-executes the app with the current data and selections.\n"
            "- **Auto rerun:** Automatically reruns the app when its source code changes.\n"
            "- **Clear cache:** Removes cached data and resources; the next run may take longer while they are rebuilt."
        )

transactions, products = _get_dataset(source)
if source == "Saved business data" and not transactions.empty:
    st.sidebar.caption(f"{len(transactions):,} saved transaction row(s)")
elif source == "CSV upload (session only)":
    st.sidebar.caption(f"{len(transactions):,} session transaction row(s)")

if page == "Overview":
    _render_overview(transactions, products, frequency)
elif page == "Executive summary":
    _render_executive_summary(transactions, products, frequency)
elif page == "Sales & finance":
    _render_sales(transactions, frequency)
elif page == "Forecasting":
    _render_forecast(transactions, frequency)
elif page == "Inventory":
    _render_inventory(transactions, products)
elif page == "Anomaly review":
    _render_anomalies(transactions)
elif page == "Business assistant":
    _render_assistant(transactions, products, frequency)
elif page == "Chatbot":
    _render_chatbot(transactions, products, frequency, source)
elif page == "Data management":
    _render_data_management(transactions, products, source)
else:
    _render_help()
