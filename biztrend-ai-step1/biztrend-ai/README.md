# BizTrend AI

An intelligent business analytics and predictive decision support system using machine learning and adaptive forecasting.

Single-business deployment, batch processing, daily/weekly/monthly/quarterly/yearly analysis. The system **supports** decisions; it never executes transactions or purchasing commitments.

## Roadmap (built step by step)

| Step | Scope | Status |
|------|-------|--------|
| 1 | Project scaffold, data layer (CSV import, validation, manual entry, aggregation), sample data, tests | Done |
| 2 | Sales Intelligence + Financial Intelligence | Done |
| 3 | Demand Forecasting (adaptive model selection) | Done |
| 4 | Inventory Intelligence | Done |
| 5 | Anomaly Detection | Done |
| 6 | AI Business Assistant (rule-based, from model outputs) | Done |
| 7 | Streamlit dashboard and deterministic business Q&A | Done |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m biztrend.sample_data # writes data/sample_*.csv
pytest
```

Start the dashboard from the project directory:

```bash
streamlit run app.py
```

## Data format

Transactions CSV (headers are matched flexibly, e.g. `qty`, `item`, `type`):

`date, record_type (sale|expense), product, category, quantity, unit_price, amount, notes`

- `amount` is derived from `quantity * unit_price` if missing.
- Invalid rows are dropped and reported, never silently.

Optional products CSV for inventory: `product, unit_cost, stock_on_hand, lead_time_days`.

Your real data lives in `data/transactions.csv` and `data/products.csv`, which are git-ignored.

## Sales and financial analysis

The `biztrend.analytics` module provides:

- `sales_summary(df)` for revenue, units, sales-row count, named products sold, and average transaction-line value.
- `sales_by_product(df)` to rank products by revenue with unit and transaction-line totals.
- `sales_trends(df, freq)` for revenue, units, and transaction-line counts by daily, weekly, monthly, quarterly, or yearly period.
- `expense_category_breakdown(df, freq)` for expense amounts grouped by period and category, including Cost of Goods classification and an uncategorized fallback.
- `financial_performance(df, freq)` for revenue, cost of goods, operating expenses, gross/operating/net profit, margins, and period-over-period growth. It flags when operating expense growth exceeds revenue growth by the configured warning gap.

These functions take the validated canonical transaction DataFrame returned by the data layer. Quarterly periods begin in January, April, July, and October; yearly periods begin in January. Growth is undefined when the previous period is zero; margins are undefined when revenue is zero. The warning is informational and does not trigger business actions.

The sample dataset records cost of goods on the sales dates and accrues recurring monthly operating expenses by day, so daily, weekly, monthly, quarterly, and yearly profit views use expenses from matching periods. Imported and manually entered expenses are grouped by their recorded dates; the app does not infer a recurring cadence or spread a user-entered monthly bill across days.

## Demand forecasting

The `biztrend.forecast.forecast_demand(df, periods=3, freq="monthly", product=None)` function forecasts units sold for the whole business or a selected product at daily, weekly, monthly, quarterly, or yearly granularity. It adaptively selects among last-value, three-period moving-average, and linear-trend models using expanding-window, one-step-ahead mean absolute error (MAE). Seasonal naive is also evaluated when at least two full seasonal cycles are available (7 daily, 52 weekly, 12 monthly, or 4 quarterly periods); yearly forecasts compare nonseasonal models.

The result includes forecast values, selected model, validation MAE, history length, and a status. Forecasting requires at least `min_history_points` observations; shorter histories return `insufficient_history` without an unsupported prediction, and a product with no sales returns `no_sales`. Negative unit predictions are clipped to zero. Forecasts are estimates for decision support, not purchasing instructions.

## Inventory intelligence

The `biztrend.inventory.inventory_recommendations(transactions, products)` function combines the adaptive daily unit-demand forecast with product stock, lead time, and unit cost. It returns expected lead-time demand, variability-based safety stock, reorder point, recommended order quantity, estimated cost, forecast model/error, and an explicit status for each product. Reorder quantity fills stock to the reorder point plus one average day of demand, representing a daily review allowance. Invalid or duplicate inventory records raise an error; missing and insufficient sales history are reported without suggesting an order. These are advisory estimates only and do not place orders or commit purchases.

## Anomaly detection

The `biztrend.anomaly` module provides two review queues:

- `detect_transaction_anomalies(transactions)` compares each sale with transactions for the same product and each expense with expenses in the same category, flagging unusual quantity or amount.
- `detect_daily_demand_anomalies(transactions)` compares each product's daily units sold with its preceding active selling days to identify unusual demand patterns.

Both use robust modified z-scores and the configurable `anomaly_zscore_threshold`. Comparison groups need at least `min_history_points` peers; daily demand uses prior active days only, excluding the day under evaluation. An empty result means no records passed the rule with sufficient comparison history, not proof that all activity is normal. Alerts are investigation aids, not findings of fraud or wrongdoing.

## Business assistant

The `biztrend.assistant.generate_business_report(transactions, products=None, freq="monthly", forecast_periods=3)` function combines the preceding analysis modules into source results and a prioritized list of explainable insights. Each insight includes a priority, category, explanation, advisory next step, and supporting evidence. It highlights configured financial warnings, unusual records or demand, forecast changes when a comparable history exists, product sales leaders, and inventory rows needing review. This is a deterministic rules-based assistant, not a generative AI service; it does not execute transactions or place orders. Results depend on available data and configured thresholds, and should be checked against business context.

## Streamlit dashboard

Run `streamlit run app.py` from the project directory. The dashboard includes overview, an executive summary, sales and finance, forecasting, inventory, anomaly review, business insights, chatbot, data management, and a Help & guide page. Sales & finance includes expense category shares and period comparisons; the executive summary highlights the largest expense category for the latest period. The executive summary also presents selected-period KPIs, performance trends, and prioritized evidence-backed insights with advisory actions. The guide is maintained in `HELP.md` and the chatbot can answer app-usage questions from it. Light and dark palettes are defined separately in `.streamlit/config.toml`; coordinated accessible accents, contrasting chart colors, and theme-aware Plotly styling are configured in `app.py`. The sidebar's App controls help section explains the top-right menu's Rerun, Auto rerun, and Clear cache actions. Choose saved local CSV data, read-only sample data, or CSV uploads that stay in the current Streamlit session. Imports and manual entries are validated; saved business CSVs are only changed after an explicit submit action. Data management provides a confirmation-protected action to clear all transactions and products from the selected writable source; type `DELETE` to confirm. Sample demo data remains read-only. Chat history is kept in the current Streamlit browser session and separated by data source and analysis frequency.

## Chatbot

The `biztrend.chatbot.answer_question(question, transactions, products=None, freq="monthly")` API answers supported questions using the existing analysis functions. It supports sales and product performance, expense category breakdowns, latest-period financial comparisons, unit-demand forecasts, stock/reorder estimates, anomaly review, and data-grounded areas to review. Answers include the relevant figures and available history dates.

This is a deterministic first version, not an LLM-powered chatbot. It analyzes the full available transaction history; custom date-range questions, product-level profitability, and revenue/profit forecasts are explicitly unsupported rather than answered with misleading totals or inferred forecasts. Where there is numeric evidence, answers include a simple chart alongside the supporting figures. The chatbot returns clear statuses and fallback text for unsupported or unclear questions, missing history, unknown products, and workload limits. Unexpected processing or display errors are logged privately and shown as one neutral, reassuring reply; technical details are not shown as red alerts. Chat analysis is limited to 100,000 transaction rows; anomaly and combined-recommendation analysis is limited to 20,000 rows, and inventory Q&A is limited to 1,000 products. No answer executes a purchase or financial action.
