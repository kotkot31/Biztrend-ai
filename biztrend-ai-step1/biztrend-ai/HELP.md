# BizTrend AI Help Guide

BizTrend AI helps you review sales, expenses, demand, inventory, and unusual activity. It analyzes the records you select and provides estimates and decision support; it does not place orders or execute financial transactions.

## Getting started

1. In the sidebar, choose a **Data source**.
2. Choose an **Analysis period**: daily, weekly, monthly, quarterly, or yearly.
3. Use **Navigate** to open a dashboard page.
4. Add or import records in **Data management**, or start with **Sample demo** to explore the dashboard.
5. Review the charts, tables, evidence, and data coverage before acting on an insight.

The sidebar selections apply across the dashboard. The overview's revenue and units sold are all-time totals; its net profit card refers to the latest selected period.

## Data sources and uploads

- **Saved business data** reads and writes local `data/transactions.csv` and `data/products.csv` files. Submitted imports and entries are saved there.
- **Sample demo** is read-only example data. It is useful for exploring features without changing your business records.
- **CSV upload (session only)** keeps uploaded data in the current Streamlit session. It does not overwrite the saved CSV files; re-upload after the session ends if needed.

In **Data management**, use **Import CSV** to review validation results before submitting an import, or **Add transaction** to enter a sale or expense. Sales require a product name. For amounts, enter a total amount or provide quantity and unit price so the amount can be calculated. Duplicate records are not added.

Use the **Manage products** tab to add product inventory details: product name, unit cost, stock on hand, and supplier lead time in days. Product records are needed for inventory recommendations.

### CSV columns

Transactions should include a date and either an amount or both quantity and unit price. The usual columns are:

`date, record_type, product, category, quantity, unit_price, amount, notes`

Use `sale` or `expense` for `record_type`. Product inventory CSVs use:

`product, unit_cost, stock_on_hand, lead_time_days`

Column names have common aliases, such as `qty` for quantity, `item` for product, and `type` for record type. Invalid rows are reported during import.

## Analysis period

The sidebar's **Analysis period** controls the time grouping used in sales and financial summaries, forecasts, and related chatbot answers:

- **Daily** groups records by calendar day.
- **Weekly** groups records into weeks ending Sunday.
- **Monthly** groups records by calendar month.
- **Quarterly** uses calendar quarters starting in January, April, July, and October.
- **Yearly** uses calendar years starting in January.

Period-based profit compares revenue and expenses recorded in the same period. The app does not automatically spread a manually entered bill across other days or months.

## Dashboard pages

- **Overview:** all-time revenue and units sold, latest-period net profit, products tracked, revenue trend, top products, and latest-period financial details. Hover over the overview cards for their date coverage and supporting details.
- **Executive summary:** selected-period revenue, profit, margin, revenue change, inventory review count, the largest expense category, and a short list of prioritized insights with recommendations and supporting evidence. Forecasts and recommendations are advisory and depend on available data.
- **Sales & finance:** sales and unit trends, product performance, a profit-and-loss view, and an Expense analysis tab. Expense analysis shows category amounts and shares, separates Cost of Goods from operating expenses, and compares categories with the previous period. Blank categories appear as “Uncategorized”; percentage changes are not shown when the prior amount is zero.
- **Forecasting:** choose all products or one product and forecast between 1 and 24 future periods. The page shows predicted units, the selected model, validation error (MAE), and available history.
- **Inventory:** review stock, lead-time demand, reorder points, and estimated replenishment quantities for products in the catalog.
- **Anomaly review:** inspect unusual transaction amounts or quantities and unusual daily product demand.
- **Business assistant:** review evidence-backed, rule-based business insights and suggested next steps.
- **Chatbot:** ask about business data, or ask how to use BizTrend AI. For example: “Which products generated the most revenue?”, “Review expense categories”, “How did profit change in the latest periods?”, or “How do I upload a CSV?”
- **Data management:** import CSVs, add transactions, manage products, and clear test data from the selected writable source.
- **Help & guide:** read this guide inside the app.

## Forecasting

Forecasts estimate future **units sold**, not future revenue or profit. The app compares candidate models against historical data and reports its selected model and validation MAE. At least 8 history periods are required to produce a forecast; shorter histories or products without sales do not receive a prediction. A forecast is an estimate for planning, not a purchasing instruction.

## Inventory

Inventory recommendations use recorded stock, product unit cost, supplier lead time, and estimated demand. Missing product details or insufficient sales history can prevent a recommendation. Review the inputs and recommendation in context; the app never places an order or commits a purchase.

## Anomaly review

Anomaly flags identify records that differ from historical patterns enough to merit review. They are investigation aids, not proof of fraud or wrongdoing. An empty alert list means no records met the configured threshold and history requirements; it does not prove that all activity is normal.

## Business assistant and chatbot

The **Business assistant** is deterministic and rule-based: it summarizes supported analysis results and presents evidence and suggested review actions. The **Chatbot** answers supported questions using the selected data and analysis period, and can also explain app features using this guide. Neither feature is a general-purpose generative AI service, and neither executes business actions.

If a question is outside the chatbot's supported topics, try asking about sales, products, expenses, profit, demand forecasts, inventory, unusual activity, recommendations, or how to use a dashboard feature.

## Data management and safe deletion

The **Delete all test data** control is in **Data management**. First select the data source you intend to clear. The control deletes all transactions and products from that selected writable source:

- For **Saved business data**, it clears the saved transaction and product CSVs.
- For **CSV upload (session only)**, it clears the uploaded-session records and products.
- **Sample demo** is read-only and cannot be cleared.

Type `DELETE` exactly to confirm. Deletion is permanent and cannot be undone. Check the selected source and record counts before confirming.

## Themes and app controls

BizTrend AI supports light and dark Streamlit themes. Change the theme in Streamlit's settings if your deployment exposes that option.

The top-right three-dot menu contains:

- **Rerun:** reruns the app with current data and selections.
- **Auto rerun:** automatically reruns the app when source code changes; it is mainly useful during development.
- **Clear cache:** removes cached data and resources. The next run may take longer while caches are rebuilt. This does not delete saved transaction or product CSV files.

## Tips and limitations

- Check the selected data source and analysis period when comparing results.
- Check imported-row validation messages and correct missing or invalid data.
- Expenses are grouped by their recorded dates; recurring expense timing is not inferred.
- Forecasts, anomaly flags, and reorder quantities are estimates for human review.
- The app is designed for one business and batch analysis of historical records, not real-time processing or automatic purchasing.
