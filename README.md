# storefront-cohort

Turn an order CSV into a single-file customer-insight report — RFM segments, monthly retention cohorts, customer lifetime value, and ready-to-use win-back / high-value customer lists. One command, runs entirely on your machine, nothing uploaded.

It is an open-source local alternative to the paid "customer insights" apps for Shopify / WooCommerce / Stripe. You export your orders, point the tool at the CSV, and get an HTML report plus action-list CSVs you can hand straight to your email tool.

## What you get

- **RFM segmentation** — every customer scored 1–5 on Recency, Frequency and Monetary value, then grouped into 11 standard segments (Champions, Loyal, At Risk, Cant Lose Them, …).
- **Monthly retention cohorts** — a heatmap of how many customers from each acquisition month keep ordering in later months.
- **Customer lifetime value** — projected value per customer over a configurable horizon (BG/NBD + Gamma-Gamma if the optional `lifetimes` package is installed, otherwise a built-in heuristic).
- **Action lists** — a win-back list (your most valuable at-risk customers) and a best-customers list, exported as CSV with a suggested next step per row.
- **A single self-contained HTML file** — inline styles and charts, no external assets, opens offline.

## 30-second start

```bash
pip install pandas numpy

# Generate the bundled sample, or use your own export
python -m storefront_cohort sample -o orders.csv

# Build the report
python -m storefront_cohort report orders.csv -o my_report

# Open my_report/report.html in a browser
```

That writes `report.html` plus `churn_risk.csv`, `high_value.csv` and `customers.csv` into `my_report/`.

If you installed the package (`pip install -e .`), the same command is available as `storefront-cohort report orders.csv`.

## Using your own data

The importer auto-detects the customer, date and amount columns from common exports. If auto-detection picks the wrong columns, force a platform preset or map columns yourself:

```bash
# Force a known platform layout
storefront-cohort report orders.csv --preset shopify
storefront-cohort report orders.csv --preset woocommerce
storefront-cohort report orders.csv --preset stripe

# Or map columns explicitly
storefront-cohort report orders.csv --map "customer=Email,date=Created at,amount=Total"
```

Each order line is one row. Refunds / non-positive amounts are dropped by default (`--keep-refunds` to keep them). Rows with an unparseable date, blank customer, or unparseable amount are dropped and reported.

## Example output

`storefront-cohort segments` on the bundled sample (900 customers, 3,794 orders):

```
            segment  customers  avg_recency_days  avg_frequency  avg_monetary  total_revenue  pct_customers  pct_revenue
          Champions        139              17.2           8.60        582.45       80960.01           15.4         33.1
    Loyal Customers        209              35.9           5.21        319.82       66843.19           23.2         27.3
     Cant Lose Them         89             290.2           7.58        641.47       57090.45            9.9         23.3
     Need Attention         86              86.6           3.28        170.79       14687.81            9.6          6.0
      New Customers         94              16.9           1.24         50.33        4731.31           10.4          1.9
            At Risk         17             253.9           3.88        218.63        3716.64            1.9          1.5
```

`storefront-cohort cohort` prints the monthly retention matrix (percent of each cohort still ordering):

```
month_offset     0     1     2     3     4     5     6     7     8
cohort
2023-07       100.0  16.0  21.0  32.0  42.0  47.0  32.0  29.0  26.0
2023-08       100.0  32.0  43.0  45.0  32.0  41.0  36.0  25.0  30.0
2023-09       100.0  38.0  45.0  33.0  28.0  33.0  33.0  30.0  36.0
```

`storefront-cohort report` prints a summary and writes the files:

```
Loaded 3,794/3,794 orders (900 customers) via auto mapping [...]

Report written to my_report/report.html
Customer lists:
  customers    my_report/customers.csv
  churn_risk   my_report/churn_risk.csv
  high_value   my_report/high_value.csv

900 customers across 11 segments. CLV via heuristic; 353 at-risk, 362 high-value.
```

The `churn_risk.csv` columns: `customer_id, recency_days, frequency, monetary, avg_order_value, last_order, r_score, f_score, m_score, segment, predicted_clv, churn_score, churn_risk, recommended_action`.

## Commands

| Command | What it does |
| --- | --- |
| `report INPUT -o DIR` | Full HTML report + the three customer-list CSVs |
| `segments INPUT` | Print the RFM segment summary table |
| `cohort INPUT` | Print the monthly retention cohort table |
| `sample [-o FILE]` | Write a generated sample order CSV (or show the bundled one) |

Useful `report` options: `--store NAME` (report title), `--horizon N` (CLV months, default 12), `--clv-method {auto,lifetimes,heuristic}`, `--segments N` (score buckets per dimension, default 5). Run `--help` on any subcommand for the full list.

## CLV: heuristic vs. lifetimes

By default CLV uses a dependency-free heuristic: each customer's historical monthly run-rate projected over the horizon, scaled down for how overdue their next order is relative to their own buying rhythm. It needs no extra packages and always runs.

For a probabilistic model, install the optional dependency:

```bash
pip install lifetimes
```

With `lifetimes` present and enough repeat buyers, `--clv-method auto` fits BG/NBD + Gamma-Gamma models instead. The report states which method was used.

## Advanced modules (optional)

`storefront_cohort/advanced/` holds heavier, scikit-learn / SHAP based modules — supervised churn prediction, customer clustering, and SHAP explainability. They are **not** part of the core report pipeline and are not imported unless you ask for them, so a missing ML dependency never breaks the main tool. To use them:

```bash
pip install scikit-learn shap
python -c "from storefront_cohort.advanced import clustering_engine"
```

These are work-in-progress and not yet wired into the CLI (see roadmap).

## Install for development

```bash
pip install -e ".[dev]"
pytest -q
```

The test suite covers ingestion, RFM scoring and segmentation, cohort math, CLV, list building, the HTML report, and every CLI subcommand.

## Roadmap

The MVP above is built and tested. Planned next:

- Wire the advanced churn / clustering modules into the CLI as opt-in subcommands.
- PDF / Excel export of the report (the data is already structured for it).
- Per-segment revenue trend over time.
- A small config file to pin column mappings per store.

## How it works

```
order CSV ──> ingest (detect & normalise columns) ──> validate
                                                         │
        ┌────────────────────────────────────────────────┤
        ▼                ▼                  ▼              ▼
      RFM            cohorts              CLV         action lists
        └────────────────┴──────────────────┴──────────────┘
                                  │
                          single-file HTML report
```

The internal contract between modules is just three columns — `customer_id`, `order_date`, `amount` — which is why adapting a new platform export is only a column-mapping change.

## License

MIT. See [LICENSE](LICENSE).
