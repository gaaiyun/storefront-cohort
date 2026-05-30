"""storefront-cohort: local customer-insight reports from an order CSV.

A single command turns a Shopify / WooCommerce / Stripe / generic order export
into a self-contained HTML report -- RFM segments, monthly retention cohorts,
customer lifetime value, and ready-to-use win-back / high-value customer lists.
Everything runs locally; no data leaves the machine.

Public surface:
    load_orders         adaptive CSV ingestion -> normalised order frame
    compute_rfm         RFM scoring + 11-segment classification
    compute_cohorts     monthly retention cohorts
    compute_clv         lifetime-value estimation (lifetimes or heuristic)
    build_customer_lists  churn-risk / high-value lists
    write_report        single-file HTML report
"""

from __future__ import annotations

__version__ = "0.3.0"

from .clv import CLVResult, compute_clv
from .cohort import CohortResult, compute_cohorts
from .ingest import IngestError, IngestResult, load_orders
from .rfm import RFMResult, compute_rfm
from .segments_export import CustomerLists, build_customer_lists
from .validate import QualityReport, check_orders

__all__ = [
    "__version__",
    "load_orders",
    "IngestResult",
    "IngestError",
    "compute_rfm",
    "RFMResult",
    "compute_cohorts",
    "CohortResult",
    "compute_clv",
    "CLVResult",
    "build_customer_lists",
    "CustomerLists",
    "check_orders",
    "QualityReport",
]
