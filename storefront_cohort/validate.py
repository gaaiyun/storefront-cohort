"""Lightweight data-quality checks on a normalised order frame.

Runs after ingestion to surface issues a store owner should know about before
trusting the report: duplicate order rows, future-dated orders, single-order
customers, very short history for cohorts, etc. Returns structured findings;
nothing here raises -- ingestion already enforced the hard contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from .ingest import AMOUNT_COL, CUSTOMER_COL, DATE_COL


@dataclass
class Finding:
    level: str  # "info" | "warning"
    code: str
    message: str


@dataclass
class QualityReport:
    n_rows: int
    n_customers: int
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    total_revenue: float
    findings: List[Finding] = field(default_factory=list)

    @property
    def span_days(self) -> int:
        return int((self.date_max - self.date_min).days)

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.level == "warning"]


def check_orders(orders: pd.DataFrame) -> QualityReport:
    """Inspect a normalised order frame and return a QualityReport."""
    df = orders
    dates = pd.to_datetime(df[DATE_COL])
    report = QualityReport(
        n_rows=len(df),
        n_customers=int(df[CUSTOMER_COL].nunique()),
        date_min=dates.min(),
        date_max=dates.max(),
        total_revenue=round(float(df[AMOUNT_COL].sum()), 2),
    )

    dup = df.duplicated(subset=[CUSTOMER_COL, DATE_COL, AMOUNT_COL]).sum()
    if dup:
        report.findings.append(
            Finding(
                "info",
                "duplicate_rows",
                f"{dup} order row(s) are exact duplicates "
                "(same customer, date and amount).",
            )
        )

    now = pd.Timestamp.now().normalize()
    future = (dates > now).sum()
    if future:
        report.findings.append(
            Finding(
                "warning",
                "future_dates",
                f"{future} order(s) are dated in the future; check the export.",
            )
        )

    span_months = (report.date_max.to_period("M") - report.date_min.to_period("M")).n
    if span_months < 2:
        report.findings.append(
            Finding(
                "warning",
                "short_history",
                "Order history spans fewer than 2 months; the retention "
                "cohort will be very small.",
            )
        )

    order_counts = df.groupby(CUSTOMER_COL)[AMOUNT_COL].count()
    one_timers = int((order_counts == 1).sum())
    if report.n_customers and one_timers / report.n_customers > 0.8:
        report.findings.append(
            Finding(
                "info",
                "mostly_one_time",
                f"{one_timers}/{report.n_customers} customers ordered only "
                "once; repeat-purchase metrics will be limited.",
            )
        )

    return report
