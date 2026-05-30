"""Monthly retention cohort analysis.

Groups customers by the calendar month of their first order (the "cohort")
and tracks, for each subsequent month offset, how many of that cohort placed
at least one order. Produces both an absolute count matrix and a percentage
retention matrix suitable for a heatmap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ingest import CUSTOMER_COL, DATE_COL


@dataclass
class CohortResult:
    """Retention cohort matrices."""

    counts: pd.DataFrame  # index=cohort month (YYYY-MM), columns=month offset, values=active customers
    retention: pd.DataFrame  # same shape, values = fraction (0..1) of cohort retained
    cohort_sizes: pd.Series  # cohort month -> size at offset 0

    @property
    def n_cohorts(self) -> int:
        return len(self.counts)

    @property
    def max_offset(self) -> int:
        return int(self.counts.columns.max()) if len(self.counts.columns) else 0


def _month_period(series: pd.Series) -> pd.Series:
    return series.dt.to_period("M")


def compute_cohorts(orders: pd.DataFrame) -> CohortResult:
    """Build monthly acquisition cohorts and their retention curves.

    Args:
        orders: normalised frame with ``customer_id`` and ``order_date``.

    Returns:
        CohortResult. ``retention`` row *c*, column *k* is the share of cohort
        *c*'s customers who ordered in their *k*-th month after acquisition
        (k=0 is the acquisition month itself and is always 1.0).
    """
    if orders.empty:
        raise ValueError("compute_cohorts received an empty orders frame.")
    for col in (CUSTOMER_COL, DATE_COL):
        if col not in orders.columns:
            raise KeyError(f"orders is missing required column '{col}'.")

    df = orders[[CUSTOMER_COL, DATE_COL]].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["order_month"] = _month_period(df[DATE_COL])

    # Each customer's acquisition month = month of their first order.
    cohort_month = df.groupby(CUSTOMER_COL)["order_month"].transform("min")
    df["cohort_month"] = cohort_month

    # Month offset between an order and the customer's acquisition month.
    df["offset"] = (
        (df["order_month"] - df["cohort_month"]).apply(lambda x: x.n)
    )

    # Distinct active customers per (cohort, offset).
    active = (
        df.groupby(["cohort_month", "offset"])[CUSTOMER_COL]
        .nunique()
        .reset_index(name="active")
    )
    counts = active.pivot(index="cohort_month", columns="offset", values="active")
    counts = counts.sort_index()
    # Ensure a contiguous 0..max offset range so the heatmap has no gaps.
    max_off = int(counts.columns.max())
    counts = counts.reindex(columns=range(max_off + 1))

    cohort_sizes = counts[0].astype("Int64")
    retention = counts.div(cohort_sizes, axis=0).astype(float)

    # Pretty string index (YYYY-MM) for display / serialisation.
    counts.index = counts.index.astype(str)
    retention.index = retention.index.astype(str)
    cohort_sizes.index = cohort_sizes.index.astype(str)

    counts.index.name = "cohort"
    retention.index.name = "cohort"
    counts.columns.name = "month_offset"
    retention.columns.name = "month_offset"

    return CohortResult(
        counts=counts,
        retention=retention,
        cohort_sizes=cohort_sizes,
    )


def overall_retention_curve(result: CohortResult) -> pd.Series:
    """Customer-weighted average retention at each month offset across cohorts.

    Weighting by cohort size avoids letting tiny, young cohorts dominate the
    headline curve.
    """
    counts = result.counts
    weighted = counts.multiply(1.0)  # copy as float
    # Average retention at offset k = sum(active_k) / sum(size for cohorts that
    # could reach k). A cohort "could reach k" if it has a non-null entry there.
    numer = counts.sum(axis=0, skipna=True)
    reachable = counts.notna().multiply(result.cohort_sizes.astype(float), axis=0)
    denom = reachable.sum(axis=0, skipna=True)
    curve = (numer / denom).replace([np.inf, -np.inf], np.nan)
    curve.name = "avg_retention"
    return curve
