"""Actionable customer lists exported as CSV.

Joins the RFM table with the CLV table and produces two ranked lists a store
owner can act on immediately:

* **churn-risk** -- customers who used to be valuable but have gone quiet
  (segments At Risk / Cant Lose Them / About To Sleep / Hibernating, or a long
  recency), ranked by what they are worth (predicted CLV).
* **high-value** -- the current best customers (Champions / Loyal, or top CLV),
  ranked by predicted CLV.

A churn-risk *flag* and a simple churn-risk *score* (0..1) are also attached to
every customer so the full table can be exported too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .clv import CLVResult
from .ingest import CUSTOMER_COL
from .rfm import RFMResult, segment_action

# Segments that indicate a lapsing / lapsed customer worth winning back.
CHURN_SEGMENTS = {
    "At Risk",
    "Cant Lose Them",
    "About To Sleep",
    "Hibernating",
    "Lost",
    "Need Attention",
}

# Segments representing current best customers.
HIGH_VALUE_SEGMENTS = {"Champions", "Loyal Customers", "Potential Loyalists"}


@dataclass
class CustomerLists:
    """Combined per-customer table plus the two ranked action lists."""

    customers: pd.DataFrame  # full joined table with risk flags
    churn_risk: pd.DataFrame
    high_value: pd.DataFrame


def _churn_score(recency_days: pd.Series, r_score: pd.Series, n_segments: int) -> pd.Series:
    """A 0..1 churn-risk score blending recency rank and raw recency.

    Higher = more likely churned. Driven mainly by the inverted recency score
    (R=1 is worst), nudged by absolute days-since-last-order so two customers
    with R=1 are still separated.
    """
    r_component = (n_segments - r_score) / (n_segments - 1)  # 0 (R=max) .. 1 (R=1)
    # Normalise recency days to 0..1 within this dataset.
    rd = recency_days.astype(float)
    span = rd.max() - rd.min()
    rd_component = (rd - rd.min()) / span if span > 0 else rd * 0.0
    score = 0.7 * r_component + 0.3 * rd_component
    return score.clip(0.0, 1.0).round(3)


def build_customer_lists(
    rfm: RFMResult,
    clv: CLVResult,
    *,
    churn_top: Optional[int] = None,
    high_value_top: Optional[int] = None,
) -> CustomerLists:
    """Join RFM + CLV and build churn-risk and high-value lists.

    Args:
        rfm: result of :func:`storefront_cohort.rfm.compute_rfm`.
        clv: result of :func:`storefront_cohort.clv.compute_clv`.
        churn_top: cap the churn-risk list to this many rows (None = all).
        high_value_top: cap the high-value list to this many rows (None = all).
    """
    rfm_cols = [
        CUSTOMER_COL,
        "recency_days",
        "frequency",
        "monetary",
        "avg_order_value",
        "last_order",
        "r_score",
        "f_score",
        "m_score",
        "segment",
    ]
    clv_cols = [CUSTOMER_COL, "predicted_clv"]

    merged = rfm.table[rfm_cols].merge(
        clv.table[clv_cols], on=CUSTOMER_COL, how="left"
    )
    merged["predicted_clv"] = merged["predicted_clv"].fillna(0.0)

    merged["churn_score"] = _churn_score(
        merged["recency_days"], merged["r_score"], rfm.n_segments
    )
    merged["churn_risk"] = merged["segment"].isin(CHURN_SEGMENTS)
    merged["recommended_action"] = merged["segment"].map(segment_action)

    churn = (
        merged[merged["churn_risk"]]
        .sort_values(["predicted_clv", "churn_score"], ascending=[False, False])
        .reset_index(drop=True)
    )
    high = (
        merged[merged["segment"].isin(HIGH_VALUE_SEGMENTS)]
        .sort_values("predicted_clv", ascending=False)
        .reset_index(drop=True)
    )

    if churn_top is not None:
        churn = churn.head(churn_top)
    if high_value_top is not None:
        high = high.head(high_value_top)

    return CustomerLists(customers=merged, churn_risk=churn, high_value=high)


def export_lists(lists: CustomerLists, out_dir: str, *, prefix: str = "") -> dict:
    """Write the action lists (and the full table) to CSV files.

    Returns a dict mapping a logical name to the written path.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    targets = {
        "customers": lists.customers,
        "churn_risk": lists.churn_risk,
        "high_value": lists.high_value,
    }
    for name, frame in targets.items():
        path = os.path.join(out_dir, f"{prefix}{name}.csv")
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths
