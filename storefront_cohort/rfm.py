"""RFM scoring and 11-segment customer classification.

Computes Recency / Frequency / Monetary scores (1..5 by quantile) for every
customer and assigns each to one of 11 standard e-commerce segments
(Champions, Loyal, At Risk, ...). Operates on the normalised order contract
from :mod:`storefront_cohort.ingest`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .ingest import AMOUNT_COL, CUSTOMER_COL, DATE_COL

# Segment rules expressed as inclusive (min, max) ranges on the 1..5 R/F/M
# scores. Rules are evaluated top-to-bottom; the first match wins, so more
# specific / higher-value segments are listed first. A customer matching no
# rule falls through to "Others".
SEGMENT_RULES: Dict[str, Dict[str, Any]] = {
    "Champions": {
        "r": (4, 5), "f": (4, 5), "m": (4, 5),
        "description": "Bought recently, buy often, spend the most.",
        "action": "Reward them; early access to new products; ask for referrals.",
    },
    "Loyal Customers": {
        "r": (3, 5), "f": (3, 5), "m": (3, 5),
        "description": "Spend well and buy regularly.",
        "action": "Upsell higher-value products; loyalty perks.",
    },
    "Potential Loyalists": {
        "r": (4, 5), "f": (1, 3), "m": (3, 5),
        "description": "Recent customers with solid spend.",
        "action": "Membership / loyalty program; keep them engaged.",
    },
    "New Customers": {
        "r": (4, 5), "f": (1, 2), "m": (1, 5),
        "description": "Bought very recently, but only once or twice.",
        "action": "Onboarding flow; support a strong second purchase.",
    },
    "Promising": {
        "r": (3, 4), "f": (1, 2), "m": (1, 2),
        "description": "Recent low spenders.",
        "action": "Build awareness; entry-level offers.",
    },
    "Need Attention": {
        "r": (2, 3), "f": (2, 4), "m": (2, 4),
        "description": "Above-average history but slipping in recency.",
        "action": "Time-limited offers; personalised recommendations.",
    },
    "About To Sleep": {
        "r": (2, 3), "f": (1, 2), "m": (1, 2),
        "description": "Below-average everywhere; nearly dormant.",
        "action": "Reactivation campaign; share valuable resources.",
    },
    # "Cant Lose Them" is a stricter case of "At Risk" (top F/M), so it is
    # listed first to win the first-match assignment.
    "Cant Lose Them": {
        "r": (1, 2), "f": (4, 5), "m": (4, 5),
        "description": "High-value buyers who have gone quiet.",
        "action": "Aggressive win-back; talk to them; new products.",
    },
    "At Risk": {
        "r": (1, 2), "f": (3, 5), "m": (3, 5),
        "description": "Used to spend a lot, but not seen for a while.",
        "action": "Win-back: personalised reconnects, renewals, discounts.",
    },
    # "Lost" is the strictest low-value case (R exactly 1); listed before the
    # broader "Hibernating" so it wins when it applies.
    "Lost": {
        "r": (1, 1), "f": (1, 2), "m": (1, 2),
        "description": "Lowest recency, frequency and spend.",
        "action": "Revive interest with a campaign or ignore.",
    },
    "Hibernating": {
        "r": (1, 2), "f": (1, 2), "m": (1, 2),
        "description": "Low spend, low frequency, long gone.",
        "action": "Low-cost reactivation; or let them go.",
    },
}


@dataclass
class RFMResult:
    """Per-customer RFM table plus segment rollups."""

    table: pd.DataFrame  # one row per customer
    segment_summary: pd.DataFrame  # one row per segment
    reference_date: pd.Timestamp
    n_segments: int

    @property
    def n_customers(self) -> int:
        return len(self.table)


def _score(values: pd.Series, *, ascending: bool, bins: int) -> pd.Series:
    """Bucket ``values`` into 1..bins by quantile.

    ``ascending=True`` means larger value -> larger score (Frequency, Monetary).
    ``ascending=False`` means larger value -> smaller score (Recency: a large
    recency-in-days is *bad*).

    Falls back to a rank-based split when the data has too few distinct values
    for ``qcut`` to produce the requested number of bins.
    """
    mid = (bins + 1) // 2  # neutral score for degenerate inputs

    # Too few rows or a single distinct value: everyone gets the mid score.
    # (qcut/cut both produce all-NaN here, which cannot be cast to int.)
    if len(values) < bins or values.nunique() <= 1:
        return pd.Series(mid, index=values.index, dtype=int)

    # Rank first so duplicate values are spread deterministically across bins.
    ranked = values.rank(method="first")
    try:
        scored = pd.qcut(ranked, q=bins, labels=False, duplicates="drop")
        produced = int(scored.max()) + 1
        if produced < bins:
            raise ValueError("not enough distinct quantiles")
    except (ValueError, IndexError):
        # Degenerate column (e.g. everyone has frequency 1): even rank split.
        scored = pd.cut(ranked, bins=bins, labels=False, include_lowest=True)

    # Any residual gaps (NaN) fall back to the neutral score before casting.
    scored = scored.fillna(mid - 1).astype(int) + 1  # shift 0..bins-1 -> 1..bins
    if not ascending:
        scored = bins + 1 - scored
    return scored.astype(int)


def _assign_segment(r: int, f: int, m: int) -> str:
    for name, rule in SEGMENT_RULES.items():
        if (
            rule["r"][0] <= r <= rule["r"][1]
            and rule["f"][0] <= f <= rule["f"][1]
            and rule["m"][0] <= m <= rule["m"][1]
        ):
            return name
    return "Others"


def compute_rfm(
    orders: pd.DataFrame,
    *,
    reference_date: Optional[datetime] = None,
    n_segments: int = 5,
) -> RFMResult:
    """Compute RFM scores and segments from normalised orders.

    Args:
        orders: frame with ``customer_id``, ``order_date``, ``amount``.
        reference_date: "today" for the recency calculation. Defaults to the
            day after the last order so the most recent buyer has recency >= 1.
        n_segments: number of score buckets per dimension (default 5).

    Returns:
        RFMResult with a per-customer ``table`` carrying columns:
        ``customer_id, recency_days, frequency, monetary, avg_order_value,
        first_order, last_order, r_score, f_score, m_score, rfm_score, segment``.
    """
    if orders.empty:
        raise ValueError("compute_rfm received an empty orders frame.")
    for col in (CUSTOMER_COL, DATE_COL, AMOUNT_COL):
        if col not in orders.columns:
            raise KeyError(f"orders is missing required column '{col}'.")

    df = orders[[CUSTOMER_COL, DATE_COL, AMOUNT_COL]].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    last_order = df[DATE_COL].max()
    if reference_date is None:
        ref = last_order + pd.Timedelta(days=1)
    else:
        ref = pd.Timestamp(reference_date)

    grouped = df.groupby(CUSTOMER_COL).agg(
        first_order=(DATE_COL, "min"),
        last_order=(DATE_COL, "max"),
        frequency=(AMOUNT_COL, "count"),
        monetary=(AMOUNT_COL, "sum"),
    )
    grouped["recency_days"] = (ref - grouped["last_order"]).dt.days.clip(lower=0)
    grouped["avg_order_value"] = grouped["monetary"] / grouped["frequency"]

    bins = n_segments
    grouped["r_score"] = _score(grouped["recency_days"], ascending=False, bins=bins)
    grouped["f_score"] = _score(grouped["frequency"], ascending=True, bins=bins)
    grouped["m_score"] = _score(grouped["monetary"], ascending=True, bins=bins)
    grouped["rfm_score"] = (
        grouped["r_score"] * 100 + grouped["f_score"] * 10 + grouped["m_score"]
    )
    grouped["segment"] = [
        _assign_segment(r, f, m)
        for r, f, m in zip(grouped["r_score"], grouped["f_score"], grouped["m_score"])
    ]

    table = grouped.reset_index()[
        [
            CUSTOMER_COL,
            "recency_days",
            "frequency",
            "monetary",
            "avg_order_value",
            "first_order",
            "last_order",
            "r_score",
            "f_score",
            "m_score",
            "rfm_score",
            "segment",
        ]
    ]
    table = table.round({"monetary": 2, "avg_order_value": 2})

    summary = summarize_segments(table)

    return RFMResult(
        table=table,
        segment_summary=summary,
        reference_date=ref,
        n_segments=n_segments,
    )


def summarize_segments(table: pd.DataFrame) -> pd.DataFrame:
    """Roll a per-customer RFM table up to one row per segment."""
    total_customers = len(table)
    total_revenue = table["monetary"].sum()

    summary = (
        table.groupby("segment")
        .agg(
            customers=(CUSTOMER_COL, "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
        )
        .reset_index()
    )
    summary["pct_customers"] = summary["customers"] / total_customers * 100
    summary["pct_revenue"] = (
        summary["total_revenue"] / total_revenue * 100 if total_revenue else 0.0
    )
    summary = summary.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    return summary.round(
        {
            "avg_recency_days": 1,
            "avg_frequency": 2,
            "avg_monetary": 2,
            "total_revenue": 2,
            "pct_customers": 1,
            "pct_revenue": 1,
        }
    )


def segment_action(segment: str) -> str:
    """Return the recommended action string for a segment name."""
    rule = SEGMENT_RULES.get(segment)
    return rule["action"] if rule else "Review manually."
