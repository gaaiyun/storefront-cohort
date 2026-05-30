"""Per-segment marketing playbook.

A static, opinionated library mapping each RFM segment to a concrete strategy,
a handful of tactics, and the channels that tend to work. This is intentionally
rule-based (not a model): it gives a store owner a sensible starting playbook
next to each segment in the report.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .rfm import RFMResult

SEGMENT_PLAYBOOK: Dict[str, Dict[str, object]] = {
    "Champions": {
        "strategy": "Reward and retain",
        "tactics": [
            "Early access to new products",
            "VIP perks and a referral program",
            "Personalised thank-you offers",
        ],
        "channels": ["Email", "SMS"],
    },
    "Loyal Customers": {
        "strategy": "Grow order value",
        "tactics": [
            "Bundle deals and cross-sell",
            "Free shipping thresholds",
            "Subscribe-and-save offers",
        ],
        "channels": ["Email", "App push"],
    },
    "Potential Loyalists": {
        "strategy": "Build the habit",
        "tactics": [
            "Loyalty / membership enrolment",
            "Recommended-for-you emails",
            "Limited-time member offers",
        ],
        "channels": ["Email", "Social"],
    },
    "New Customers": {
        "strategy": "Onboard and convert second order",
        "tactics": [
            "Welcome email series",
            "Post-purchase follow-up discount",
            "Product education content",
        ],
        "channels": ["Email", "SMS"],
    },
    "Promising": {
        "strategy": "Nurture awareness",
        "tactics": [
            "Entry-level offers",
            "Browse-abandonment reminders",
            "Social proof and reviews",
        ],
        "channels": ["Email", "Retargeting"],
    },
    "Need Attention": {
        "strategy": "Re-engage before they lapse",
        "tactics": [
            "Time-limited comeback offer",
            "Personalised recommendations",
            "New-arrival alerts",
        ],
        "channels": ["Email", "SMS"],
    },
    "About To Sleep": {
        "strategy": "Reactivate",
        "tactics": [
            "We-miss-you campaign",
            "Reminder of unused benefits",
            "Best-seller highlights",
        ],
        "channels": ["Email"],
    },
    "At Risk": {
        "strategy": "Win back",
        "tactics": [
            "Strong personalised discount",
            "One-to-one outreach",
            "Satisfaction survey and fix issues",
        ],
        "channels": ["Email", "SMS", "Phone"],
    },
    "Cant Lose Them": {
        "strategy": "Aggressive win-back",
        "tactics": [
            "Best available offer",
            "Personal outreach from the team",
            "No-questions returns guarantee",
        ],
        "channels": ["Phone", "Email"],
    },
    "Hibernating": {
        "strategy": "Low-cost reactivation",
        "tactics": [
            "Clearance / sale notifications",
            "Free sample or trial",
            "Member-day invitation",
        ],
        "channels": ["Email"],
    },
    "Lost": {
        "strategy": "Minimal-cost maintenance",
        "tactics": [
            "Holiday greetings",
            "Newsletter only",
            "Re-opt-in invitation",
        ],
        "channels": ["Email"],
    },
    "Others": {
        "strategy": "Review manually",
        "tactics": ["Inspect the RFM scores and decide case by case"],
        "channels": ["Email"],
    },
}


def playbook_for(segment: str) -> Dict[str, object]:
    """Return the playbook entry for a segment (falls back to 'Others')."""
    return SEGMENT_PLAYBOOK.get(segment, SEGMENT_PLAYBOOK["Others"])


def build_advice_table(rfm: RFMResult) -> pd.DataFrame:
    """Build a per-segment advice table ordered by revenue contribution.

    Columns: segment, customers, pct_revenue, strategy, tactics, channels.
    Only segments that actually appear in the data are included.
    """
    rows: List[Dict[str, object]] = []
    summary = rfm.segment_summary.set_index("segment")
    for segment in summary.index:
        play = playbook_for(segment)
        rows.append(
            {
                "segment": segment,
                "customers": int(summary.loc[segment, "customers"]),
                "pct_revenue": float(summary.loc[segment, "pct_revenue"]),
                "strategy": play["strategy"],
                "tactics": "; ".join(play["tactics"]),  # type: ignore[arg-type]
                "channels": ", ".join(play["channels"]),  # type: ignore[arg-type]
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values("pct_revenue", ascending=False).reset_index(drop=True)
