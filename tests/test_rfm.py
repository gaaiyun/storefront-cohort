"""Tests for RFM scoring and segmentation."""

from __future__ import annotations

import pandas as pd
import pytest

from storefront_cohort.ingest import CUSTOMER_COL
from storefront_cohort.rfm import (
    SEGMENT_RULES,
    _assign_segment,
    compute_rfm,
    segment_action,
    summarize_segments,
)


def test_compute_rfm_columns(normalized_orders):
    res = compute_rfm(normalized_orders)
    expected = {
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
    }
    assert expected.issubset(set(res.table.columns))


def test_one_row_per_customer(normalized_orders):
    res = compute_rfm(normalized_orders)
    assert res.table[CUSTOMER_COL].is_unique
    assert res.n_customers == normalized_orders[CUSTOMER_COL].nunique()


def test_scores_in_range(normalized_orders):
    res = compute_rfm(normalized_orders, n_segments=5)
    for col in ("r_score", "f_score", "m_score"):
        assert res.table[col].min() >= 1
        assert res.table[col].max() <= 5


def test_monetary_matches_raw_sum(tiny_orders):
    res = compute_rfm(tiny_orders)
    table = res.table.set_index(CUSTOMER_COL)
    assert table.loc["A", "monetary"] == pytest.approx(310.0)
    assert table.loc["A", "frequency"] == 3
    assert table.loc["B", "monetary"] == pytest.approx(500.0)
    assert table.loc["C", "frequency"] == 2


def test_recency_uses_reference_date(tiny_orders):
    ref = pd.Timestamp("2024-04-01")
    res = compute_rfm(tiny_orders, reference_date=ref)
    table = res.table.set_index(CUSTOMER_COL)
    # A last ordered 2024-03-15 -> 17 days before 2024-04-01.
    assert table.loc["A", "recency_days"] == 17
    # B last ordered 2024-01-20 -> 72 days.
    assert table.loc["B", "recency_days"] == 72


def test_default_reference_is_day_after_last(tiny_orders):
    res = compute_rfm(tiny_orders)
    # Last order overall is 2024-03-15; ref = +1 day.
    assert res.reference_date == pd.Timestamp("2024-03-16")


def test_recency_nonnegative(normalized_orders):
    res = compute_rfm(normalized_orders)
    assert (res.table["recency_days"] >= 0).all()


def test_rfm_score_composition(normalized_orders):
    res = compute_rfm(normalized_orders)
    row = res.table.iloc[0]
    assert row["rfm_score"] == row["r_score"] * 100 + row["f_score"] * 10 + row["m_score"]


def test_segments_are_known(normalized_orders):
    res = compute_rfm(normalized_orders)
    known = set(SEGMENT_RULES) | {"Others"}
    assert set(res.table["segment"]).issubset(known)


def test_champion_rule():
    assert _assign_segment(5, 5, 5) == "Champions"


def test_at_risk_rule():
    # Low recency, high frequency/monetary -> At Risk.
    # f/m at the top (4-5) is the stricter "Cant Lose Them" case.
    assert _assign_segment(1, 4, 4) == "Cant Lose Them"
    # r=1 rules out "Need Attention" (which needs r>=2), leaving At Risk.
    assert _assign_segment(1, 3, 3) == "At Risk"


def test_lost_rule():
    assert _assign_segment(1, 1, 1) == "Lost"


def test_segment_summary_percentages_sum_100(normalized_orders):
    res = compute_rfm(normalized_orders)
    s = res.segment_summary
    assert s["pct_customers"].sum() == pytest.approx(100.0, abs=0.5)
    assert s["pct_revenue"].sum() == pytest.approx(100.0, abs=0.5)
    assert s["customers"].sum() == res.n_customers


def test_summarize_segments_standalone(tiny_orders):
    res = compute_rfm(tiny_orders)
    again = summarize_segments(res.table)
    assert set(again.columns) == set(res.segment_summary.columns)


def test_segment_action_returns_text():
    assert isinstance(segment_action("Champions"), str)
    assert segment_action("Champions")
    assert segment_action("Nonexistent") == "Review manually."


def test_empty_orders_raises():
    empty = pd.DataFrame(columns=["customer_id", "order_date", "amount"])
    with pytest.raises(ValueError, match="empty"):
        compute_rfm(empty)


def test_missing_column_raises(tiny_orders):
    with pytest.raises(KeyError):
        compute_rfm(tiny_orders.drop(columns=["amount"]))


def test_degenerate_single_value_column():
    # Everyone orders exactly once for the same amount on different days.
    rows = [(f"c{i}", f"2024-01-{i+1:02d}", 50.0) for i in range(10)]
    df = pd.DataFrame(rows, columns=["customer_id", "order_date", "amount"])
    res = compute_rfm(df)  # must not raise despite identical frequency/monetary
    assert res.n_customers == 10
    assert res.table["f_score"].between(1, 5).all()
