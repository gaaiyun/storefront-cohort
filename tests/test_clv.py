"""Tests for CLV estimation."""

from __future__ import annotations

import pandas as pd
import pytest

from storefront_cohort.clv import compute_clv, lifetimes_available
from storefront_cohort.ingest import CUSTOMER_COL


def test_heuristic_runs(normalized_orders):
    res = compute_clv(normalized_orders, method="heuristic")
    assert res.method == "heuristic"
    assert res.table[CUSTOMER_COL].is_unique
    assert "predicted_clv" in res.table.columns


def test_clv_nonnegative(normalized_orders):
    res = compute_clv(normalized_orders, method="heuristic")
    assert (res.table["predicted_clv"] >= 0).all()


def test_one_row_per_customer(normalized_orders):
    res = compute_clv(normalized_orders, method="heuristic")
    assert len(res.table) == normalized_orders[CUSTOMER_COL].nunique()


def test_summary_keys(normalized_orders):
    res = compute_clv(normalized_orders, method="heuristic")
    for key in ("method", "horizon_months", "total_clv", "mean_clv", "median_clv"):
        assert key in res.summary


def test_horizon_scales_value(normalized_orders):
    short = compute_clv(normalized_orders, method="heuristic", horizon_months=3)
    long = compute_clv(normalized_orders, method="heuristic", horizon_months=24)
    # A longer horizon should not produce less total projected value.
    assert long.total_clv >= short.total_clv


def test_recent_buyer_outvalues_lapsed():
    # Two customers with identical spend/frequency but different recency.
    rows = [
        # recent buyer: orders spread up to the observation end
        ("recent", "2024-01-01", 100.0),
        ("recent", "2024-06-01", 100.0),
        ("recent", "2024-12-01", 100.0),
        # lapsed buyer: same orders but all early, none recent
        ("lapsed", "2023-01-01", 100.0),
        ("lapsed", "2023-03-01", 100.0),
        ("lapsed", "2023-05-01", 100.0),
        # anchor so observation_end is late 2024
        ("anchor", "2024-12-15", 10.0),
    ]
    df = pd.DataFrame(rows, columns=["customer_id", "order_date", "amount"])
    res = compute_clv(df, method="heuristic")
    table = res.table.set_index(CUSTOMER_COL)
    assert table.loc["recent", "predicted_clv"] > table.loc["lapsed", "predicted_clv"]


def test_sorted_descending(normalized_orders):
    res = compute_clv(normalized_orders, method="heuristic")
    vals = res.table["predicted_clv"].tolist()
    assert vals == sorted(vals, reverse=True)


def test_force_lifetimes_without_package(normalized_orders):
    if lifetimes_available():
        pytest.skip("lifetimes is installed; cannot test the missing-package path")
    with pytest.raises(RuntimeError, match="lifetimes"):
        compute_clv(normalized_orders, method="lifetimes")


def test_auto_uses_heuristic_when_no_lifetimes(normalized_orders):
    if lifetimes_available():
        pytest.skip("lifetimes installed; auto may pick the model")
    res = compute_clv(normalized_orders, method="auto")
    assert res.method == "heuristic"


def test_unknown_method_raises(normalized_orders):
    with pytest.raises(ValueError, match="Unknown method"):
        compute_clv(normalized_orders, method="bogus")


def test_empty_raises():
    empty = pd.DataFrame(columns=["customer_id", "order_date", "amount"])
    with pytest.raises(ValueError, match="empty"):
        compute_clv(empty)
