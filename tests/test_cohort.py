"""Tests for monthly retention cohort analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from storefront_cohort.cohort import compute_cohorts, overall_retention_curve


def test_offset_zero_is_full_retention(normalized_orders):
    res = compute_cohorts(normalized_orders)
    # Every cohort retains 100% of itself at offset 0.
    col0 = res.retention[0].dropna()
    assert col0.tolist() == pytest.approx([1.0] * len(col0))


def test_counts_and_retention_same_shape(normalized_orders):
    res = compute_cohorts(normalized_orders)
    assert res.counts.shape == res.retention.shape


def test_tiny_cohort_structure(tiny_orders):
    res = compute_cohorts(tiny_orders)
    # Acquisition months: A,B,C all first ordered in 2024-01 -> one cohort.
    assert res.n_cohorts == 1
    cohort = res.counts.index[0]
    assert cohort == "2024-01"
    # Cohort size = 3 customers (A, B, C).
    assert int(res.cohort_sizes.iloc[0]) == 3
    # Offset 0: all 3 active. Offset 1 (Feb): only A. Offset 2 (Mar): A and C.
    assert res.counts.loc["2024-01", 0] == 3
    assert res.counts.loc["2024-01", 1] == 1
    assert res.counts.loc["2024-01", 2] == 2


def test_tiny_cohort_retention_values(tiny_orders):
    res = compute_cohorts(tiny_orders)
    assert res.retention.loc["2024-01", 0] == pytest.approx(1.0)
    assert res.retention.loc["2024-01", 1] == pytest.approx(1 / 3)
    assert res.retention.loc["2024-01", 2] == pytest.approx(2 / 3)


def test_max_offset(tiny_orders):
    res = compute_cohorts(tiny_orders)
    assert res.max_offset == 2


def test_contiguous_offset_columns(normalized_orders):
    res = compute_cohorts(normalized_orders)
    cols = list(res.counts.columns)
    assert cols == list(range(0, max(cols) + 1))


def test_index_is_year_month_strings(normalized_orders):
    res = compute_cohorts(normalized_orders)
    for label in res.counts.index:
        assert len(label) == 7 and label[4] == "-"


def test_retention_between_0_and_1(normalized_orders):
    res = compute_cohorts(normalized_orders)
    vals = res.retention.values
    finite = vals[~np.isnan(vals)]
    assert (finite >= 0).all() and (finite <= 1).all()


def test_overall_retention_curve(tiny_orders):
    res = compute_cohorts(tiny_orders)
    curve = overall_retention_curve(res)
    assert curve[0] == pytest.approx(1.0)
    # Only one cohort, so curve mirrors that cohort's retention.
    assert curve[1] == pytest.approx(1 / 3)
    assert curve[2] == pytest.approx(2 / 3)


def test_empty_orders_raises():
    empty = pd.DataFrame(columns=["customer_id", "order_date"])
    with pytest.raises(ValueError, match="empty"):
        compute_cohorts(empty)


def test_multi_cohort_counts():
    # Two acquisition months.
    rows = [
        ("a", "2024-01-05", 10.0),
        ("a", "2024-02-05", 10.0),
        ("b", "2024-02-10", 10.0),  # acquired in Feb
        ("b", "2024-03-10", 10.0),
    ]
    df = pd.DataFrame(rows, columns=["customer_id", "order_date", "amount"])
    res = compute_cohorts(df)
    assert res.n_cohorts == 2
    assert res.counts.loc["2024-01", 0] == 1
    assert res.counts.loc["2024-01", 1] == 1  # a active in Feb
    assert res.counts.loc["2024-02", 0] == 1
    assert res.counts.loc["2024-02", 1] == 1  # b active in Mar
