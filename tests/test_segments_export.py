"""Tests for customer-list building and CSV export."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from storefront_cohort.clv import compute_clv
from storefront_cohort.ingest import CUSTOMER_COL
from storefront_cohort.rfm import compute_rfm
from storefront_cohort.segments_export import (
    CHURN_SEGMENTS,
    HIGH_VALUE_SEGMENTS,
    build_customer_lists,
    export_lists,
)


@pytest.fixture()
def lists(normalized_orders):
    rfm = compute_rfm(normalized_orders)
    clv = compute_clv(normalized_orders, method="heuristic")
    return build_customer_lists(rfm, clv)


def test_every_customer_present(lists, normalized_orders):
    assert len(lists.customers) == normalized_orders[CUSTOMER_COL].nunique()
    assert lists.customers[CUSTOMER_COL].is_unique


def test_churn_list_only_churn_segments(lists):
    assert set(lists.churn_risk["segment"]).issubset(CHURN_SEGMENTS)


def test_high_value_list_only_high_segments(lists):
    assert set(lists.high_value["segment"]).issubset(HIGH_VALUE_SEGMENTS)


def test_churn_score_range(lists):
    assert lists.customers["churn_score"].between(0, 1).all()


def test_churn_flag_consistent(lists):
    flagged = lists.customers[lists.customers["churn_risk"]]
    assert set(flagged["segment"]).issubset(CHURN_SEGMENTS)


def test_predicted_clv_joined(lists):
    assert "predicted_clv" in lists.customers.columns
    assert lists.customers["predicted_clv"].notna().all()


def test_recommended_action_present(lists):
    assert "recommended_action" in lists.customers.columns
    assert lists.customers["recommended_action"].notna().all()


def test_churn_sorted_by_value(lists):
    vals = lists.churn_risk["predicted_clv"].tolist()
    assert vals == sorted(vals, reverse=True)


def test_top_caps(normalized_orders):
    rfm = compute_rfm(normalized_orders)
    clv = compute_clv(normalized_orders, method="heuristic")
    capped = build_customer_lists(rfm, clv, churn_top=5, high_value_top=3)
    assert len(capped.churn_risk) <= 5
    assert len(capped.high_value) <= 3


def test_export_writes_three_csvs(lists, tmp_path):
    paths = export_lists(lists, str(tmp_path))
    assert set(paths) == {"customers", "churn_risk", "high_value"}
    for path in paths.values():
        assert os.path.exists(path)
        # Re-read to confirm valid CSV.
        df = pd.read_csv(path)
        assert CUSTOMER_COL in df.columns


def test_export_prefix(lists, tmp_path):
    paths = export_lists(lists, str(tmp_path), prefix="store1_")
    assert os.path.basename(paths["churn_risk"]) == "store1_churn_risk.csv"
