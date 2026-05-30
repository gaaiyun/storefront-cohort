"""Tests for HTML report assembly and the validate module."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from storefront_cohort.clv import compute_clv
from storefront_cohort.cohort import compute_cohorts
from storefront_cohort.ingest import normalize_orders
from storefront_cohort.report import ReportInputs, render_html, write_report
from storefront_cohort.rfm import compute_rfm
from storefront_cohort.segments_export import build_customer_lists
from storefront_cohort.validate import check_orders


@pytest.fixture()
def report_inputs(normalized_orders):
    rfm = compute_rfm(normalized_orders)
    return ReportInputs(
        ingest=_FakeIngest(normalized_orders),
        quality=check_orders(normalized_orders),
        rfm=rfm,
        cohort=compute_cohorts(normalized_orders),
        clv=compute_clv(normalized_orders, method="heuristic"),
        lists=build_customer_lists(rfm, compute_clv(normalized_orders, method="heuristic")),
        store_name="Test Store",
    )


class _FakeIngest:
    """Minimal stand-in for IngestResult for the report."""

    def __init__(self, orders):
        self.orders = orders
        self.preset = "auto"
        self.rows_in = len(orders)
        self.rows_out = len(orders)
        self.column_mapping = {}


def test_render_returns_html(report_inputs):
    html = render_html(report_inputs)
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")


def test_report_is_self_contained(report_inputs):
    html = render_html(report_inputs)
    # No external network assets -> works offline.
    assert "http://" not in html
    assert "https://" not in html
    assert "cdn." not in html
    assert "<script" not in html  # no JS dependency at all


def test_report_contains_all_sections(report_inputs):
    html = render_html(report_inputs)
    for heading in (
        "Overview",
        "RFM segments",
        "Monthly retention cohorts",
        "Customer lifetime value",
        "Customer action lists",
        "Marketing playbook",
    ):
        assert f">{heading}</h2>" in html


def test_report_mentions_store_name(report_inputs):
    html = render_html(report_inputs)
    assert "Test Store" in html


def test_report_has_cohort_cells(report_inputs):
    html = render_html(report_inputs)
    assert "class='cohort'" in html


def test_report_escapes_html(normalized_orders):
    # A store name with markup must be escaped.
    rfm = compute_rfm(normalized_orders)
    clv = compute_clv(normalized_orders, method="heuristic")
    inputs = ReportInputs(
        ingest=_FakeIngest(normalized_orders),
        quality=check_orders(normalized_orders),
        rfm=rfm,
        cohort=compute_cohorts(normalized_orders),
        clv=clv,
        lists=build_customer_lists(rfm, clv),
        store_name="<script>alert(1)</script>",
    )
    html = render_html(inputs)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_write_report_creates_file(report_inputs, tmp_path):
    path = str(tmp_path / "report.html")
    written = write_report(report_inputs, path)
    assert written == path
    assert os.path.exists(path)
    assert os.path.getsize(path) > 1000


# --- validate module ---


def test_check_orders_basic(normalized_orders):
    report = check_orders(normalized_orders)
    assert report.n_rows == len(normalized_orders)
    assert report.total_revenue > 0
    assert report.span_days > 0


def test_future_date_warning():
    future = (pd.Timestamp.now().normalize() + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    rows = [("a", "2024-01-01", 10.0), ("b", future, 20.0)]
    df = pd.DataFrame(rows, columns=["customer_id", "order_date", "amount"])
    df, _ = normalize_orders(df, {"customer_id": "customer_id", "order_date": "order_date", "amount": "amount"})
    report = check_orders(df)
    codes = [f.code for f in report.findings]
    assert "future_dates" in codes
    assert any(w.code == "future_dates" for w in report.warnings)


def test_short_history_warning():
    rows = [("a", "2024-01-01", 10.0), ("b", "2024-01-15", 20.0)]
    df = pd.DataFrame(rows, columns=["customer_id", "order_date", "amount"])
    report = check_orders(df)
    assert any(f.code == "short_history" for f in report.findings)


def test_duplicate_detection():
    rows = [("a", "2024-01-01", 10.0), ("a", "2024-01-01", 10.0), ("b", "2024-03-01", 5.0)]
    df = pd.DataFrame(rows, columns=["customer_id", "order_date", "amount"])
    report = check_orders(df)
    assert any(f.code == "duplicate_rows" for f in report.findings)
