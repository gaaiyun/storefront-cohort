"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from storefront_cohort.sample_data import generate_orders


@pytest.fixture(scope="session")
def sample_orders_df() -> pd.DataFrame:
    """A deterministic, generated order line-item frame (raw columns)."""
    return generate_orders(n_customers=300, seed=7)


@pytest.fixture()
def normalized_orders(sample_orders_df):
    """Sample orders mapped to the internal contract."""
    from storefront_cohort.ingest import normalize_orders

    mapping = {
        "customer_id": "customer_id",
        "order_date": "order_date",
        "amount": "amount",
    }
    out, _dropped = normalize_orders(sample_orders_df, mapping)
    return out


@pytest.fixture()
def tiny_orders() -> pd.DataFrame:
    """A hand-built order frame with a known, checkable structure.

    Customer A: 3 orders across 3 months (loyal, recent).
    Customer B: 1 old order (lapsed).
    Customer C: 2 orders, one recent.
    """
    rows = [
        ("A", "2024-01-15", 100.0),
        ("A", "2024-02-15", 120.0),
        ("A", "2024-03-15", 90.0),
        ("B", "2024-01-20", 500.0),
        ("C", "2024-01-25", 40.0),
        ("C", "2024-03-10", 60.0),
    ]
    return pd.DataFrame(rows, columns=["customer_id", "order_date", "amount"])
