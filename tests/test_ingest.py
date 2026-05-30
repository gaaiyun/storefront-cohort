"""Tests for adaptive CSV ingestion."""

from __future__ import annotations

import pandas as pd
import pytest

from storefront_cohort.ingest import (
    AMOUNT_COL,
    CUSTOMER_COL,
    DATE_COL,
    IngestError,
    detect_mapping,
    load_orders,
    normalize_orders,
    parse_map_arg,
)


def _write_csv(tmp_path, name, frame):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return str(path)


def test_detect_shopify_headers():
    cols = ["Name", "Email", "Financial Status", "Created at", "Total"]
    mapping = detect_mapping(cols)
    assert mapping[CUSTOMER_COL] == "Email"
    assert mapping[DATE_COL] == "Created at"
    assert mapping[AMOUNT_COL] == "Total"


def test_detect_generic_headers():
    mapping = detect_mapping(["customer_id", "order_date", "amount", "product"])
    assert mapping == {
        CUSTOMER_COL: "customer_id",
        DATE_COL: "order_date",
        AMOUNT_COL: "amount",
    }


def test_detect_does_not_reuse_same_column():
    # "amount" should claim AMOUNT_COL, not be double-assigned.
    mapping = detect_mapping(["email", "date", "amount"])
    assert len(set(mapping.values())) == 3


def test_load_auto_detect(tmp_path):
    frame = pd.DataFrame(
        {
            "Email": ["a@x.com", "b@x.com", "a@x.com"],
            "Created at": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "Total": ["$10.00", "20", "30.5"],
        }
    )
    path = _write_csv(tmp_path, "shop.csv", frame)
    res = load_orders(path)
    assert res.preset == "auto"
    assert res.rows_out == 3
    assert res.n_customers == 2
    # Currency symbol stripped and coerced to float.
    assert res.orders[AMOUNT_COL].tolist() == [10.0, 20.0, 30.5]
    assert str(res.orders[DATE_COL].dtype).startswith("datetime64")


def test_load_with_preset(tmp_path):
    frame = pd.DataFrame(
        {
            "Customer Email": ["a@x.com"],
            "Created (UTC)": ["2024-01-01 10:00:00"],
            "Amount": [12.0],
            "extra": ["ignored"],
        }
    )
    path = _write_csv(tmp_path, "stripe.csv", frame)
    res = load_orders(path, preset="stripe")
    assert res.preset == "stripe"
    assert res.rows_out == 1


def test_load_with_explicit_mapping(tmp_path):
    frame = pd.DataFrame(
        {"who": ["a"], "when": ["2024-01-01"], "how_much": [9.0]}
    )
    path = _write_csv(tmp_path, "weird.csv", frame)
    mapping = {CUSTOMER_COL: "who", DATE_COL: "when", AMOUNT_COL: "how_much"}
    res = load_orders(path, mapping=mapping)
    assert res.preset == "custom"
    assert res.rows_out == 1


def test_unmappable_columns_raise(tmp_path):
    frame = pd.DataFrame({"foo": [1], "bar": [2]})
    path = _write_csv(tmp_path, "bad.csv", frame)
    with pytest.raises(IngestError, match="Could not map"):
        load_orders(path)


def test_unknown_preset_raises(tmp_path):
    frame = pd.DataFrame({"a": [1]})
    path = _write_csv(tmp_path, "x.csv", frame)
    with pytest.raises(IngestError, match="Unknown preset"):
        load_orders(path, preset="magento")


def test_normalize_drops_bad_rows():
    frame = pd.DataFrame(
        {
            "customer_id": ["a", "b", "", "d", "e"],
            "order_date": ["2024-01-01", "not-a-date", "2024-01-03", "2024-01-04", "2024-01-05"],
            "amount": [10.0, 20.0, 30.0, -5.0, None],
        }
    )
    mapping = {CUSTOMER_COL: "customer_id", DATE_COL: "order_date", AMOUNT_COL: "amount"}
    out, dropped = normalize_orders(frame, mapping)
    # row a kept; b bad date; c blank customer; d nonpositive; e nan amount
    assert out[CUSTOMER_COL].tolist() == ["a"]
    assert dropped["unparseable_date"] == 1
    assert dropped["blank_customer"] == 1
    assert dropped["nonpositive_amount"] == 1
    assert dropped["unparseable_amount"] == 1


def test_keep_refunds_keeps_nonpositive():
    frame = pd.DataFrame(
        {
            "customer_id": ["a", "b"],
            "order_date": ["2024-01-01", "2024-01-02"],
            "amount": [10.0, -5.0],
        }
    )
    mapping = {CUSTOMER_COL: "customer_id", DATE_COL: "order_date", AMOUNT_COL: "amount"}
    out, dropped = normalize_orders(frame, mapping, drop_nonpositive=False)
    assert len(out) == 2
    assert "nonpositive_amount" not in dropped


def test_all_rows_dropped_raises(tmp_path):
    frame = pd.DataFrame(
        {"customer_id": ["", ""], "order_date": ["x", "y"], "amount": [1, 2]}
    )
    path = _write_csv(tmp_path, "empty.csv", frame)
    with pytest.raises(IngestError, match="No valid order rows"):
        load_orders(path)


def test_tz_aware_dates_made_naive(tmp_path):
    frame = pd.DataFrame(
        {
            "customer_id": ["a"],
            "order_date": ["2024-01-01T10:00:00+02:00"],
            "amount": [10.0],
        }
    )
    path = _write_csv(tmp_path, "tz.csv", frame)
    res = load_orders(path)
    assert res.orders[DATE_COL].dt.tz is None


def test_parse_map_arg():
    mapping = parse_map_arg("customer=Email,date=Created at,amount=Total")
    assert mapping == {
        CUSTOMER_COL: "Email",
        DATE_COL: "Created at",
        AMOUNT_COL: "Total",
    }


def test_parse_map_arg_bad_key():
    with pytest.raises(IngestError, match="Unknown --map key"):
        parse_map_arg("foo=bar")


def test_parse_map_arg_bad_format():
    with pytest.raises(IngestError, match="Bad --map entry"):
        parse_map_arg("customer")
