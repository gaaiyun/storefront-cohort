"""Adaptive order-CSV ingestion.

Reads an export from Shopify, WooCommerce, Stripe, or a generic store and
normalises it to a single internal contract used by every analysis module:

    customer_id : str    -- stable customer identifier
    order_date  : datetime64[ns]
    amount      : float   -- order/line total in the store currency

The detector works on column *names* (case/punctuation insensitive). A named
preset can force a mapping; otherwise columns are auto-detected by matching
against per-field synonym lists. One row is treated as one order line; lines
are not merged, so an order split across several rows simply contributes its
parts (which is what RFM/CLV want anyway).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

# Internal column contract shared across the whole package.
CUSTOMER_COL = "customer_id"
DATE_COL = "order_date"
AMOUNT_COL = "amount"


class IngestError(ValueError):
    """Raised when an order CSV cannot be mapped to the internal contract."""


# Explicit column maps for known platform exports. Keys are the *target*
# internal names; values are the source column names in that platform's CSV.
PRESETS: Dict[str, Dict[str, str]] = {
    "shopify": {
        CUSTOMER_COL: "Email",
        DATE_COL: "Created at",
        AMOUNT_COL: "Total",
    },
    "woocommerce": {
        CUSTOMER_COL: "Customer ID",
        DATE_COL: "Order Date",
        AMOUNT_COL: "Order Total",
    },
    "stripe": {
        CUSTOMER_COL: "Customer Email",
        DATE_COL: "Created (UTC)",
        AMOUNT_COL: "Amount",
    },
    "generic": {
        CUSTOMER_COL: "customer_id",
        DATE_COL: "order_date",
        AMOUNT_COL: "amount",
    },
}

# Synonyms used for auto-detection when no preset is given. Order matters:
# earlier entries win. Matching is done on a normalised form of the header
# (lowercased, non-alphanumeric stripped).
_SYNONYMS: Dict[str, List[str]] = {
    CUSTOMER_COL: [
        "customerid",
        "customer",
        "customeremail",
        "email",
        "billingemail",
        "clientid",
        "userid",
        "buyer",
        "buyeremail",
        "customername",
    ],
    DATE_COL: [
        "orderdate",
        "createdat",
        "createdutc",
        "created",
        "date",
        "transactiondate",
        "purchasedate",
        "paidat",
        "processedat",
        "timestamp",
    ],
    AMOUNT_COL: [
        "amount",
        "total",
        "ordertotal",
        "totalprice",
        "grandtotal",
        "netamount",
        "revenue",
        "subtotal",
        "lineamount",
        "price",
        "paid",
    ],
}


@dataclass
class IngestResult:
    """Outcome of :func:`load_orders`."""

    orders: pd.DataFrame  # normalised: customer_id, order_date, amount
    column_mapping: Dict[str, str]  # internal name -> source column
    preset: str  # preset name used, or "auto"
    rows_in: int
    rows_out: int
    dropped: Dict[str, int] = field(default_factory=dict)

    @property
    def n_customers(self) -> int:
        return int(self.orders[CUSTOMER_COL].nunique())


def _normalise(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def detect_mapping(columns: List[str]) -> Dict[str, str]:
    """Auto-detect a column mapping from a list of source headers.

    Returns a dict ``{internal_name: source_column}`` containing only the
    fields that could be confidently matched.
    """
    norm_to_original: Dict[str, str] = {}
    for col in columns:
        norm = _normalise(col)
        # First header wins for a given normalised key.
        norm_to_original.setdefault(norm, col)

    mapping: Dict[str, str] = {}
    used: set[str] = set()
    for target, synonyms in _SYNONYMS.items():
        for syn in synonyms:
            if syn in norm_to_original and norm_to_original[syn] not in used:
                mapping[target] = norm_to_original[syn]
                used.add(norm_to_original[syn])
                break
    return mapping


def _coerce_amount(series: pd.Series) -> pd.Series:
    """Parse a money column that may carry currency symbols / thousands separators."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = (
        series.astype(str)
        .str.replace(r"[^\d.\-]", "", regex=True)
        .replace("", pd.NA)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_orders(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    *,
    drop_nonpositive: bool = True,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Apply ``mapping`` to ``df`` and clean it into the internal contract.

    Returns the normalised frame plus a dict of how many rows were dropped
    for each reason.
    """
    # Real exports carry mixed/odd date formats; let pandas infer per-element
    # and quietly coerce failures to NaT (we drop those rows below). The
    # "could not infer format" warning is expected here, so silence it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed_dates = pd.to_datetime(
            df[mapping[DATE_COL]], errors="coerce", utc=False
        )
    out = pd.DataFrame(
        {
            CUSTOMER_COL: df[mapping[CUSTOMER_COL]].astype(str).str.strip(),
            DATE_COL: parsed_dates,
            AMOUNT_COL: _coerce_amount(df[mapping[AMOUNT_COL]]),
        }
    )
    # Make tz-aware timestamps naive so date arithmetic stays simple.
    if isinstance(out[DATE_COL].dtype, pd.DatetimeTZDtype):
        out[DATE_COL] = out[DATE_COL].dt.tz_localize(None)

    dropped: Dict[str, int] = {}
    before = len(out)

    blank_cust = out[CUSTOMER_COL].isin(["", "nan", "None", "NaN"])
    if blank_cust.any():
        dropped["blank_customer"] = int(blank_cust.sum())
        out = out[~blank_cust]

    bad_date = out[DATE_COL].isna()
    if bad_date.any():
        dropped["unparseable_date"] = int(bad_date.sum())
        out = out[~bad_date]

    bad_amount = out[AMOUNT_COL].isna()
    if bad_amount.any():
        dropped["unparseable_amount"] = int(bad_amount.sum())
        out = out[~bad_amount]

    if drop_nonpositive:
        nonpos = out[AMOUNT_COL] <= 0
        if nonpos.any():
            dropped["nonpositive_amount"] = int(nonpos.sum())
            out = out[~nonpos]

    out = out.reset_index(drop=True)
    assert len(out) <= before
    return out, dropped


def load_orders(
    path: str,
    *,
    preset: Optional[str] = None,
    mapping: Optional[Dict[str, str]] = None,
    drop_nonpositive: bool = True,
    read_csv_kwargs: Optional[dict] = None,
) -> IngestResult:
    """Load and normalise an order CSV.

    Args:
        path: CSV file path.
        preset: one of ``PRESETS`` (``shopify``/``woocommerce``/``stripe``/
            ``generic``). If ``None`` the mapping is auto-detected.
        mapping: explicit ``{internal: source}`` map that overrides everything.
        drop_nonpositive: drop rows whose amount is <= 0 (refunds/zero lines).
        read_csv_kwargs: extra kwargs passed to ``pandas.read_csv``.

    Raises:
        IngestError: if required columns cannot be located or no rows survive.
    """
    raw = pd.read_csv(path, **(read_csv_kwargs or {}))
    rows_in = len(raw)

    if mapping is not None:
        used_mapping = dict(mapping)
        preset_name = "custom"
    elif preset is not None:
        key = preset.lower()
        if key not in PRESETS:
            raise IngestError(
                f"Unknown preset '{preset}'. Choose from: {', '.join(sorted(PRESETS))}."
            )
        used_mapping = dict(PRESETS[key])
        preset_name = key
    else:
        used_mapping = detect_mapping(list(raw.columns))
        preset_name = "auto"

    missing = [
        field_name
        for field_name in (CUSTOMER_COL, DATE_COL, AMOUNT_COL)
        if field_name not in used_mapping
    ]
    if missing:
        raise IngestError(
            "Could not map required field(s) "
            f"{missing} from columns {list(raw.columns)}. "
            "Pass --preset or --map customer=<col>,date=<col>,amount=<col>."
        )

    src_missing = [c for c in used_mapping.values() if c not in raw.columns]
    if src_missing:
        raise IngestError(
            f"Mapped source column(s) {src_missing} are not present in the CSV "
            f"(have: {list(raw.columns)})."
        )

    orders, dropped = normalize_orders(
        raw, used_mapping, drop_nonpositive=drop_nonpositive
    )

    if orders.empty:
        raise IngestError(
            "No valid order rows after cleaning. "
            f"Read {rows_in} rows; dropped {dropped}."
        )

    return IngestResult(
        orders=orders,
        column_mapping=used_mapping,
        preset=preset_name,
        rows_in=rows_in,
        rows_out=len(orders),
        dropped=dropped,
    )


def parse_map_arg(arg: str) -> Dict[str, str]:
    """Parse a ``--map`` CLI string like ``customer=Email,date=Created at,amount=Total``."""
    aliases = {
        "customer": CUSTOMER_COL,
        "customer_id": CUSTOMER_COL,
        "date": DATE_COL,
        "order_date": DATE_COL,
        "amount": AMOUNT_COL,
    }
    mapping: Dict[str, str] = {}
    for piece in arg.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise IngestError(
                f"Bad --map entry '{piece}'. Use key=column (e.g. customer=Email)."
            )
        key, col = piece.split("=", 1)
        key = key.strip().lower()
        if key not in aliases:
            raise IngestError(
                f"Unknown --map key '{key}'. Use customer / date / amount."
            )
        mapping[aliases[key]] = col.strip()
    return mapping
