"""Customer Lifetime Value (CLV) estimation.

Two estimators are available:

* ``lifetimes`` (optional dependency) -- fits BG/NBD + Gamma-Gamma models for
  a probabilistic CLV. Used automatically when the package is installed and
  the data is rich enough.
* A lightweight, dependency-free heuristic that always works: it projects each
  customer's historical monthly value forward over the horizon, scaled by an
  "alive" factor derived from how recently they last purchased relative to
  their own typical inter-purchase gap.

Both return the same column contract so the rest of the pipeline does not care
which ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .ingest import AMOUNT_COL, CUSTOMER_COL, DATE_COL

try:  # optional dependency
    from lifetimes import BetaGeoFitter, GammaGammaFitter  # type: ignore

    _LIFETIMES_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    _LIFETIMES_AVAILABLE = False


@dataclass
class CLVResult:
    """Per-customer predicted CLV plus metadata."""

    table: pd.DataFrame  # customer_id, predicted_clv, plus model inputs
    method: str  # "bgnbd_gamma_gamma" or "heuristic"
    horizon_months: int
    summary: Dict[str, Any]

    @property
    def total_clv(self) -> float:
        return float(self.table["predicted_clv"].sum())


def _summary_company_periods(orders: pd.DataFrame) -> pd.DataFrame:
    """Build the lifetimes-style summary: frequency, recency, T, monetary_value.

    All time units are in days here and converted as needed by callers.
    frequency = number of *repeat* purchase periods (distinct purchase days
    after the first). recency = days between first and last purchase. T = days
    between first purchase and the observation end.
    """
    df = orders[[CUSTOMER_COL, DATE_COL, AMOUNT_COL]].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.normalize()
    observation_end = df[DATE_COL].max()

    grp = df.groupby(CUSTOMER_COL)
    first = grp[DATE_COL].min()
    last = grp[DATE_COL].max()
    distinct_days = grp[DATE_COL].nunique()
    total_value = grp[AMOUNT_COL].sum()
    n_orders = grp[AMOUNT_COL].count()

    summary = pd.DataFrame(
        {
            "frequency": (distinct_days - 1).clip(lower=0).astype(float),
            "recency": (last - first).dt.days.astype(float),
            "T": (observation_end - first).dt.days.astype(float),
            "monetary_value": (total_value / n_orders).astype(float),
            "total_revenue": total_value.astype(float),
            "n_orders": n_orders.astype(int),
            "first_order": first,
            "last_order": last,
        }
    )
    return summary


def _predict_lifetimes(
    summary: pd.DataFrame, horizon_months: int, discount_rate: float
) -> pd.Series:
    """Fit BG/NBD + Gamma-Gamma and return predicted CLV per customer."""
    freq = summary["frequency"].values
    rec = summary["recency"].values
    T = summary["T"].values

    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(freq, rec, T)

    # Gamma-Gamma requires repeat buyers with positive monetary value.
    repeat = summary[(summary["frequency"] > 0) & (summary["monetary_value"] > 0)]
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(repeat["frequency"].values, repeat["monetary_value"].values)

    clv = ggf.customer_lifetime_value(
        bgf,
        summary["frequency"],
        summary["recency"],
        summary["T"],
        summary["monetary_value"],
        time=horizon_months,
        freq="D",
        discount_rate=discount_rate,
    )
    return pd.Series(np.asarray(clv, dtype=float), index=summary.index)


def _predict_heuristic(
    summary: pd.DataFrame, horizon_months: int, discount_rate: float
) -> pd.Series:
    """Dependency-free CLV projection.

    For each customer:
        monthly_value = total_revenue / max(active_months, 1)
        alive_factor  = exp(-recency_gap_ratio)  in (0, 1]
        clv = monthly_value * horizon * alive_factor * discount_factor

    where ``recency_gap_ratio`` compares how long since the last order to the
    customer's own average gap between orders, so dormant customers are
    discounted toward zero and steady buyers keep most of their run-rate.
    """
    # Active span in months (at least the horizon's smallest unit).
    active_days = (summary["T"]).clip(lower=1.0)
    active_months = (active_days / 30.0).clip(lower=1.0)
    monthly_value = summary["total_revenue"] / active_months

    # Days since last order = T - recency.
    days_since_last = (summary["T"] - summary["recency"]).clip(lower=0.0)
    # Typical gap between orders; single-purchase customers get their whole span.
    avg_gap = np.where(
        summary["frequency"] > 0,
        summary["recency"] / summary["frequency"].clip(lower=1),
        active_days,
    )
    avg_gap = np.clip(avg_gap, 1.0, None)
    gap_ratio = days_since_last.values / avg_gap
    alive_factor = np.exp(-gap_ratio)  # 1.0 when just purchased, ->0 when overdue

    # Simple monthly discounting over the horizon.
    monthly_discount = 1.0 / (1.0 + discount_rate)
    if discount_rate > 0:
        discount_factor = (1 - monthly_discount ** horizon_months) / (
            1 - monthly_discount
        )
    else:
        discount_factor = float(horizon_months)

    # monthly_value already represents one month; multiply by discounted months.
    clv = monthly_value.values * alive_factor * discount_factor
    return pd.Series(np.maximum(clv, 0.0), index=summary.index)


def compute_clv(
    orders: pd.DataFrame,
    *,
    horizon_months: int = 12,
    discount_rate: float = 0.01,
    method: str = "auto",
) -> CLVResult:
    """Estimate forward-looking CLV for every customer.

    Args:
        orders: normalised order frame.
        horizon_months: months to project value over.
        discount_rate: monthly discount rate applied to future value.
        method: ``"auto"`` (use lifetimes if available and data is rich enough,
            else heuristic), ``"lifetimes"`` (force, error if unavailable), or
            ``"heuristic"`` (force the built-in model).

    Returns:
        CLVResult whose ``table`` has ``customer_id, predicted_clv,
        avg_order_value, frequency, recency_days, predicted plus model inputs``.
    """
    if orders.empty:
        raise ValueError("compute_clv received an empty orders frame.")

    summary = _summary_company_periods(orders)
    n_repeat = int((summary["frequency"] > 0).sum())

    use_lifetimes = False
    if method == "lifetimes":
        if not _LIFETIMES_AVAILABLE:
            raise RuntimeError(
                "method='lifetimes' requested but the 'lifetimes' package is "
                "not installed. Install it or use method='heuristic'."
            )
        use_lifetimes = True
    elif method == "auto":
        # lifetimes needs a reasonable number of repeat buyers to fit well.
        use_lifetimes = _LIFETIMES_AVAILABLE and n_repeat >= 20
    elif method != "heuristic":
        raise ValueError(f"Unknown method '{method}'.")

    if use_lifetimes:
        try:
            clv = _predict_lifetimes(summary, horizon_months, discount_rate)
            used = "bgnbd_gamma_gamma"
        except Exception:
            # Numerical failure -> fall back rather than crash the report.
            clv = _predict_heuristic(summary, horizon_months, discount_rate)
            used = "heuristic"
    else:
        clv = _predict_heuristic(summary, horizon_months, discount_rate)
        used = "heuristic"

    table = summary.copy()
    table["predicted_clv"] = clv.round(2)
    table = table.reset_index().rename(columns={"index": CUSTOMER_COL})
    table = table.rename(columns={"recency": "recency_span_days"})
    table["recency_days"] = (table["T"] - table["recency_span_days"]).clip(lower=0)

    out = table[
        [
            CUSTOMER_COL,
            "predicted_clv",
            "monetary_value",
            "total_revenue",
            "frequency",
            "n_orders",
            "recency_days",
            "T",
        ]
    ].rename(columns={"monetary_value": "avg_order_value"})
    out = out.sort_values("predicted_clv", ascending=False).reset_index(drop=True)

    clv_values = out["predicted_clv"]
    summary_stats = {
        "method": used,
        "horizon_months": horizon_months,
        "total_clv": round(float(clv_values.sum()), 2),
        "mean_clv": round(float(clv_values.mean()), 2),
        "median_clv": round(float(clv_values.median()), 2),
        "top10pct_clv_share": round(
            float(
                clv_values.sort_values(ascending=False)
                .head(max(1, len(clv_values) // 10))
                .sum()
                / clv_values.sum()
                * 100
            ),
            1,
        )
        if clv_values.sum() > 0
        else 0.0,
    }

    return CLVResult(
        table=out,
        method=used,
        horizon_months=horizon_months,
        summary=summary_stats,
    )


def lifetimes_available() -> bool:
    """Whether the optional ``lifetimes`` dependency is importable."""
    return _LIFETIMES_AVAILABLE
