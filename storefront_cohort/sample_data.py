"""Generate a realistic sample order CSV so the tool runs offline.

The generated file uses generic column names (``customer_id``, ``order_date``,
``amount``) plus a few extra columns a real export would carry. It contains a
mix of customer types (loyal repeat buyers, one-time shoppers, lapsed
high-value customers, recent new customers) spread across acquisition months
so RFM, cohort and CLV all have something meaningful to chew on.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

_PRODUCTS = [
    ("Ceramic mug", 18.0),
    ("Pour-over kit", 42.0),
    ("Single-origin beans 250g", 16.0),
    ("Cold brew bottle", 28.0),
    ("Gift set", 65.0),
    ("Espresso blend 1kg", 38.0),
    ("Travel tumbler", 24.0),
]


def generate_orders(
    n_customers: int = 900,
    *,
    seed: int = 42,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Generate a synthetic order line-item table.

    Returns a DataFrame with columns: order_id, customer_id, order_date,
    amount, product, quantity.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    end = end_date or datetime(2024, 12, 31)
    history_days = 540  # ~18 months

    # Customer archetypes: (weight, n_orders_lambda, recency_window_days, spend_mult)
    archetypes = {
        "loyal": (0.18, 9.0, (1, 45), 1.3),
        "regular": (0.30, 4.0, (1, 90), 1.0),
        "one_time": (0.27, 1.0, (1, 400), 0.8),
        "lapsed_high": (0.10, 7.0, (200, 460), 1.6),
        "new": (0.15, 2.0, (1, 40), 1.0),
    }

    rows = []
    cust_idx = 0
    order_seq = 0
    for archetype, (weight, lam, recency_window, spend_mult) in archetypes.items():
        n = int(round(n_customers * weight))
        for _ in range(n):
            cust_idx += 1
            cid = f"C{cust_idx:05d}"
            email = f"customer{cust_idx:05d}@example.com"

            n_orders = max(1, int(rng.poisson(lam)))
            # When did they last buy?
            last_gap = random.randint(*recency_window)
            last_date = end - timedelta(days=last_gap)

            # First order: spread acquisition across the history window.
            if archetype == "new":
                first_offset = random.randint(0, 60)
            elif archetype == "lapsed_high":
                first_offset = random.randint(history_days - 120, history_days)
            else:
                first_offset = random.randint(60, history_days)
            first_date = end - timedelta(days=min(first_offset, history_days))
            if first_date > last_date:
                first_date, last_date = last_date, first_date

            # Distribute order dates between first and last.
            if n_orders == 1:
                order_dates = [last_date]
            else:
                total_span = max((last_date - first_date).days, n_orders)
                offsets = sorted(
                    rng.integers(0, total_span + 1, size=n_orders).tolist()
                )
                offsets[0] = 0
                offsets[-1] = (last_date - first_date).days
                order_dates = [first_date + timedelta(days=int(o)) for o in offsets]

            for od in order_dates:
                order_seq += 1
                product, base_price = random.choice(_PRODUCTS)
                qty = 1 + int(rng.geometric(0.6) - 1)
                qty = max(1, min(qty, 5))
                noise = rng.normal(1.0, 0.12)
                amount = round(base_price * qty * spend_mult * max(noise, 0.4), 2)
                rows.append(
                    {
                        "order_id": f"ORD{order_seq:06d}",
                        "customer_id": email,  # emails are a realistic id
                        "order_date": od.strftime("%Y-%m-%d"),
                        "amount": amount,
                        "product": product,
                        "quantity": qty,
                    }
                )

    df = pd.DataFrame(rows).sort_values("order_date").reset_index(drop=True)
    return df


def write_sample(path: str, **kwargs) -> str:
    """Generate orders and write them to ``path``; returns the path."""
    df = generate_orders(**kwargs)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    import os

    here = os.path.dirname(__file__)
    out = os.path.join(here, "data", "sample_orders.csv")
    write_sample(out)
    print(f"Wrote sample orders to {out}")
