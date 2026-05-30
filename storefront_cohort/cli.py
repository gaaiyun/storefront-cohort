"""Command-line interface for storefront-cohort.

Subcommands:
    report    end-to-end: CSV -> HTML report + customer-list CSVs
    segments  print the RFM segment summary to the terminal
    cohort    print the monthly retention cohort table to the terminal
    sample    write the bundled sample order CSV to a path

Run ``storefront-cohort --help`` or ``python -m storefront_cohort --help``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import __version__
from .advisor import build_advice_table  # noqa: F401  (kept importable for users)
from .clv import compute_clv, lifetimes_available
from .cohort import compute_cohorts
from .ingest import IngestError, load_orders, parse_map_arg
from .report import ReportInputs, write_report
from .rfm import compute_rfm
from .sample_data import write_sample
from .segments_export import build_customer_lists, export_lists
from .validate import check_orders


def _bundled_sample_path() -> str:
    return os.path.join(os.path.dirname(__file__), "data", "sample_orders.csv")


def _load(args) -> "object":
    """Shared ingestion for analysis subcommands."""
    mapping = parse_map_arg(args.map) if getattr(args, "map", None) else None
    return load_orders(
        args.input,
        preset=getattr(args, "preset", None),
        mapping=mapping,
        drop_nonpositive=not getattr(args, "keep_refunds", False),
    )


def _print_ingest_banner(res) -> None:
    cols = ", ".join(f"{k}={v}" for k, v in res.column_mapping.items())
    print(
        f"Loaded {res.rows_out:,}/{res.rows_in:,} orders "
        f"({res.n_customers:,} customers) via {res.preset} mapping [{cols}]"
    )
    if res.dropped:
        dropped = ", ".join(f"{k}:{v}" for k, v in res.dropped.items())
        print(f"  dropped rows -> {dropped}")


def cmd_report(args) -> int:
    res = _load(args)
    _print_ingest_banner(res)

    quality = check_orders(res.orders)
    rfm = compute_rfm(res.orders, n_segments=args.segments)
    cohort = compute_cohorts(res.orders)
    clv = compute_clv(
        res.orders,
        horizon_months=args.horizon,
        method=args.clv_method,
    )
    lists = build_customer_lists(rfm, clv)

    os.makedirs(args.outdir, exist_ok=True)
    store_name = args.store or os.path.splitext(os.path.basename(args.input))[0]
    report_path = os.path.join(args.outdir, "report.html")
    inputs = ReportInputs(
        ingest=res,
        quality=quality,
        rfm=rfm,
        cohort=cohort,
        clv=clv,
        lists=lists,
        store_name=store_name,
    )
    write_report(inputs, report_path)
    csv_paths = export_lists(lists, args.outdir)

    print(f"\nReport written to {report_path}")
    print("Customer lists:")
    for name, path in csv_paths.items():
        print(f"  {name:<12} {path}")
    print(
        f"\n{rfm.n_customers:,} customers across "
        f"{rfm.segment_summary['segment'].nunique()} segments. "
        f"CLV via {clv.method}; {len(lists.churn_risk):,} at-risk, "
        f"{len(lists.high_value):,} high-value."
    )
    if quality.warnings:
        print("Warnings:")
        for w in quality.warnings:
            print(f"  - {w.message}")
    return 0


def cmd_segments(args) -> int:
    res = _load(args)
    _print_ingest_banner(res)
    rfm = compute_rfm(res.orders, n_segments=args.segments)
    print()
    print(rfm.segment_summary.to_string(index=False))
    return 0


def cmd_cohort(args) -> int:
    res = _load(args)
    _print_ingest_banner(res)
    cohort = compute_cohorts(res.orders)
    print(f"\n{cohort.n_cohorts} monthly cohorts, up to month +{cohort.max_offset}")
    print("\nRetention (% of cohort active):")
    pct = (cohort.retention * 100).round(0)
    # Limit columns for terminal readability.
    max_cols = min(13, pct.shape[1])
    view = pct.iloc[:, :max_cols].fillna("")
    print(view.to_string())
    return 0


def cmd_sample(args) -> int:
    if args.output:
        path = write_sample(args.output, n_customers=args.customers, seed=args.seed)
        print(f"Wrote sample orders to {path}")
    else:
        # Print the bundled sample's location and a preview.
        path = _bundled_sample_path()
        print(f"Bundled sample: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="storefront-cohort",
        description=(
            "Turn an order CSV into a single-file customer-insight HTML report: "
            "RFM segments, monthly retention cohorts, CLV, and win-back / "
            "high-value customer lists. Runs entirely on your machine."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_input_opts(p, *, with_segments=True):
        p.add_argument("input", help="path to the order CSV")
        p.add_argument(
            "--preset",
            choices=["shopify", "woocommerce", "stripe", "generic"],
            help="force a known platform column mapping (default: auto-detect)",
        )
        p.add_argument(
            "--map",
            help="explicit column map, e.g. 'customer=Email,date=Created at,amount=Total'",
        )
        p.add_argument(
            "--keep-refunds",
            action="store_true",
            help="keep rows with amount <= 0 (default: drop them)",
        )
        if with_segments:
            p.add_argument(
                "--segments",
                type=int,
                default=5,
                help="number of RFM score buckets per dimension (default: 5)",
            )

    # report
    p_report = sub.add_parser(
        "report", help="generate the full HTML report and customer-list CSVs"
    )
    add_input_opts(p_report)
    p_report.add_argument(
        "-o", "--outdir", default="storefront_report", help="output directory"
    )
    p_report.add_argument("--store", help="store name shown in the report header")
    p_report.add_argument(
        "--horizon", type=int, default=12, help="CLV horizon in months (default: 12)"
    )
    p_report.add_argument(
        "--clv-method",
        choices=["auto", "lifetimes", "heuristic"],
        default="auto",
        help="CLV estimator (default: auto)",
    )
    p_report.set_defaults(func=cmd_report)

    # segments
    p_seg = sub.add_parser("segments", help="print the RFM segment summary")
    add_input_opts(p_seg)
    p_seg.set_defaults(func=cmd_segments)

    # cohort
    p_coh = sub.add_parser("cohort", help="print the monthly retention cohort table")
    add_input_opts(p_coh, with_segments=False)
    p_coh.set_defaults(func=cmd_cohort)

    # sample
    p_sample = sub.add_parser("sample", help="write or locate the bundled sample CSV")
    p_sample.add_argument(
        "-o", "--output", help="write a freshly generated sample CSV to this path"
    )
    p_sample.add_argument(
        "--customers", type=int, default=900, help="number of customers to generate"
    )
    p_sample.add_argument("--seed", type=int, default=42, help="random seed")
    p_sample.set_defaults(func=cmd_sample)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc.filename}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
