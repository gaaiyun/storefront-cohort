"""Single-file, offline HTML report assembly.

Everything (styles, charts, tables) is inlined into one ``.html`` file with no
external assets or CDNs, so the report opens correctly even with no network.
Charts are hand-rendered as compact inline SVG / coloured HTML tables rather
than pulling in a JS plotting library -- this keeps the file small and the
dependency surface limited to pandas + numpy.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from .advisor import build_advice_table
from .clv import CLVResult
from .cohort import CohortResult, overall_retention_curve
from .ingest import CUSTOMER_COL, IngestResult
from .rfm import RFMResult
from .segments_export import CustomerLists
from .validate import QualityReport

_CSS = """
:root{
  --bg:#0f1115;--panel:#171a21;--ink:#e8eaed;--muted:#9aa3b2;
  --line:#272b34;--accent:#4f8cff;--good:#3ecf8e;--warn:#f5a623;--bad:#ef5350;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.5;font-size:15px}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 64px}
header h1{font-size:1.9rem;margin:0 0 4px}
header .sub{color:var(--muted);margin:0 0 8px}
.meta{color:var(--muted);font-size:.85rem}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:20px 22px;margin:22px 0}
section h2{margin:0 0 14px;font-size:1.25rem}
section h3{margin:18px 0 8px;font-size:1rem;color:var(--muted);font-weight:600;
  text-transform:uppercase;letter-spacing:.04em}
p.lead{color:var(--muted);margin-top:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin:6px 0 4px}
.kpi{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:14px}
.kpi .v{font-size:1.5rem;font-weight:700}
.kpi .l{color:var(--muted);font-size:.8rem;margin-top:2px}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--muted);font-weight:600;font-size:.8rem;text-transform:uppercase;letter-spacing:.03em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr:hover td{background:rgba(255,255,255,.02)}
.bar{height:9px;border-radius:5px;background:var(--accent);display:inline-block;vertical-align:middle}
.barwrap{display:flex;align-items:center;gap:8px}
.cohort{border-collapse:separate;border-spacing:2px;font-size:.78rem}
.cohort td{padding:6px 8px;border:none;border-radius:4px;text-align:center;color:#0b0d10;font-weight:600}
.cohort th{padding:4px 6px;border:none;font-size:.72rem}
.cohort td.empty{background:var(--bg);color:var(--muted);font-weight:400}
.cohort td.label{background:transparent;color:var(--ink);font-weight:600;text-align:left}
.spark{display:block}
.pill{display:inline-block;padding:2px 8px;border-radius:99px;font-size:.75rem;font-weight:600}
.pill.good{background:rgba(62,207,142,.15);color:var(--good)}
.pill.warn{background:rgba(245,166,35,.15);color:var(--warn)}
.note{color:var(--muted);font-size:.85rem}
.footer{color:var(--muted);font-size:.8rem;text-align:center;margin-top:28px}
ul.tact{margin:4px 0 0;padding-left:18px;color:var(--muted);font-size:.85rem}
"""


@dataclass
class ReportInputs:
    ingest: IngestResult
    quality: QualityReport
    rfm: RFMResult
    cohort: CohortResult
    clv: CLVResult
    lists: CustomerLists
    store_name: str = "Your store"


def _esc(value) -> str:
    return html.escape(str(value))


def _fmt_money(x: float) -> str:
    return f"{x:,.0f}"


def _heat_color(frac: float) -> str:
    """Map a 0..1 retention fraction to a blue gradient hex colour."""
    if frac is None or (isinstance(frac, float) and np.isnan(frac)):
        return "#171a21"
    frac = max(0.0, min(1.0, float(frac)))
    # Interpolate light->dark blue.
    light = (210, 226, 255)
    dark = (37, 99, 235)
    rgb = tuple(int(light[i] + (dark[i] - light[i]) * frac) for i in range(3))
    return "#%02x%02x%02x" % rgb


def _kpi(value: str, label: str) -> str:
    return f'<div class="kpi"><div class="v">{_esc(value)}</div><div class="l">{_esc(label)}</div></div>'


def _df_table(
    df: pd.DataFrame,
    columns: List[tuple],
    *,
    num_cols: Optional[set] = None,
) -> str:
    """Render a DataFrame as an HTML table.

    ``columns`` is a list of (source_col, header_label) pairs.
    ``num_cols`` are source columns to right-align as numbers.
    """
    num_cols = num_cols or set()
    head = "".join(
        f'<th class="{"num" if src in num_cols else ""}">{_esc(label)}</th>'
        for src, label in columns
    )
    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for src, _label in columns:
            cls = "num" if src in num_cols else ""
            cells.append(f'<td class="{cls}">{_esc(row[src])}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )


def _segment_table(rfm: RFMResult) -> str:
    s = rfm.segment_summary.copy()
    max_rev = s["total_revenue"].max() or 1
    rows = []
    for _, r in s.iterrows():
        bar_w = int(round(r["total_revenue"] / max_rev * 120))
        bar = (
            f'<div class="barwrap"><span class="bar" style="width:{bar_w}px"></span>'
            f'<span>{r["pct_revenue"]:.1f}%</span></div>'
        )
        rows.append(
            "<tr>"
            f'<td>{_esc(r["segment"])}</td>'
            f'<td class="num">{int(r["customers"])}</td>'
            f'<td class="num">{r["pct_customers"]:.1f}%</td>'
            f'<td class="num">{r["avg_recency_days"]:.0f}d</td>'
            f'<td class="num">{r["avg_frequency"]:.1f}</td>'
            f'<td class="num">{_fmt_money(r["avg_monetary"])}</td>'
            f"<td>{bar}</td>"
            "</tr>"
        )
    head = (
        "<tr><th>Segment</th><th class='num'>Customers</th><th class='num'>% Cust.</th>"
        "<th class='num'>Avg recency</th><th class='num'>Avg orders</th>"
        "<th class='num'>Avg spend</th><th>Revenue share</th></tr>"
    )
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _cohort_heatmap(cohort: CohortResult, max_offset: int = 12) -> str:
    ret = cohort.retention
    offsets = [c for c in ret.columns if c <= max_offset]
    header = (
        "<th class='label'>Cohort</th><th>Size</th>"
        + "".join(f"<th>M{o}</th>" for o in offsets)
    )
    rows = []
    for cohort_label, series in ret.iterrows():
        size = cohort.cohort_sizes.get(cohort_label, "")
        cells = [
            f"<td class='label'>{_esc(cohort_label)}</td>",
            f"<td class='label'>{_esc(size)}</td>",
        ]
        for o in offsets:
            val = series.get(o, np.nan)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                cells.append("<td class='empty'>·</td>")
            else:
                color = _heat_color(val)
                cells.append(
                    f"<td style='background:{color}'>{val*100:.0f}%</td>"
                )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table class='cohort'><thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _retention_sparkline(cohort: CohortResult, max_offset: int = 12) -> str:
    curve = overall_retention_curve(cohort)
    pts = [(o, curve.get(o, np.nan)) for o in range(0, max_offset + 1)]
    pts = [(o, v) for o, v in pts if not (isinstance(v, float) and np.isnan(v))]
    if len(pts) < 2:
        return ""
    w, h, pad = 460, 110, 24
    xs = [p[0] for p in pts]
    x_min, x_max = min(xs), max(xs)
    sx = lambda o: pad + (o - x_min) / (x_max - x_min) * (w - 2 * pad)
    sy = lambda v: pad + (1 - v) * (h - 2 * pad)
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{sx(o):.1f},{sy(v):.1f}"
        for i, (o, v) in enumerate(pts)
    )
    dots = "".join(
        f"<circle cx='{sx(o):.1f}' cy='{sy(v):.1f}' r='3' fill='#4f8cff'/>"
        for o, v in pts
    )
    labels = (
        f"<text x='{pad}' y='14' fill='#9aa3b2' font-size='11'>100%</text>"
        f"<text x='{pad}' y='{h-6}' fill='#9aa3b2' font-size='11'>0%</text>"
    )
    return (
        f"<svg class='spark' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"
        f"<line x1='{pad}' y1='{sy(0)}' x2='{w-pad}' y2='{sy(0)}' stroke='#272b34'/>"
        f"<path d='{path}' fill='none' stroke='#4f8cff' stroke-width='2'/>"
        f"{dots}{labels}</svg>"
    )


def _clv_histogram(clv: CLVResult, bins: int = 12) -> str:
    values = clv.table["predicted_clv"].values
    values = values[values > 0]
    if len(values) == 0:
        return "<p class='note'>No positive CLV values to chart.</p>"
    counts, edges = np.histogram(values, bins=bins)
    max_c = counts.max() or 1
    w, h, pad = 460, 140, 22
    bw = (w - 2 * pad) / len(counts)
    bars = []
    for i, c in enumerate(counts):
        bh = (c / max_c) * (h - 2 * pad)
        x = pad + i * bw
        y = h - pad - bh
        bars.append(
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{bw-2:.1f}' height='{bh:.1f}' "
            f"fill='#4f8cff' rx='2'><title>{int(c)} customers, "
            f"{edges[i]:.0f}-{edges[i+1]:.0f}</title></rect>"
        )
    axis = (
        f"<text x='{pad}' y='{h-4}' fill='#9aa3b2' font-size='11'>{edges[0]:.0f}</text>"
        f"<text x='{w-pad-30}' y='{h-4}' fill='#9aa3b2' font-size='11'>{edges[-1]:.0f}</text>"
    )
    return (
        f"<svg class='spark' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"
        f"{''.join(bars)}{axis}</svg>"
    )


def _lists_section(lists: CustomerLists, top: int = 15) -> str:
    cols = [
        (CUSTOMER_COL, "Customer"),
        ("segment", "Segment"),
        ("monetary", "Spent"),
        ("frequency", "Orders"),
        ("recency_days", "Days since"),
        ("predicted_clv", "Pred. CLV"),
    ]
    nums = {"monetary", "frequency", "recency_days", "predicted_clv"}
    churn = lists.churn_risk.head(top).copy()
    high = lists.high_value.head(top).copy()
    for f in (churn, high):
        if not f.empty:
            f["monetary"] = f["monetary"].map(_fmt_money)
            f["predicted_clv"] = f["predicted_clv"].map(_fmt_money)
    churn_tbl = (
        _df_table(churn, cols, num_cols=nums)
        if not churn.empty
        else "<p class='note'>No at-risk customers detected.</p>"
    )
    high_tbl = (
        _df_table(high, cols, num_cols=nums)
        if not high.empty
        else "<p class='note'>No high-value customers detected.</p>"
    )
    return (
        f"<h3>Win-back priority &mdash; {len(lists.churn_risk)} at-risk customers "
        f"(top {min(top,len(lists.churn_risk))} by value)</h3>{churn_tbl}"
        f"<h3>Best customers &mdash; {len(lists.high_value)} high-value customers "
        f"(top {min(top,len(lists.high_value))})</h3>{high_tbl}"
    )


def _advice_section(rfm: RFMResult) -> str:
    advice = build_advice_table(rfm).head(8)
    rows = []
    for _, r in advice.iterrows():
        tactics = "".join(
            f"<li>{_esc(t.strip())}</li>" for t in str(r["tactics"]).split(";")
        )
        rows.append(
            "<tr>"
            f"<td>{_esc(r['segment'])}</td>"
            f"<td>{_esc(r['strategy'])}<ul class='tact'>{tactics}</ul></td>"
            f"<td>{_esc(r['channels'])}</td>"
            f"<td class='num'>{r['pct_revenue']:.1f}%</td>"
            "</tr>"
        )
    head = (
        "<tr><th>Segment</th><th>Strategy &amp; tactics</th>"
        "<th>Channels</th><th class='num'>Revenue</th></tr>"
    )
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def render_html(inputs: ReportInputs) -> str:
    """Render the full report as a single self-contained HTML string."""
    ing = inputs.ingest
    q = inputs.quality
    rfm = inputs.rfm
    clv = inputs.clv
    lists = inputs.lists

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    aov = q.total_revenue / q.n_rows if q.n_rows else 0
    repeat_rate = (
        (rfm.table["frequency"] > 1).mean() * 100 if rfm.n_customers else 0
    )

    warnings_html = ""
    if q.findings:
        items = "".join(
            f"<li><span class='pill {'warn' if f.level=='warning' else 'good'}'>"
            f"{_esc(f.level)}</span> {_esc(f.message)}</li>"
            for f in q.findings
        )
        warnings_html = f"<h3>Data notes</h3><ul class='note'>{items}</ul>"

    clv_method = (
        "BG/NBD + Gamma-Gamma (lifetimes)"
        if clv.method == "bgnbd_gamma_gamma"
        else "built-in heuristic"
    )

    parts = []
    parts.append(f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    parts.append(
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
    )
    parts.append(f"<title>Customer insights &mdash; {_esc(inputs.store_name)}</title>")
    parts.append(f"<style>{_CSS}</style></head><body><div class='wrap'>")

    # Header
    parts.append(
        f"<header><h1>Customer insights</h1>"
        f"<p class='sub'>{_esc(inputs.store_name)}</p>"
        f"<p class='meta'>Generated {generated} &middot; "
        f"{q.n_rows:,} orders &middot; {q.n_customers:,} customers &middot; "
        f"{q.date_min.date()} to {q.date_max.date()} &middot; "
        f"source: {_esc(ing.preset)}</p></header>"
    )

    # Overview KPIs
    parts.append("<section><h2>Overview</h2>")
    parts.append("<div class='kpis'>")
    parts.append(_kpi(f"{q.n_customers:,}", "Customers"))
    parts.append(_kpi(_fmt_money(q.total_revenue), "Total revenue"))
    parts.append(_kpi(_fmt_money(aov), "Avg order value"))
    parts.append(_kpi(f"{repeat_rate:.0f}%", "Repeat customers"))
    parts.append(_kpi(_fmt_money(clv.summary["total_clv"]), f"Predicted {clv.horizon_months}m CLV"))
    parts.append("</div>")
    parts.append(warnings_html)
    parts.append("</section>")

    # RFM segments
    parts.append(
        "<section><h2>RFM segments</h2>"
        "<p class='lead'>Customers scored 1-5 on Recency, Frequency and Monetary "
        "value, then grouped into standard segments.</p>"
    )
    parts.append(_segment_table(rfm))
    parts.append("</section>")

    # Cohort retention
    parts.append(
        "<section><h2>Monthly retention cohorts</h2>"
        "<p class='lead'>Each row is the group of customers acquired in that "
        "month; cells show the share who ordered again in later months.</p>"
    )
    parts.append(_retention_sparkline(inputs.cohort))
    parts.append("<h3>Cohort detail</h3>")
    parts.append(_cohort_heatmap(inputs.cohort))
    parts.append("</section>")

    # CLV
    parts.append(
        f"<section><h2>Customer lifetime value</h2>"
        f"<p class='lead'>Projected value over the next {clv.horizon_months} "
        f"months ({_esc(clv_method)}).</p>"
    )
    parts.append("<div class='kpis'>")
    parts.append(_kpi(_fmt_money(clv.summary["mean_clv"]), "Mean CLV"))
    parts.append(_kpi(_fmt_money(clv.summary["median_clv"]), "Median CLV"))
    parts.append(_kpi(f"{clv.summary['top10pct_clv_share']:.0f}%", "From top 10% of customers"))
    parts.append("</div>")
    parts.append(_clv_histogram(clv))
    parts.append("</section>")

    # Action lists
    parts.append("<section><h2>Customer action lists</h2>")
    parts.append(_lists_section(lists))
    parts.append(
        "<p class='note'>Full lists are exported as CSV alongside this report.</p>"
    )
    parts.append("</section>")

    # Marketing playbook
    parts.append(
        "<section><h2>Marketing playbook</h2>"
        "<p class='lead'>Suggested next step for each segment, ordered by "
        "revenue contribution.</p>"
    )
    parts.append(_advice_section(rfm))
    parts.append("</section>")

    parts.append(
        "<p class='footer'>Generated locally by storefront-cohort. "
        "No data left this machine.</p>"
    )
    parts.append("</div></body></html>")
    return "".join(parts)


def write_report(inputs: ReportInputs, path: str) -> str:
    """Render and write the HTML report to ``path``; returns the path."""
    html_str = render_html(inputs)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html_str)
    return path
