"""End-to-end tests for the CLI subcommands."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from storefront_cohort.cli import main
from storefront_cohort.sample_data import write_sample


@pytest.fixture()
def sample_csv(tmp_path):
    path = str(tmp_path / "orders.csv")
    write_sample(path, n_customers=200, seed=11)
    return path


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "report" in out and "segments" in out and "sample" in out


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_report_subcommand(sample_csv, tmp_path, capsys):
    outdir = str(tmp_path / "out")
    rc = main(["report", sample_csv, "-o", outdir, "--store", "Acme"])
    assert rc == 0
    assert os.path.exists(os.path.join(outdir, "report.html"))
    assert os.path.exists(os.path.join(outdir, "churn_risk.csv"))
    assert os.path.exists(os.path.join(outdir, "high_value.csv"))
    assert os.path.exists(os.path.join(outdir, "customers.csv"))
    out = capsys.readouterr().out
    assert "Report written" in out


def test_report_csv_contents(sample_csv, tmp_path):
    outdir = str(tmp_path / "out")
    main(["report", sample_csv, "-o", outdir])
    churn = pd.read_csv(os.path.join(outdir, "churn_risk.csv"))
    assert "predicted_clv" in churn.columns
    assert "recommended_action" in churn.columns


def test_report_with_horizon(sample_csv, tmp_path):
    outdir = str(tmp_path / "out")
    rc = main(["report", sample_csv, "-o", outdir, "--horizon", "6", "--clv-method", "heuristic"])
    assert rc == 0
    html = open(os.path.join(outdir, "report.html"), encoding="utf-8").read()
    assert "next 6" in html


def test_segments_subcommand(sample_csv, capsys):
    rc = main(["segments", sample_csv])
    assert rc == 0
    out = capsys.readouterr().out
    assert "segment" in out
    assert "Champions" in out or "Loyal Customers" in out


def test_cohort_subcommand(sample_csv, capsys):
    rc = main(["cohort", sample_csv])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cohorts" in out


def test_sample_subcommand_writes(tmp_path, capsys):
    out_path = str(tmp_path / "gen.csv")
    rc = main(["sample", "-o", out_path, "--customers", "50"])
    assert rc == 0
    assert os.path.exists(out_path)
    df = pd.read_csv(out_path)
    assert {"customer_id", "order_date", "amount"}.issubset(df.columns)


def test_sample_subcommand_locates_bundled(capsys):
    rc = main(["sample"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sample_orders.csv" in out


def test_missing_file_returns_error_code(capsys):
    rc = main(["segments", "does_not_exist.csv"])
    assert rc == 2


def test_unmappable_returns_error_code(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(bad, index=False)
    rc = main(["report", str(bad)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "error:" in err


def test_explicit_map_flag(tmp_path):
    p = tmp_path / "weird.csv"
    pd.DataFrame(
        {"who": ["a", "a"], "when": ["2024-01-01", "2024-02-01"], "cost": [10, 20]}
    ).to_csv(p, index=False)
    rc = main(["segments", str(p), "--map", "customer=who,date=when,amount=cost"])
    assert rc == 0
