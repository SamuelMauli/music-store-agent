from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from emporio.clock import FixedClock
from emporio.ingestion.__main__ import main
from emporio.ingestion.pipeline import run_pipeline
from emporio.ingestion.quality import format_quality_report
from emporio.settings import Settings
from tests.conftest import REFERENCE, ROOT
from tests.integration.test_ingestion import append_rows, dirty_raw

SNAPSHOT = ROOT / "tests" / "snapshots" / "quality_report.json"

EXPECTED_IDS = {
    "empty_categories": [6, 7, 8],
    "brand_consistency": [135, 136, 137, 139, 140, 142, 144],
    "specs_match_description": [137, 139, 140, 142],
    "active_without_stock": [96],
    "discontinued_without_stock": [113],
    "discontinued_has_stock": [136],
    "unreleased_in_orders": [3, 14],
    "order_total_mismatch": [3, 20],
    "duplicate_email": [11, 43],
    "multiple_promotions_per_product": [93, 96, 99],
    "multiple_active_promotions_per_product": [],
    "active_promotions": [90, 94, 121, 127],
    "shipped_overdue": [8],
}


def test_quality_report_snapshot(ingested: Settings) -> None:
    report = json.loads(ingested.quality_report_path.read_text(encoding="utf-8"))
    assert report == json.loads(SNAPSHOT.read_text(encoding="utf-8"))


@pytest.mark.parametrize(("rule", "ids"), EXPECTED_IDS.items())
def test_rule_flags_the_known_pitfalls(ingested: Settings, rule: str, ids: list[int]) -> None:
    report = json.loads(ingested.quality_report_path.read_text(encoding="utf-8"))
    assert report["rules"][rule]["affected_ids"] == ids


def test_report_covers_d1_to_d14(ingested: Settings) -> None:
    report = json.loads(ingested.quality_report_path.read_text(encoding="utf-8"))
    codes = {rule["finding"] for rule in report["rules"].values()}
    assert codes == {f"D{n}" for n in range(1, 15)}
    assert report["summary"]["rules_evaluated"] == len(report["rules"])


def test_report_never_leaks_raw_email(ingested: Settings) -> None:
    text = ingested.quality_report_path.read_text(encoding="utf-8")
    con = duckdb.connect(str(ingested.db_path), read_only=True)
    try:
        email = con.execute("SELECT email FROM silver.customers WHERE customer_id = 11").fetchone()
    finally:
        con.close()
    assert email is not None
    assert str(email[0]) not in text


def test_second_active_promotion_is_reported(tmp_path: Path, reference_clock: FixedClock) -> None:
    raw = dirty_raw(tmp_path)
    append_rows(raw / "promotions.csv", "900,127,25,Segunda promoção,true")
    settings = Settings(
        db_path=tmp_path / "q.duckdb",
        raw_dir=raw,
        contracts_dir=ROOT / "data" / "contracts",
        quality_report_path=tmp_path / "q.json",
        reference_datetime=REFERENCE,
    )
    result = run_pipeline(settings, reference_clock)
    assert result.quality["rules"]["multiple_active_promotions_per_product"]["affected_ids"] == [
        127
    ]


def test_format_quality_report_lists_every_rule(ingested: Settings) -> None:
    report = json.loads(ingested.quality_report_path.read_text(encoding="utf-8"))
    text = format_quality_report(report)
    assert "16/17 regras com achados" in text
    assert "order_total_mismatch" in text
    assert "..." in text  # listas longas são truncadas


def test_cli_main_runs_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("EMPORIO_DB_PATH", str(tmp_path / "cli.duckdb"))
    monkeypatch.setenv("EMPORIO_RAW_DIR", str(ROOT / "data" / "raw"))
    monkeypatch.setenv("EMPORIO_CONTRACTS_DIR", str(ROOT / "data" / "contracts"))
    monkeypatch.setenv("EMPORIO_QUALITY_REPORT_PATH", str(tmp_path / "report.json"))
    assert main() == 0
    out = capsys.readouterr().out
    assert "Relatório de qualidade" in out
    assert (tmp_path / "report.json").exists()


def test_cli_main_reports_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("EMPORIO_DB_PATH", str(tmp_path / "cli.duckdb"))
    monkeypatch.setenv("EMPORIO_RAW_DIR", str(tmp_path / "vazio"))
    monkeypatch.setenv("EMPORIO_CONTRACTS_DIR", str(ROOT / "data" / "contracts"))
    monkeypatch.setenv("EMPORIO_QUALITY_REPORT_PATH", str(tmp_path / "report.json"))
    assert main() == 1
    assert "falha na ingestão" in capsys.readouterr().err
