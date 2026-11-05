from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pytest

from emporio.clock import FixedClock
from emporio.ingestion.pipeline import run_pipeline
from emporio.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = datetime(2026, 3, 25, 10, 0, tzinfo=ZoneInfo("America/Campo_Grande"))


@pytest.fixture(scope="session")
def reference_clock() -> FixedClock:
    return FixedClock(REFERENCE)


@pytest.fixture(scope="session")
def real_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    workdir = tmp_path_factory.mktemp("emporio")
    return Settings(
        db_path=workdir / "emporio.duckdb",
        raw_dir=ROOT / "data" / "raw",
        contracts_dir=ROOT / "data" / "contracts",
        quality_report_path=workdir / "quality_report.json",
        reference_datetime=REFERENCE,
        # classificador real (fastembed) só nos testes que o pedem; aqui, heurística pura
        injection_classifier=False,
    )


@pytest.fixture(scope="session")
def ingested(real_settings: Settings, reference_clock: FixedClock) -> Settings:
    """Roda o pipeline uma única vez sobre os CSVs reais."""
    run_pipeline(real_settings, reference_clock)
    return real_settings


@pytest.fixture
def con(ingested: Settings) -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect(str(ingested.db_path), read_only=True)
    yield connection
    connection.close()
