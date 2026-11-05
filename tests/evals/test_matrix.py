"""Matriz custo x qualidade: agrupamento por família, último relatório e família não rodada."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from emporio.evals.__main__ import main
from emporio.evals.matrix import NOT_RUN, build_matrix, family_of, render_markdown


def _report(
    provider: str, models: dict[str, str], when: str, *, passed: int = 20, cost: float = 0.04
) -> dict[str, Any]:
    return {
        "generated_at": when,
        "provider": provider,
        "models": models,
        "prompt_hash": "abc123",
        "summary": {
            "cases": 23,
            "passed": passed,
            "tool_accuracy": {"rate": 1.0},
            "groundedness": {"rate": 0.9},
            "security": {"rate": 0.8},
            "latency_ms": {"p50": 100, "p95": 900},
            "cost_usd": cost,
            "cost_per_conversation_usd": round(cost / 23, 6),
        },
    }


def _write(directory: Path, name: str, report: dict[str, Any]) -> None:
    (directory / name).write_text(json.dumps(report), encoding="utf-8")


def test_family_from_spec_prefix_or_legacy_host() -> None:
    assert family_of({"models": {"small": "google:g", "medium": "google:h"}}) == "google"
    assert family_of({"models": {"small": "groq:a", "medium": "anthropic:b"}}) == "anthropic+groq"
    legacy = {"provider": "api.groq.com", "models": {"small": "openai/gpt-oss-20b"}}
    assert family_of(legacy) == "groq"
    assert family_of({"provider": "anthropic", "models": {}}) == "anthropic"


def test_latest_per_family_and_history(tmp_path: Path) -> None:
    gem = {"small": "google:a", "medium": "google:b"}
    _write(tmp_path, "live-1.json", _report("google", gem, "2026-11-05T10:00:00+00:00"))
    _write(tmp_path, "live-2.json", _report("google", gem, "2026-11-05T12:00:00+00:00", passed=23))
    _write(
        tmp_path,
        "live-3.json",
        _report("anthropic", {"small": "anthropic:h"}, "2026-11-05T11:00:00+00:00"),
    )
    matrix = build_matrix(tmp_path, {})
    google = matrix["families"]["google"]
    assert google["latest"]["report"] == "live-2.json"
    assert [r["report"] for r in google["history"]] == ["live-1.json"]
    assert google["latest"]["usd_per_passed_case"] == round(0.04 / 23, 6)
    assert matrix["families"]["anthropic"]["history"] == []


def test_expected_family_without_report_is_not_run(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "live-1.json",
        _report("google", {"small": "google:a"}, "2026-11-05T10:00:00+00:00"),
    )
    matrix = build_matrix(tmp_path, {"openai": NOT_RUN, "google": NOT_RUN})
    assert matrix["not_run"] == {"openai": NOT_RUN}  # google tem relatório: não entra
    text = render_markdown(matrix)
    assert "openai | not run (no API key)" in text
    # toda linha da tabela tem o mesmo número de colunas do cabeçalho
    table = [line for line in text.splitlines() if line.startswith("|")]
    assert len({line.count("|") for line in table}) == 1


def test_zero_passed_has_no_cost_per_passed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "live-1.json",
        _report("google", {"small": "google:a"}, "2026-11-05T10:00:00+00:00", passed=0),
    )
    row = build_matrix(tmp_path, {})["families"]["google"]["latest"]
    assert row["usd_per_passed_case"] is None


def test_cli_writes_json_and_fails_without_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="nenhum live"):
        main(["matrix", "--reports", str(tmp_path), "--out", str(tmp_path / "m.json")])
    _write(
        tmp_path,
        "live-1.json",
        _report("google", {"small": "google:a"}, "2026-11-05T10:00:00+00:00"),
    )
    out = tmp_path / "out" / "m.json"
    assert (
        main(["matrix", "--reports", str(tmp_path), "--out", str(out), "--expect", "openai"]) == 0
    )
    assert "Matriz custo x qualidade" in capsys.readouterr().out
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["not_run"] == {"openai": NOT_RUN}
