from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from emporio.bootstrap import build_container
from emporio.interfaces.cli import app, run_chat
from emporio.settings import Settings
from tests.agent.conftest import fake_settings


def script(*lines: str) -> object:
    feed = iter(lines)

    def read() -> str:
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    return read


def drive(ingested: Settings, *lines: str, debug: bool = False) -> str:
    container = build_container(fake_settings(ingested))
    console = Console(record=True, width=100, force_terminal=False)
    try:
        run_chat(container.service, "cli1", script(*lines), console, debug=debug)  # type: ignore[arg-type]
    finally:
        container.close()
    return console.export_text()


def test_chat_prints_the_answer_in_a_panel(ingested: Settings) -> None:
    out = drive(ingested, "Quais são as promoções ativas?", "/sair")
    assert "Crafter HT-100" in out
    assert "Tom" in out
    assert "assistente virtual" in out


def test_debug_shows_intents_tools_tokens_and_cost(ingested: Settings) -> None:
    out = drive(ingested, "/debug", "Quais são as promoções ativas?")
    for label in ("active_promotions", "list_active_promotions", "tokens", "custo do turno"):
        assert label in out


def test_debug_can_be_toggled_off(ingested: Settings) -> None:
    out = drive(ingested, "/debug", "/debug", "Oi")
    assert "desligado" in out
    assert "custo do turno" not in out


def test_reset_restarts_the_conversation(ingested: Settings) -> None:
    out = drive(ingested, "Oi", "/reset", "Oi", debug=True)
    assert "Conversa reiniciada" in out
    assert out.count("turno 1") == 2


def test_blank_lines_are_ignored_and_eof_ends(ingested: Settings) -> None:
    assert "Tom" not in drive(ingested, "", "   ")


def test_typer_command_runs_end_to_end(
    ingested: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = {
        "EMPORIO_DB_PATH": str(ingested.db_path),
        "EMPORIO_RAW_DIR": str(ingested.raw_dir),
        "EMPORIO_LLM_PROVIDER": "fake",
        "EMPORIO_CHECKPOINT_PATH": str(tmp_path / "ck.sqlite"),
        "EMPORIO_TICKETS_PATH": str(tmp_path / "tk.sqlite"),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    result = CliRunner().invoke(
        app,
        ["--session", "t1", "--reference-date", "2026-02-21", "--debug"],
        input="Vocês abrem hoje?\n/sair\n",
    )
    assert result.exit_code == 0, result.output
    assert "abertos agora" in result.output  # sábado 10h, data simulada
    assert "get_store_info" in result.output
