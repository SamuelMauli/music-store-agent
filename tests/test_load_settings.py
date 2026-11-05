"""load_settings: chave de SDK pode morar no arquivo de ambiente, sem vencer o ambiente real."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from emporio.settings import load_settings


def test_sdk_key_in_file_reaches_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMPORIO_TEST_SDK_KEY", raising=False)
    env = tmp_path / "local.env"
    env.write_text("EMPORIO_TEST_SDK_KEY=do-arquivo\n")
    load_settings(env)
    assert os.environ["EMPORIO_TEST_SDK_KEY"] == "do-arquivo"
    monkeypatch.delenv("EMPORIO_TEST_SDK_KEY")


def test_environment_wins_over_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMPORIO_TEST_SDK_KEY", "do-orquestrador")
    env = tmp_path / "local.env"
    env.write_text("EMPORIO_TEST_SDK_KEY=do-arquivo\n")
    load_settings(env)
    assert os.environ["EMPORIO_TEST_SDK_KEY"] == "do-orquestrador"


def test_missing_file_is_fine(tmp_path: Path) -> None:
    assert load_settings(tmp_path / "nao-existe").llm_provider
