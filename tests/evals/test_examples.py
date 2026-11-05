from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from emporio.evals import examples
from emporio.settings import Settings
from tests.agent.conftest import fake_settings


def test_five_examples_with_at_least_three_multi_turn() -> None:
    names = [e.filename for e in examples.EXAMPLES]
    assert names == [
        "01_catalogo_violoes.txt",
        "02_preco_promocao_expirada.txt",
        "03_devolucao_arrependimento.txt",
        "04_pedido_atrasado_reclamacao.txt",
        "05_fora_de_escopo_e_seguranca.txt",
    ]
    assert sum(len(e.turns) > 1 for e in examples.EXAMPLES) >= 3


def test_transcript_has_header_turns_and_metrics_footer(ingested: Settings) -> None:
    example = examples.Example(
        "x.txt",
        "Devolução",
        ("Quero devolver o pedido 7", "leticia.rocha@jmail.com", "comprei online"),
        date(2026, 2, 21),
    )
    text = examples.run_example(example, fake_settings(ingested), sleep=0)
    assert "Data de referência: 21/02/2026 10:00" in text
    assert text.count("Cliente: ") == 3
    assert text.count("Tom: ") == 3
    assert "dia 4 de 7" in text
    assert "aparecem mascarados na transcrição" in text
    assert "l***@jmail.com" in text
    assert "leticia.rocha@jmail.com" not in text
    for footer in (
        "Tools chamadas por turno:",
        "Latência por turno:",
        "Tokens:",
        "Custo estimado:",
    ):
        assert footer in text


def test_main_requires_a_real_provider(
    ingested: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EMPORIO_LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMPORIO_DB_PATH", str(ingested.db_path))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="provedor real"):
        examples.main([])


def test_greeting_is_not_glued_to_first_list_item() -> None:
    assert examples.break_list("Oi! Posso ajudar. - A – R$ 1\n- B – R$ 2") == (
        "Oi! Posso ajudar.\n- A – R$ 1\n- B – R$ 2"
    )
    assert examples.break_list("Sem lista aqui - só um hífen") == "Sem lista aqui - só um hífen"
