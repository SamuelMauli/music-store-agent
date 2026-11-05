"""Calibração do juiz: kappa com valores conhecidos, harness offline e integridade dos rótulos."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emporio.agent.llm.types import Usage
from emporio.evals import calibration
from emporio.evals.__main__ import main
from emporio.evals.calibration import (
    LabeledItem,
    best_threshold,
    calibrate,
    cassette_key,
    confusion,
    heuristic_judge,
    kappa,
    load_labels,
    replay_judge,
    run_calibration,
)
from emporio.evals.judge import JudgeScore

REPORTS = Path("evals/reports")


def test_kappa_known_values() -> None:
    # exemplo clássico (Wikipedia): 20 sim/sim, 5 sim/não, 10 não/sim, 15 não/não
    truth = [True] * 25 + [False] * 25
    pred = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    assert kappa(truth, pred) == pytest.approx(0.4)
    assert kappa([True, False, True], [True, False, True]) == pytest.approx(1.0)
    # inverso exato: pior que o acaso
    assert kappa([True, False, True, False], [False, True, False, True]) == pytest.approx(-1.0)
    # concordância de 50% com marginais iguais = acaso puro
    assert kappa([True, True, False, False], [True, False, True, False]) == pytest.approx(0.0)


def test_kappa_undefined_and_multiclass() -> None:
    assert kappa([], []) is None
    assert kappa([True, True], [True, True]) is None  # só uma categoria: pe == 1
    assert kappa([1, 2, 3, 1], [1, 2, 3, 1]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="tamanhos"):
        kappa([True], [True, False])


def test_confusion_matrix() -> None:
    m = confusion([True, True, False, False, True], [True, False, True, False, True])
    assert m == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}


def _item(i: int, tone: bool, clarity: bool = True, completeness: bool = True) -> LabeledItem:
    return LabeledItem.model_validate(
        {
            "id": f"i{i}",
            "source": {"kind": "synthetic"},
            "question": f"q{i}",
            "answer": f"a{i}",
            "labels": {"tone": tone, "clarity": clarity, "completeness": completeness},
            "labeler": "ai-draft",
            "reviewed_by_human": False,
        }
    )


def test_calibrate_picks_threshold_with_best_kappa() -> None:
    items = [_item(0, True), _item(1, True), _item(2, False), _item(3, False)]
    # o juiz dá 5/4 para o bom e 3/2 para o ruim: limiar 4 separa perfeitamente
    scores = [JudgeScore(tone=t, clarity=5, completeness=5) for t in (5, 4, 3, 2)]
    report = calibrate(items, scores)
    tone = report["criteria"]["tone"]
    assert tone["chosen"]["threshold"] == 4
    assert tone["chosen"]["kappa"] == pytest.approx(1.0)
    assert tone["chosen"]["confusion"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}
    assert tone["score_distribution"] == {"2": 1, "3": 1, "4": 1, "5": 1}
    # critério sem negativos: kappa indefinido, mas o relatório sai
    assert report["criteria"]["clarity"]["chosen"]["kappa"] is None


def test_best_threshold_tie_prefers_stricter() -> None:
    sweep = [
        {"threshold": 3, "accuracy": 0.8, "kappa": 0.5, "confusion": {}},
        {"threshold": 4, "accuracy": 0.8, "kappa": 0.5, "confusion": {}},
    ]
    assert best_threshold(sweep)["threshold"] == 4


def test_calibrate_rejects_size_mismatch() -> None:
    with pytest.raises(ValueError, match="mesmo tamanho"):
        calibrate([_item(0, True)], [])


def test_fake_mode_runs_offline_and_flags_not_llm(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["judge-calibrate", "--mode", "fake"]) == 0
    out = capsys.readouterr().out
    assert "não é LLM" in out
    assert "rascunho de IA" in out
    for criterion in calibration.CRITERIA:
        assert criterion in out


def test_run_calibration_report_marks_fake_and_draft_labels() -> None:
    report = run_calibration(load_labels(), heuristic_judge, "fake")
    assert report["judge_is_llm"] is False
    assert report["reviewed_by_human"] == 0
    assert report["warning"]


def test_replay_roundtrip_and_missing_pair(tmp_path: Path) -> None:
    cassette = tmp_path / "judge.json"
    row = {"score": {"tone": 4, "clarity": 5, "completeness": 3}, "usage": Usage().to_dict()}
    cassette.write_text(json.dumps({cassette_key("q0", "a0"): row}), encoding="utf-8")
    fn = replay_judge(cassette)
    score, _ = fn("q0", "a0")
    assert (score.tone, score.clarity, score.completeness) == (4, 5, 3)
    with pytest.raises(LookupError, match="sem gravação"):
        fn("outra", "coisa")


def test_replay_without_cassette_is_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="cassete do juiz ausente"):
        main(["judge-calibrate", "--mode", "replay", "--cassette", str(tmp_path / "nao.json")])


def test_cassette_key_changes_with_rubric(monkeypatch: pytest.MonkeyPatch) -> None:
    before = cassette_key("q", "a")
    monkeypatch.setattr(calibration, "RUBRIC", "rubrica nova")
    assert cassette_key("q", "a") != before


# --- integridade dos rótulos versionados ---------------------------------------------------


def test_labels_file_is_honest_and_grounded_in_reports() -> None:
    items = load_labels()
    assert len(items) >= 40
    # nenhum rótulo se apresenta como humano enquanto ninguém revisou
    assert all(i.labeler == "ai-draft" and i.reviewed_by_human is False for i in items)
    real = [i for i in items if i.source["kind"] == "real"]
    synthetic = [i for i in items if i.source["kind"] == "synthetic"]
    assert len(real) >= 30
    assert len(synthetic) >= 5
    # cada item real é cópia literal de uma resposta nos relatórios ao vivo
    answers: dict[tuple[str, str, int], tuple[str, str]] = {}
    for path in REPORTS.glob("live-*.json"):
        for case in json.loads(path.read_text(encoding="utf-8"))["cases"]:
            for n, turn in enumerate(case["turns"]):
                answers[(path.name, case["id"], n)] = (turn["user"], turn["answer"])
    for item in real:
        key = (item.source["report"], item.source["case"], item.source["turn"])
        assert answers[key] == (item.question, item.answer), item.id
    # inclui respostas de casos reprovados e há negativos em todo critério
    failed = {
        (path.name, c["id"])
        for path in REPORTS.glob("live-*.json")
        for c in json.loads(path.read_text(encoding="utf-8"))["cases"]
        if not c["passed"]
    }
    assert any((i.source["report"], i.source["case"]) in failed for i in real)
    for criterion in calibration.CRITERIA:
        assert any(not getattr(i.labels, criterion) for i in items)
