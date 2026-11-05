from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from emporio.agent.llm.types import LLMError, LLMResponse, Usage
from emporio.agent.prompts import prompt_hash
from emporio.evals import __main__ as cli
from emporio.evals.cases import SUITES, Case, Expect, ToolExpect, load_cases
from emporio.evals.cassette import Cassette, RecordingLLM, ReplayLLM, request_key
from emporio.evals.checks import TurnRecord, evaluate, tool_matches
from emporio.evals.judge import JudgeScore, judge
from emporio.evals.runner import (
    CaseResult,
    Evaluator,
    RunOptions,
    build_report,
    percentile,
    summarize,
)
from emporio.settings import Settings
from tests.agent.conftest import fake_settings

ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT / "evals"


def turn(answer: str, traces: list[dict[str, Any]] | None = None, **kw: Any) -> TurnRecord:
    return TurnRecord(
        user=kw.get("user", "oi"),
        answer=answer,
        intents=kw.get("intents", []),
        traces=traces or [],
        handoff=False,
        blocked=False,
        latency_ms=kw.get("latency_ms", 100),
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        calls=1,
    )


def case(**expect: Any) -> Case:
    return Case(id="c", turns=["oi"], expect=Expect(**expect))


def trace(name: str, args: dict[str, Any], payload: str = "{}") -> dict[str, Any]:
    return {"name": name, "args": args, "payload": payload}


# ---------- casos ----------


def test_shipped_suites_load_with_unique_ids_and_live_subset() -> None:
    golden = load_cases("golden", EVALS)
    adversarial = load_cases("adversarial", EVALS)
    assert len(golden) >= 50
    assert len(adversarial) >= 20
    assert sum(c.live for c in golden) >= 15
    assert sum(c.live for c in adversarial) >= 8
    assert {"bola", "pii", "injection"} <= {t for c in adversarial for t in c.tags}
    assert SUITES == ("golden", "adversarial")


def test_case_without_any_check_is_rejected() -> None:
    with pytest.raises(ValidationError, match="sem nenhuma checagem"):
        Case(id="vazio", turns=["oi"], expect=Expect())


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Expect.model_validate({"must_mention": ["a"], "inventado": 1})


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    row = "- {id: a, turns: [oi], expect: {must_mention: [x]}}\n"
    (tmp_path / "golden.yaml").write_text(row * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="ids repetidos"):
        load_cases("golden", tmp_path)


# ---------- checagens ----------


def test_tool_matches_subset_and_normalization() -> None:
    traces = [trace("estimate_shipping", {"city": "São Paulo", "extra": 1})]
    assert tool_matches(ToolExpect(name="estimate_shipping", args={"city": "sao paulo"}), traces)
    assert not tool_matches(ToolExpect(name="estimate_shipping", args={"city": "Rio"}), traces)
    assert not tool_matches(ToolExpect(name="get_product"), traces)
    assert tool_matches(ToolExpect(name="x", args={"n": 2}), [trace("x", {"n": 2.0})])


def test_evaluate_passes_when_everything_matches() -> None:
    c = case(
        intents=["product_detail"],
        tool_calls=[ToolExpect(name="get_product")],
        must_mention=["2.199,00"],
        grounded_numbers=True,
    )
    t = turn(
        "Sai por R$ 2.199,00",
        [trace("get_product", {"name_or_id": "x"}, '{"price": "2199.00"}')],
        intents=["product_detail"],
    )
    out = evaluate(c, [t], [])
    assert out.passed
    assert out.tool_ok is True
    assert out.grounded_ok is True


def test_evaluate_reports_each_failure_dimension() -> None:
    c = case(
        intents=["shipping"],
        tool_calls=[ToolExpect(name="estimate_shipping")],
        forbidden_tools=["open_ticket"],
        must_mention=["PAC"],
        must_mention_any=["x1", "x2"],
        must_not_mention=["grátis"],
        no_pii=True,
        handoff=True,
        grounded_numbers=True,
    )
    t = turn("É grátis, fale com a@b.com por R$ 7,77", [trace("open_ticket", {})])
    out = evaluate(c, [t], [])
    text = " | ".join(out.failures)
    for piece in (
        "intent não detectada",
        "tool não chamada",
        "tool proibida",
        "faltou mencionar: 'PAC'",
        "faltou mencionar qualquer",
        "mencionou o proibido",
        "PII",
        "handoff",
        "números sem fonte",
    ):
        assert piece in text
    assert out.tool_ok is False
    assert out.grounded_ok is False


def test_mentions_apply_to_final_answer_and_tools_to_whole_conversation() -> None:
    c = case(tool_calls=[ToolExpect(name="verify_customer")], must_mention=["entregue"])
    first = turn("preciso do e-mail", [trace("verify_customer", {"order_id": 7})])
    last = turn("pedido entregue")
    assert evaluate(c, [first, last], []).passed


def test_groundedness_accepts_numbers_from_user_and_policy() -> None:
    c = case(grounded_numbers=True)
    t = turn("Prazo de 7 dias e 12x", user="quero 12x")
    assert evaluate(c, [t], ["prazo de 7 dias corridos"]).passed


# ---------- cassete ----------


def test_request_key_is_stable_and_sensitive() -> None:
    m = [{"role": "user", "content": "oi"}]
    assert request_key("complete", "medium", m, None) == request_key("complete", "medium", m, None)
    assert request_key("complete", "medium", m, None) != request_key("complete", "small", m, None)
    assert request_key("complete", "medium", m, None) != request_key(
        "structured", "medium", m, None
    )


class Scripted:
    """LLM de teste: devolve respostas fixas e conta as chamadas."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: Any, **_: Any) -> LLMResponse:
        self.calls += 1
        return LLMResponse(f"resp {self.calls}", (), Usage(3, 2, 0.5, 1))

    def structured(self, messages: Any, schema: Any, **_: Any) -> tuple[Any, Usage]:
        self.calls += 1
        return schema(tone=5, clarity=4, completeness=3), Usage(4, 1, 0.1, 1)


def test_cassette_record_replay_roundtrip_with_fifo_per_key(tmp_path: Path) -> None:
    path = tmp_path / "c" / "live.jsonl"
    inner = Scripted()
    recorder = RecordingLLM(inner, Cassette(path))
    msgs = [{"role": "user", "content": "oi"}]
    first = recorder.complete(msgs)
    second = recorder.complete(msgs)
    scored, _ = recorder.structured(msgs, JudgeScore)
    assert (first.text, second.text) == ("resp 1", "resp 2")
    assert scored.mean == 4.0
    replay = ReplayLLM(Cassette(path))
    assert replay.complete(msgs).text == "resp 1"
    assert replay.complete(msgs).text == "resp 2"
    again, usage = replay.structured(msgs, JudgeScore)
    assert again.tone == 5
    assert usage.cost_usd == pytest.approx(0.1)
    assert inner.calls == 3


def test_replay_without_recording_fails_with_hint(tmp_path: Path) -> None:
    replay = ReplayLLM(Cassette(tmp_path / "vazio.jsonl"))
    with pytest.raises(LLMError, match="make eval-live"):
        replay.complete([{"role": "user", "content": "novo"}])


# ---------- juiz ----------


def test_judge_scores_with_structured_output() -> None:
    score, usage = judge(Scripted(), "pergunta", "resposta")
    assert (score.tone, score.clarity, score.completeness) == (5, 4, 3)
    assert usage.calls == 1
    with pytest.raises(ValidationError):
        JudgeScore(tone=6, clarity=1, completeness=1)


# ---------- métricas ----------


def test_percentile_nearest_rank() -> None:
    assert percentile([], 95) == 0
    assert percentile([5], 50) == 5
    values = list(range(1, 101))
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile([30, 10, 20], 100) == 30


def result(cid: str, passed: bool, tags: list[str], latency: int = 100) -> CaseResult:
    record = {
        "latency_ms": latency,
        "input_tokens": 10,
        "output_tokens": 5,
        "cost_usd": 0.002,
        "llm_calls": 2,
    }
    return CaseResult(cid, "golden", tags, passed, [] if passed else ["x"], True, passed, [record])


def test_summarize_aggregates_rates_cost_and_security() -> None:
    results = [result("a", True, []), result("b", False, ["bola"]), result("c", True, ["pii"])]
    s = summarize(results)
    assert s["cases"] == 3
    assert s["pass_rate"] == pytest.approx(0.6667)
    assert s["security"] == {"n": 2, "passed": 1, "rate": 0.5}
    assert s["tokens"] == {"input": 30, "output": 15}
    assert s["llm_calls"] == 6
    assert s["cost_usd"] == pytest.approx(0.006)
    assert s["cost_per_conversation_usd"] == pytest.approx(0.002)
    assert summarize([])["pass_rate"] is None


def test_gate_flags_failures_and_quality_floors() -> None:
    ok = [result("a", True, [])]
    assert cli.gate(summarize(ok), ok) == []
    bad = [result("a", False, [])]
    problems = cli.gate(summarize(bad), bad)
    assert any("caso reprovado" in p for p in problems)
    assert any("acurácia de tool" in p for p in problems) is False
    assert any("groundedness" in p for p in problems)


# ---------- execução ----------


def test_fake_run_passes_and_skips_fake_false(ingested: Settings) -> None:
    settings = fake_settings(ingested)
    cases = [
        Case(
            id="gd20",
            turns=["Quanto custa o Takamine GD20?"],
            expect=Expect(
                tool_calls=[ToolExpect(name="get_product")],
                must_mention=["2.199,00"],
                grounded_numbers=True,
            ),
        ),
        Case(
            id="so-llm-real",
            fake=False,
            turns=["Compare o Yamaha C40 com o C70"],
            expect=Expect(must_mention=["C70"]),
        ),
    ]
    evaluator = Evaluator(settings, RunOptions(mode="fake"))
    results = evaluator.run(cases)
    assert [r.id for r in results] == ["gd20"]
    assert results[0].passed, results[0].failures
    assert evaluator.skipped == ["so-llm-real"]


def test_failing_expectation_is_reported(ingested: Settings) -> None:
    bad = Case(
        id="x", turns=["Quanto custa o Takamine GD20?"], expect=Expect(must_mention=["R$ 1,00"])
    )
    [res] = Evaluator(fake_settings(ingested), RunOptions()).run([bad])
    assert not res.passed
    assert "faltou mencionar" in res.failures[0]


def test_live_only_filter_and_throttle(ingested: Settings) -> None:
    sleeps: list[float] = []
    cases = [
        Case(
            id="a",
            live=True,
            turns=["Qual o horário?", "e domingo?"],
            expect=Expect(intents=["store_info"]),
        ),
        Case(id="b", turns=["Qual o horário?"], expect=Expect(intents=["store_info"])),
    ]
    evaluator = Evaluator(
        fake_settings(ingested), RunOptions(live_only=True, sleep=7), sleep=sleeps.append
    )
    results = evaluator.run(cases)
    assert [r.id for r in results] == ["a"]
    assert sleeps == [7]


def test_reference_date_changes_the_return_verdict(ingested: Settings) -> None:
    turns = ["Quero devolver o pedido 7", "leticia.rocha@jmail.com", "comprei online"]
    eligible = Case(
        id="in",
        turns=turns,
        reference_date="2026-02-21",
        expect=Expect(must_mention=["4.1", "dia 4 de 7"]),
    )
    late = Case(
        id="out", turns=turns, expect=Expect(must_mention=["4.1"], must_not_mention=["dia 4 de 7"])
    )
    results = Evaluator(fake_settings(ingested), RunOptions()).run([eligible, late])
    assert all(r.passed for r in results), [r.failures for r in results]


def test_live_record_then_replay_reproduces_results(ingested: Settings, tmp_path: Path) -> None:
    cassette = tmp_path / "live.jsonl"
    cases = [
        Case(
            id="gd20",
            live=True,
            turns=["Quanto custa o Takamine GD20?"],
            expect=Expect(must_mention=["2.199,00"], tool_calls=[ToolExpect(name="get_product")]),
        )
    ]
    settings = fake_settings(ingested)
    live = Evaluator(
        settings, RunOptions(mode="live", live_only=True, cassette=cassette), judge_llm=Scripted()
    )
    first = live.run(cases)
    assert cassette.exists()
    assert first[0].judge is not None
    assert first[0].judge["mean"] == 4.0
    replay = Evaluator(
        settings, RunOptions(mode="replay", live_only=True, cassette=cassette), judge_llm=Scripted()
    )
    second = replay.run(cases)
    assert second[0].passed
    assert second[0].turns[0]["answer"] == first[0].turns[0]["answer"]


def test_judge_minimum_fails_the_case(ingested: Settings, tmp_path: Path) -> None:
    c = Case(
        id="j",
        live=True,
        turns=["Qual o horário?"],
        expect=Expect(intents=["store_info"], judge_min=4.5),
    )
    options = RunOptions(mode="live", live_only=True, cassette=tmp_path / "c.jsonl")
    [res] = Evaluator(fake_settings(ingested), options, judge_llm=Scripted()).run([c])
    assert any("juiz" in f for f in res.failures)


def test_runtime_error_is_a_failed_case_not_a_crash(ingested: Settings, tmp_path: Path) -> None:
    c = Case(
        id="r",
        live=True,
        turns=["Quanto custa o Takamine GD20?"],
        expect=Expect(must_mention=["x"]),
    )
    options = RunOptions(mode="replay", live_only=True, cassette=tmp_path / "nada.jsonl")
    [res] = Evaluator(fake_settings(ingested), options).run([c])
    assert not res.passed
    assert "faltou mencionar" in res.failures[0]
    broken = Evaluator(fake_settings(ingested), RunOptions())._result(c, [], "erro na execução: X")
    assert broken.failures == ["erro na execução: X"]


def test_report_masks_pii_and_carries_prompt_hash(ingested: Settings) -> None:
    c = Case(id="p", turns=["Meu e-mail é joao.silva@example.com"], expect=Expect(no_pii=True))
    options = RunOptions()
    results = Evaluator(fake_settings(ingested), options).run([c])
    report = build_report(results, fake_settings(ingested), options, 1.23)
    dumped = json.dumps(report, ensure_ascii=False)
    assert "joao.silva@example.com" not in dumped
    assert report["prompt_hash"] == prompt_hash()
    assert report["mode"] == "fake"
    assert report["summary"]["cases"] == 1


def test_cli_fake_mode_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ingested: Settings
) -> None:
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("EMPORIO_DB_PATH", str(ingested.db_path))
    out = tmp_path / "r.json"
    code = cli.main(["--mode", "fake", "--suite", "adversarial", "--out", str(out)])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0
    assert data["summary"]["pass_rate"] == 1.0


def test_judge_requires_a_model_and_live_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    with pytest.raises(SystemExit):
        cli.main(["--mode", "fake", "--judge"])
    monkeypatch.delenv("EMPORIO_EVAL_JUDGE_MODEL", raising=False)
    with pytest.raises(SystemExit):
        cli.main(["--mode", "replay", "--judge"])


# ---------- frescor do relatório ao vivo ----------


def test_latest_live_report_matches_current_prompt() -> None:
    reports = sorted((EVALS / "reports").glob("live-*.json"))
    if not reports:
        pytest.skip("nenhum relatório ao vivo ainda")
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    if report["prompt_hash"] != prompt_hash():
        # eval-live exige provedor real e cota; não roda em CI, então o frescor é aviso, não gate.
        pytest.skip("prompt mudou desde o último eval-live: rode make eval-live e versione")
