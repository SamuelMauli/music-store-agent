"""Métricas Prometheus da API (complementa o OTel: traces por turno, séries para alerta).

Cada app tem o seu ``CollectorRegistry``: o registry global acumularia entre apps e testes.
Rótulos têm cardinalidade limitada (rota é o *template*, intent vem da lista fechada do roteador).
"""

from __future__ import annotations

from collections.abc import Iterable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from emporio.agent.service import ChatResult

CONTENT_TYPE = CONTENT_TYPE_LATEST
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


class Metrics:
    """Contadores e histogramas de um app, num registry próprio."""

    def __init__(self) -> None:
        """Cria o registry e registra as séries."""
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "emporio_http_requests_total",
            "Requisições HTTP por método, rota e status.",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "emporio_http_request_duration_seconds",
            "Duração da requisição (no SSE, até o fim do stream).",
            ["method", "route"],
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.turns = Counter(
            "emporio_turns_total",
            "Turnos concluídos por intent.",
            ["intent"],
            registry=self.registry,
        )
        self.turn_errors = Counter(
            "emporio_turn_errors_total", "Turnos que falharam no meio.", registry=self.registry
        )
        self.handoffs = Counter(
            "emporio_handoffs_total", "Turnos encerrados em handoff humano.", registry=self.registry
        )
        self.guardrail_blocks = Counter(
            "emporio_guardrail_blocks_total",
            "Turnos bloqueados por guardrail.",
            registry=self.registry,
        )
        self.llm_tokens = Counter(
            "emporio_llm_tokens_total",
            "Tokens do LLM por direção (input/output).",
            ["direction"],
            registry=self.registry,
        )
        self.llm_cost = Counter(
            "emporio_llm_cost_usd_total",
            "Custo estimado do LLM em US$.",
            registry=self.registry,
        )

    def observe_request(self, method: str, route: str, status: int, seconds: float) -> None:
        """Registra uma requisição HTTP concluída."""
        self.requests.labels(method, route, str(status)).inc()
        self.latency.labels(method, route).observe(seconds)

    def observe_turn(self, result: ChatResult) -> None:
        """Registra um turno a partir do resumo final do ``ChatService``."""
        intents: Iterable[str] = result.intents or ["none"]
        for intent in intents:
            self.turns.labels(intent).inc()
        if result.handoff:
            self.handoffs.inc()
        if result.blocked:
            self.guardrail_blocks.inc()
        self.llm_tokens.labels("input").inc(result.usage.input_tokens)
        self.llm_tokens.labels("output").inc(result.usage.output_tokens)
        self.llm_cost.inc(result.usage.cost_usd)

    def render(self) -> bytes:
        """Corpo do ``/metrics`` no formato de exposição do Prometheus."""
        return generate_latest(self.registry)
