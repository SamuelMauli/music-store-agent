"""Fábrica de tracer e exportadores; sem exportador configurado, tudo é no-op."""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from emporio.settings import Settings

SERVICE_NAME = "emporio-agent"
type AttributeValue = str | int | float | bool


class JsonlSpanExporter(SpanExporter):
    """Um span por linha (JSON OTLP-like) em arquivo; uso local e auditoria."""

    def __init__(self, path: Path) -> None:
        """Cria o exportador, garantindo o diretório."""
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Anexa os spans ao arquivo."""
        with self._path.open("a", encoding="utf-8") as handle:
            for span in spans:
                handle.write(span.to_json(indent=None) + "\n")
        return SpanExportResult.SUCCESS


class Telemetry:
    """Fachada fina sobre o tracer: o resto do código não importa o SDK."""

    def __init__(self, tracer: Tracer, provider: TracerProvider | None = None) -> None:
        """Cria a fachada.

        Args:
            tracer: Tracer usado para todos os spans.
            provider: Provider do SDK, quando houver (para ``shutdown``).
        """
        self._tracer = tracer
        self._provider = provider

    @property
    def enabled(self) -> bool:
        """Verdadeiro quando há exportador por trás."""
        return self._provider is not None

    @contextmanager
    def span(self, name: str, **attributes: AttributeValue) -> Iterator[Span]:
        """Abre um span filho do atual; exceção marca erro e propaga."""
        with self._tracer.start_as_current_span(
            name, attributes=attributes, record_exception=True, set_status_on_exception=True
        ) as span:
            yield span

    @staticmethod
    def fail(span: Span, message: str) -> None:
        """Marca o span como erro sem levantar exceção."""
        span.set_status(Status(StatusCode.ERROR, message))

    def shutdown(self) -> None:
        """Descarrega e fecha o exportador."""
        if self._provider is not None:
            self._provider.shutdown()


NOOP = Telemetry(trace.NoOpTracer())


def _otlp_exporter(endpoint: str) -> SpanExporter:
    try:
        module = importlib.import_module("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    except ImportError as exc:
        raise ValueError(
            "EMPORIO_OTEL_EXPORTER=otlp exige o pacote opentelemetry-exporter-otlp-proto-http."
        ) from exc
    exporter: SpanExporter = module.OTLPSpanExporter(endpoint=endpoint)
    return exporter


def _exporter(settings: Settings) -> SpanExporter | None:
    match settings.otel_exporter:
        case "console":
            return ConsoleSpanExporter(formatter=lambda s: s.to_json(indent=None) + "\n")
        case "jsonl":
            return JsonlSpanExporter(settings.otel_jsonl_path)
        case "otlp":
            return _otlp_exporter(settings.otel_endpoint)
        case _:
            return None


def build_telemetry(settings: Settings, exporter: SpanExporter | None = None) -> Telemetry:
    """Monta o tracer conforme ``settings.otel_exporter`` (``none`` devolve o no-op).

    Args:
        settings: Configuração (exportador, caminho do JSONL, endpoint OTLP).
        exporter: Exportador explícito (testes); vence o configurado.

    Returns:
        ``Telemetry`` com provider próprio (sem tocar no provider global do processo).
    """
    chosen = exporter or _exporter(settings)
    if chosen is None:
        return NOOP
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    provider.add_span_processor(SimpleSpanProcessor(chosen))
    return Telemetry(provider.get_tracer(SERVICE_NAME), provider)


def attributes_of(**values: Any) -> dict[str, AttributeValue]:
    """Descarta ``None`` e converte listas em string para virar atributo válido."""
    return {k: v for k, v in values.items() if isinstance(v, str | int | float | bool)}
