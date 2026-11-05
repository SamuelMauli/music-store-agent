"""Diagnóstico da configuração de LLM (``make llm-check``).

Para cada papel (small, medium, reservas) mostra provedor, modelo e se há credencial (só a
fonte, nunca o valor), e faz duas chamadas mínimas reais: tool calling e saída estruturada.
Sai com código 1 se algo falhar.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TextIO

from pydantic import BaseModel

from emporio.agent.llm import providers
from emporio.agent.llm.chat import LangChainLLM
from emporio.agent.llm.errors import classify
from emporio.agent.llm.spec import ModelSpec, role_specs
from emporio.agent.llm.types import LLMProvider, Message
from emporio.settings import Settings, load_settings

PING_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Responde ao diagnóstico de conectividade.",
        "parameters": {
            "type": "object",
            "properties": {"eco": {"type": "string", "description": "Texto a ecoar."}},
            "required": ["eco"],
        },
    },
}
TOOL_PROMPT: list[Message] = [
    {"role": "system", "content": "Você é um verificador. Sempre use a ferramenta pedida."},
    {"role": "user", "content": "Chame a ferramenta ping com eco='ok'. Não responda em texto."},
]
STRUCTURED_PROMPT: list[Message] = [
    {"role": "system", "content": "Responda só no formato pedido."},
    {"role": "user", "content": "Classifique: 'quero devolver meu violão'. Tema e confiança."},
]


class CheckFailedError(Exception):
    """A chamada respondeu, mas não como o diagnóstico exige."""


class Probe(BaseModel):
    """Schema mínimo da checagem de saída estruturada."""

    tema: str
    confianca: float


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Resultado de uma checagem."""

    name: str
    ok: bool
    latency_ms: float
    detail: str


Builder = Callable[[Settings, ModelSpec], LLMProvider]


def _default_builder(settings: Settings, spec: ModelSpec) -> LLMProvider:
    return LangChainLLM(settings, model_override=spec)


def _timed(name: str, call: Callable[[], str]) -> CheckResult:
    start = time.perf_counter()
    try:
        detail = call()
    except Exception as exc:  # o diagnóstico reporta qualquer falha
        error = (
            exc if isinstance(exc, CheckFailedError | providers.LLMConfigError) else classify(exc)
        )
        elapsed = (time.perf_counter() - start) * 1000
        return CheckResult(name, False, elapsed, f"{type(error).__name__}: {error}")
    return CheckResult(name, True, (time.perf_counter() - start) * 1000, detail)


def _tool_check(llm: LLMProvider) -> str:
    response = llm.complete(TOOL_PROMPT, tier="small", temperature=0.0, tools=[PING_TOOL])
    if not response.tool_calls:
        raise CheckFailedError("o modelo respondeu sem tool call")
    call = response.tool_calls[0]
    usage = response.usage
    return f"{call.name}({call.args}) tokens={usage.input_tokens}+{usage.output_tokens}"


def _structured_check(llm: LLMProvider) -> str:
    probe, usage = llm.structured(STRUCTURED_PROMPT, Probe, tier="small")
    tokens = f"{usage.input_tokens}+{usage.output_tokens}"
    return f"tema={probe.tema!r} confianca={probe.confianca} tokens={tokens}"


def credential_status(spec: ModelSpec, settings: Settings) -> str:
    """Situação da credencial do provedor, sem revelar o valor."""
    info = providers.info_for(spec.provider)
    found = providers.credential(info, settings)
    if found.value is not None:
        return f"presente ({found.source})"
    return "ausente" if info.requires_key else "não exigida (cadeia padrão/local)"


def check_role(
    role: str, spec: ModelSpec, settings: Settings, builder: Builder = _default_builder
) -> list[CheckResult]:
    """Valida a configuração do papel e faz as duas chamadas mínimas.

    Args:
        role: Nome do papel (``small``, ``medium``, ``fallback[0]``...).
        spec: Provedor e modelo do papel.
        settings: Configuração.
        builder: Monta o provedor da spec; injetável em testes.

    Returns:
        Resultados de configuração, tool calling e saída estruturada.
    """
    config = _timed(f"{role} config", lambda: _validate(spec, settings))
    if not config.ok:
        return [config]
    llm = builder(settings, spec)
    return [
        config,
        _timed(f"{role} tool_calling", lambda: _tool_check(llm)),
        _timed(f"{role} structured", lambda: _structured_check(llm)),
    ]


def _validate(spec: ModelSpec, settings: Settings) -> str:
    providers.validate(spec, settings)
    return "ok"


def run(settings: Settings, out: TextIO = sys.stdout, builder: Builder = _default_builder) -> int:
    """Roda o diagnóstico de todos os papéis.

    Args:
        settings: Configuração.
        out: Destino do relatório.
        builder: Monta o provedor de cada spec; injetável em testes.

    Returns:
        0 se tudo passou, 1 caso contrário.
    """
    if settings.llm_provider == "fake":
        print("EMPORIO_LLM_PROVIDER=fake: nada a diagnosticar (sem rede).", file=out)
        return 0
    failed = False
    for role, spec in role_specs(settings).items():
        print(f"\n[{role}] {spec.label}", file=out)
        print(f"  credencial: {credential_status(spec, settings)}", file=out)
        for result in check_role(role, spec, settings, builder):
            mark = "OK  " if result.ok else "FALHA"
            print(
                f"  {mark} {result.name:<28} {result.latency_ms:8.0f} ms  {result.detail}", file=out
            )
            failed = failed or not result.ok
    print("\nresultado:", "FALHOU" if failed else "OK", file=out)
    return 1 if failed else 0


def main() -> None:
    """Ponto de entrada de ``python -m emporio.agent.llm.doctor``."""
    sys.exit(run(load_settings()))


if __name__ == "__main__":
    main()
