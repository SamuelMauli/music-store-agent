"""Contrato único das tools: validação, autorização, rastreio e limite de tempo."""

from __future__ import annotations

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel, ValidationError

from emporio.agent.tools.results import (
    Ok,
    PolicyBlocked,
    ToolError,
    ToolResult,
    to_json,
)
from emporio.domain.pii import mask_pii

logger = logging.getLogger("emporio.tools")

TOOL_TIMEOUT_SECONDS: Final = 2.0
MAX_VERIFICATION_ATTEMPTS: Final = 3
NOT_VERIFIED_REASON: Final = (
    "é preciso confirmar a identidade (número do pedido e e-mail ou telefone do cadastro) "
    "antes de consultar pedidos"
)


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Contexto de uma chamada; nunca é renderizado no prompt.

    Attributes:
        session_id: Sessão da conversa.
        verified_customer_id: Cliente verificado, se houver.
        verification_attempts: Tentativas de verificação já feitas na sessão.
    """

    session_id: str = "anonymous"
    verified_customer_id: int | None = None
    verification_attempts: int = 0


@dataclass(frozen=True, slots=True)
class ToolExecution:
    """Resultado de uma chamada, com tudo que o grafo e o trace precisam.

    Attributes:
        name: Nome da tool.
        args: Argumentos recebidos (crus; mascare antes de logar).
        result: Resultado tipado.
        payload: JSON compacto entregue ao LLM.
        duration_ms: Duração da chamada.
    """

    name: str
    args: Mapping[str, Any]
    result: ToolResult[Any]
    payload: str
    duration_ms: int

    @property
    def sources(self) -> tuple[str, ...]:
        """Rastro das fontes (só ``Ok`` as carrega)."""
        return self.result.sources if isinstance(self.result, Ok) else ()

    @property
    def state_updates(self) -> Mapping[str, Any]:
        """Atualizações de estado (só ``Ok`` as carrega)."""
        return self.result.state_updates if isinstance(self.result, Ok) else {}


class Tool[I: BaseModel, O: BaseModel](ABC):
    """Comando com entrada e saída Pydantic.

    A descrição é escrita como prompt: diz quando usar e quando não usar. Tool não
    formata prosa: devolve dados compactos já em PT-BR.
    """

    name: str
    description: str
    input_model: type[I]
    requires_verified_customer: bool = False
    writes: bool = False

    @abstractmethod
    def run(self, args: I, ctx: ToolContext) -> ToolResult[O]:
        """Executa a regra de negócio."""

    def schema(self) -> dict[str, Any]:
        """Schema de função no formato OpenAI para o LLM."""
        parameters = self.input_model.model_json_schema()
        parameters.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def execute(self, raw_args: Mapping[str, Any], ctx: ToolContext) -> ToolResult[Any]:
        """Valida a entrada, checa a verificação e converte exceções em ``ToolError``."""
        try:
            args = self.input_model.model_validate(dict(raw_args))
        except ValidationError as exc:
            fields = ", ".join(str(e["loc"][0]) for e in exc.errors() if e["loc"])
            return ToolError(
                f"Argumentos inválidos ({fields or 'formato'}). Corrija e tente de novo."
            )
        if self.requires_verified_customer and ctx.verified_customer_id is None:
            return PolicyBlocked(NOT_VERIFIED_REASON)
        try:
            return self.run(args, ctx)
        except Exception:
            logger.exception("tool %s falhou", self.name)
            return ToolError("Não consegui consultar agora. Ofereça abrir um chamado.")


class ToolRegistry:
    """Registro das tools: schema por intent, execução com timeout e rastreio."""

    def __init__(
        self,
        tools: list[Tool[Any, Any]],
        lock: threading.RLock | None = None,
        timeout: float = TOOL_TIMEOUT_SECONDS,
    ) -> None:
        """Cria o registro.

        Args:
            tools: Tools disponíveis; nomes são únicos.
            lock: Trava compartilhada com quem lê o DuckDB fora das tools.
            timeout: Limite por chamada, em segundos.

        Raises:
            ValueError: Se houver nome de tool repetido.
        """
        names = [t.name for t in tools]
        if len(set(names)) != len(names):
            raise ValueError("nome de tool duplicado")
        self._tools = {t.name: t for t in tools}
        self._lock = lock or threading.RLock()
        self._timeout = timeout
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tool")

    @property
    def names(self) -> list[str]:
        """Nomes registrados, em ordem de cadastro."""
        return list(self._tools)

    def get(self, name: str) -> Tool[Any, Any]:
        """Busca a tool pelo nome."""
        return self._tools[name]

    def write_tools(self) -> list[str]:
        """Nomes das tools que escrevem (o CI exige só ``open_ticket``)."""
        return [name for name, tool in self._tools.items() if tool.writes]

    def schemas(self, names: list[str] | tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """Schemas das tools pedidas (carregamento dinâmico por intent)."""
        selected = self._tools if names is None else {n: self._tools[n] for n in names}
        return [tool.schema() for tool in selected.values()]

    def call(self, name: str, raw_args: Mapping[str, Any], ctx: ToolContext) -> ToolExecution:
        """Executa uma tool com timeout, trava de dados e log com PII mascarada."""
        started = time.monotonic()
        tool = self._tools.get(name)
        if tool is None:
            result: ToolResult[Any] = ToolError(f"Ferramenta desconhecida: {name}.")
        else:
            result = self._run_bounded(tool, raw_args, ctx)
        duration = int((time.monotonic() - started) * 1000)
        logger.info(
            "tool=%s result=%s ms=%d args=%s",
            name,
            type(result).__name__,
            duration,
            mask_pii(json.dumps(dict(raw_args), ensure_ascii=False, default=str)),
        )
        return ToolExecution(name, dict(raw_args), result, to_json(result), duration)

    def _run_bounded(
        self, tool: Tool[Any, Any], raw_args: Mapping[str, Any], ctx: ToolContext
    ) -> ToolResult[Any]:
        def locked() -> ToolResult[Any]:
            with self._lock:
                return tool.execute(raw_args, ctx)

        future: Future[ToolResult[Any]] = self._pool.submit(locked)
        try:
            return future.result(timeout=self._timeout)
        except FutureTimeout:
            return ToolError("A consulta demorou demais. Ofereça tentar de novo ou abrir chamado.")
