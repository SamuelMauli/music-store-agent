"""``ToolResult``: estados explícitos que o LLM recebe, nunca stack trace.

Cada estado carrega a instrução de como responder. ``Unauthorized`` serializa
exatamente como ``NotFound`` de pedido (anti-BOLA): o cliente não distingue
"não existe" de "não é seu".
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from pydantic import BaseModel

ORDER_NOT_FOUND_HINT: Final = (
    "Não encontrei esse pedido para o cadastro informado. Peça para conferir o número do pedido."
)


@dataclass(frozen=True, slots=True)
class ProductRef:
    """Produto citado numa resposta, para resolver "esse" e "o mais barato" depois.

    Attributes:
        product_id: Identificador do produto.
        name: Nome comercial.
        price: Preço que o cliente paga, já formatado (resolve "o mais barato").
    """

    product_id: int
    name: str
    price: str = ""


@dataclass(frozen=True, slots=True)
class Ok[T: BaseModel]:
    """Sucesso.

    Attributes:
        value: Saída da tool, já formatada em PT-BR (o LLM copia, não calcula).
        sources: Rastro para o trace: ``product:95``, ``order:7``, ``§4.1``.
        refs: Produtos citados.
        state_updates: Mudanças de estado do grafo que nunca vão ao LLM.
    """

    value: T
    sources: tuple[str, ...] = ()
    refs: tuple[ProductRef, ...] = ()
    state_updates: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NotFound:
    """Nada encontrado.

    Attributes:
        hint: O que o agente deve dizer ou sugerir.
        similar: Candidatos parecidos (nome e ID), se houver.
    """

    hint: str = "Não encontrei."
    similar: tuple[ProductRef, ...] = ()


@dataclass(frozen=True, slots=True)
class Unauthorized:
    """Recurso existe mas não pertence ao cliente verificado; indistinguível de NotFound."""


@dataclass(frozen=True, slots=True)
class Ambiguous:
    """Falta escolher entre opções ou informar um dado.

    Attributes:
        question: Pergunta a fazer ao cliente.
        options: Opções em texto.
    """

    question: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PolicyBlocked:
    """A política da loja impede ou adia a ação.

    Attributes:
        reason: Motivo para explicar ao cliente.
        options: Caminhos que o agente pode oferecer.
        section: Seção do manual de origem, ex.: ``"4.1"``.
    """

    reason: str
    options: tuple[str, ...] = ()
    section: str | None = None


@dataclass(frozen=True, slots=True)
class ToolError:
    """Falha genérica (entrada inválida, timeout, exceção), sem detalhes internos.

    Attributes:
        message: Mensagem curta e segura.
    """

    message: str


type ToolResult[T: BaseModel] = (
    Ok[T] | NotFound | Unauthorized | Ambiguous | PolicyBlocked | ToolError
)


def to_payload(result: ToolResult[Any]) -> dict[str, Any]:
    """Converte o resultado no dicionário compacto entregue ao LLM.

    Args:
        result: Resultado da tool.

    Returns:
        Dicionário JSON-serializável com ``status`` e instruções.
    """
    match result:
        case Ok(value=value):
            return {"status": "ok", "data": value.model_dump(mode="json", exclude_none=True)}
        case NotFound(hint=hint, similar=similar):
            payload: dict[str, Any] = {"status": "not_found", "message": hint}
            if similar:
                payload["similar"] = [{"product_id": r.product_id, "name": r.name} for r in similar]
            return payload
        case Unauthorized():
            return {"status": "not_found", "message": ORDER_NOT_FOUND_HINT}
        case Ambiguous(question=question, options=options):
            return {"status": "ambiguous", "ask": question, "options": list(options)}
        case PolicyBlocked(reason=reason, options=options, section=section):
            payload = {"status": "policy_blocked", "reason": reason, "options": list(options)}
            if section:
                payload["section"] = f"§{section}"
            return payload
        case ToolError(message=message):
            return {"status": "error", "message": message}
    raise AssertionError("ToolResult inexaustivo")  # pragma: no cover


def to_json(result: ToolResult[Any]) -> str:
    """Serializa o payload sem espaços supérfluos (cada token volta como entrada)."""
    return json.dumps(to_payload(result), ensure_ascii=False, separators=(",", ":"))
