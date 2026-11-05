"""LLM fake determinístico: roteia por regras, chama tools por plano e narra os payloads.

Serve ao CI e à avaliação offline (sem chave). Não calcula nada: todo valor da resposta vem
copiado do resultado das tools, como o prompt exige do modelo real.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from typing import Any, Final

from pydantic import BaseModel

from emporio.agent.intents import Intent
from emporio.agent.llm import fake_nlu
from emporio.agent.llm.fake_render import CLOSING, render_answer
from emporio.agent.llm.types import LLMError, LLMResponse, Message, Tier, ToolCall, Usage
from emporio.agent.prompts import SUMMARY_MARKER
from emporio.domain.text import normalize_text

_INTENTS_LINE: Final = re.compile(r"Interpretação desta mensagem \(roteador\): (.*)\.\s*$", re.M)
_ENTITIES_LINE: Final = re.compile(r"^Entidades: (\{.*\})\s*$", re.M)
_ORDER_LINE: Final = re.compile(r"Pedido em atendimento: (\d+)")
_PENDING_LINE: Final = "O atendimento aguardava dados do cliente"
_SOCIAL: Final = "conversa social"
_SUPPORT_TEXT: Final = (
    "Posso te ajudar com instrumentos, preços, pedidos, trocas e informações da loja. "
    "O que você procura?"
)
_ASKS: Final[dict[Intent, str]] = {
    Intent.PRODUCT_DETAIL: "De qual modelo você está falando?",
    Intent.PAYMENT_SIMULATION: "De qual modelo você quer simular o pagamento?",
    Intent.SHIPPING: "Para qual cidade seria a entrega?",
    Intent.ORDER_STATUS: "Qual é o número do pedido?",
    Intent.RETURN_EXCHANGE: "Qual é o número do pedido?",
}
_TOPICS: Final = (
    ("address", r"endereco|onde fica|localiza"),
    ("contacts", r"telefone|whatsapp|contato|e-?mail"),
    ("hours", r"horario"),
    ("open_now", r"abre|abrem|aberto|aberta|abertos|fecha|fecham"),
)
_TICKET_KIND: Final = {
    Intent.COMPLAINT: "complaint",
    Intent.PRIVACY_REQUEST: "privacy",
    Intent.HUMAN_HANDOFF: "human",
}
_SUMMARY_CHARS: Final = 600

Result = tuple[str, dict[str, Any]]
Call = tuple[str, dict[str, Any]]


def _tokens(messages: Sequence[Message]) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages) // 4


def _last_user(messages: Sequence[Message]) -> str:
    return next((str(m["content"]) for m in reversed(messages) if m["role"] == "user"), "")


def _turn_results(messages: Sequence[Message]) -> list[Result]:
    """Resultados das tools chamadas depois da última mensagem do cliente."""
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
    names: dict[str, str] = {}
    results: list[Result] = []
    for m in messages[last_user + 1 :]:
        for call in m.get("tool_calls") or ():
            names[call["id"]] = call["function"]["name"]
        if m["role"] == "tool":
            name = names.get(m.get("tool_call_id", ""), str(m.get("name", "")))
            results.append((name, json.loads(m["content"])))
    return results


def _turn_calls(messages: Sequence[Message]) -> set[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
    for m in messages[last_user + 1 :]:
        for call in m.get("tool_calls") or ():
            fn = call["function"]
            seen.add((fn["name"], json.dumps(json.loads(fn["arguments"]), sort_keys=True)))
    return seen


def _clean(args: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in args.items() if v not in (None, "", [])}


class _Plan:
    """Plano de tool calls de um turno, a partir das intents e do que já voltou."""

    def __init__(
        self,
        intents: list[Intent],
        entities: dict[str, Any],
        text: str,
        results: list[Result],
        allowed: set[str],
    ) -> None:
        self.intents, self.e, self.text = intents, entities, text
        self.results, self.allowed = results, allowed
        self.norm = normalize_text(text)

    def _product_ids(self) -> list[int]:
        refs = [int(i) for i in self.e.get("referenced_product_ids") or []]
        found = [
            r["data"]["product_id"]
            for name, r in self.results
            if name == "get_product" and r.get("status") == "ok"
        ]
        return refs or found

    def _product_lookup(self) -> list[Call]:
        if self.e.get("referenced_product_ids"):
            return []
        name = self.e.get("product_name")
        return [("get_product", {"name_or_id": name})] if name else []

    def _store_topic(self) -> str:
        return next((t for t, pat in _TOPICS if re.search(pat, self.norm)), "open_now")

    def _catalog(self) -> list[Call]:
        e = self.e
        if e.get("product_name") and not e.get("category"):
            return self._product_lookup()
        keys = ("category", "min_price", "max_price", "strings", "electric")
        return [("search_products", _clean({k: e.get(k) for k in keys}))]

    def _detail(self) -> list[Call]:
        if self.e.get("referenced_product_ids"):
            return [("get_product", {"name_or_id": str(i)}) for i in self._product_ids()[:1]]
        return self._product_lookup()

    def _payment(self) -> list[Call]:
        ids = self._product_ids()
        if ids:
            return [("simulate_payment", {"product_ids": ids[:1]})]
        return self._product_lookup()

    def _shipping(self) -> list[Call]:
        e = self.e
        if not e.get("city"):
            return []
        ids = self._product_ids()
        if ids or not e.get("product_name"):
            return [("estimate_shipping", _clean({"city": e["city"], "product_ids": ids}))]
        return self._product_lookup()

    def _order(self) -> list[Call]:
        order_id = self.e.get("order_id")
        return [("get_order_status", {"order_id": order_id})] if order_id else []

    def _return(self) -> list[Call]:
        e = self.e
        if not e.get("order_id"):
            return []
        args = {
            "order_id": e["order_id"],
            "reason": e.get("return_reason") or "regret",
            "channel": e.get("channel"),
        }
        return [("check_return_eligibility", _clean(args))]

    def _ticket(self, intent: Intent) -> list[Call]:
        args = {
            "kind": _TICKET_KIND[intent],
            "summary": self.text[:500],
            "order_id": self.e.get("order_id"),
        }
        return [("open_ticket", _clean(args))]

    def _for_intent(self, intent: Intent) -> list[Call]:
        simple: dict[Intent, list[Call]] = {
            Intent.ACTIVE_PROMOTIONS: [("list_active_promotions", {})],
            Intent.STORE_INFO: [("get_store_info", {"topic": self._store_topic()})],
            Intent.WARRANTY: [("search_policies", {"query": "garantia " + self.text[:200]})],
        }
        if intent in simple:
            return simple[intent]
        if intent in _TICKET_KIND:
            return self._ticket(intent)
        handlers = {
            Intent.CATALOG_SEARCH: self._catalog,
            Intent.RECOMMENDATION: self._catalog,
            Intent.PRODUCT_DETAIL: self._detail,
            Intent.PAYMENT_SIMULATION: self._payment,
            Intent.SHIPPING: self._shipping,
            Intent.ORDER_STATUS: self._order,
            Intent.RETURN_EXCHANGE: self._return,
        }
        handler = handlers.get(intent)
        return handler() if handler else []

    def calls(self, already: set[tuple[str, str]]) -> list[Call]:
        wanted: list[Call] = []
        for intent in self.intents:
            wanted.extend(self._for_intent(intent))
        fresh: list[Call] = []
        for name, args in wanted:
            key = (name, json.dumps(args, sort_keys=True, default=str))
            if name in self.allowed and key not in already and (name, args) not in fresh:
                fresh.append((name, args))
        return fresh


class FakeLLM:
    """Implementação offline do :class:`LLMProvider`."""

    def __init__(self, catalog_names: Iterable[str] = ()) -> None:
        """Cria o fake.

        Args:
            catalog_names: Nomes de produtos; a primeira palavra de cada um vira "marca".
        """
        self._brands = frozenset(
            normalize_text(n.split()[0]) for n in catalog_names if n and n.split()
        )

    def structured[T: BaseModel](
        self,
        messages: list[Message],
        schema: type[T],
        *,
        tier: Tier = "small",
        temperature: float = 0.0,
    ) -> tuple[T, Usage]:
        """Roteia por palavras-chave (só o schema de roteamento é suportado)."""
        system = str(messages[0]["content"])
        order = _ORDER_LINE.search(system)
        decision = fake_nlu.analyze(
            _last_user(messages),
            brands=self._brands,
            focus=fake_nlu.parse_focus(system),
            order_id=int(order.group(1)) if order else None,
            pending=_PENDING_LINE in system,
        )
        try:
            parsed = schema.model_validate(decision.model_dump(mode="json"))
        except ValueError as exc:  # schema que o fake não conhece
            raise LLMError("FakeLLM só sabe produzir RouteDecision") from exc
        return parsed, Usage(_tokens(messages), len(decision.model_dump_json()) // 4, 0.0, 1)

    def complete(
        self,
        messages: list[Message],
        *,
        tier: Tier = "medium",
        temperature: float = 0.3,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Resume, pede tools conforme o plano ou escreve a resposta final."""
        system = str(messages[0]["content"])
        if system.startswith(SUMMARY_MARKER):
            text = self._summarize(system)
            return LLMResponse(text, (), Usage(_tokens(messages), len(text) // 4, 0.0, 1))
        intents_line = _INTENTS_LINE.search(system)
        intents = [
            Intent(i.strip())
            for i in (intents_line.group(1) if intents_line else "").split(",")
            if i.strip() and i.strip() != _SOCIAL
        ]
        entities_line = _ENTITIES_LINE.search(system)
        entities = json.loads(entities_line.group(1)) if entities_line else {}
        allowed = {t["function"]["name"] for t in tools or ()}
        results = _turn_results(messages)
        plan = _Plan(intents, entities, _last_user(messages), results, allowed)
        calls = plan.calls(_turn_calls(messages))
        if calls:
            base = len(_turn_calls(messages))
            tool_calls = tuple(
                ToolCall(f"call_{base + i}", name, args) for i, (name, args) in enumerate(calls)
            )
            return LLMResponse("", tool_calls, Usage(_tokens(messages), 20, 0.0, 1))
        text = self._answer(intents, results)
        return LLMResponse(text, (), Usage(_tokens(messages), len(text) // 4, 0.0, 1))

    @staticmethod
    def _summarize(prompt: str) -> str:
        said = [ln[9:] for ln in prompt.splitlines() if ln.startswith("Cliente: ")]
        text = "Cliente perguntou: " + " | ".join(said)
        return text[:_SUMMARY_CHARS]

    @staticmethod
    def _answer(intents: list[Intent], results: list[Result]) -> str:
        if not results:
            asks = [_ASKS[i] for i in intents if i in _ASKS]
            return asks[0] if asks else _SUPPORT_TEXT
        body = render_answer(results, complaint=Intent.COMPLAINT in intents)
        missing_city = Intent.SHIPPING in intents and all(
            n != "estimate_shipping" for n, _ in results
        )
        if missing_city:
            return body.removesuffix("\n\n" + CLOSING) + "\n\n" + _ASKS[Intent.SHIPPING]
        return body
