"""Canal WhatsApp: entrada do webhook, ponte com o agente e handoff humano com fila de operador.

Caso de uso puro (sem FastAPI/httpx): fala com o mundo por portas (``MessageSender``,
``HandoffStore``, ``MessageDeduplicator``, ``RateLimiter``) e com o agente por ``ChatService``.
O número do cliente só existe em claro dentro do adaptador de envio e do ``HandoffStore``;
em log e na API aparece apenas o id opaco (HMAC com pepper, ver ``domain.handoff``).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from emporio.agent.service import ChatService
from emporio.domain.errors import (
    HandoffConflictError,
    HandoffNotFoundError,
    MessageDeliveryError,
)
from emporio.domain.handoff import ClaimResult, HandoffConversation, contact_id
from emporio.domain.pii import mask_pii
from emporio.guardrails.rate_limit import RateLimitedError
from emporio.ports import HandoffStore, MessageDeduplicator, MessageSender, RateLimiter

log = logging.getLogger("emporio.whatsapp")

MEDIA_REPLY = (
    "Por enquanto só consigo ler mensagens de texto. Pode escrever sua dúvida? "
    "Se preferir, peça para falar com um atendente."
)
TOO_LONG_REPLY = "Sua mensagem ficou longa demais. Pode resumir em poucas linhas?"
FAILURE_REPLY = "Tive um problema técnico agora. Tente novamente em instantes, por favor."
HANDOFF_NOTICE = (
    "Encaminhei sua conversa para a nossa equipe. A partir de agora um atendente "
    "responde por aqui e eu fico em silêncio até ele concluir."
)
RETURN_NOTICE = "Atendimento humano concluído. Voltei a ajudar por aqui: é só mandar sua dúvida."
MEDIA_PLACEHOLDER = "[mensagem não textual: {kind}]"
_IGNORED_KINDS = frozenset({"reaction", "system", "request_welcome", "ephemeral"})
_PREVIEW_CHARS = 80


@dataclass(frozen=True, slots=True)
class InboundMessage:
    """Mensagem recebida do cliente, já extraída do envelope da Meta."""

    message_id: str
    sender: str
    kind: str
    text: str | None


def parse_webhook(payload: object, phone_number_id: str) -> list[InboundMessage]:
    """Extrai as mensagens do cliente do ``POST`` da Meta, ignorando todo o resto.

    ``statuses`` (entregue/lido), reações e eventos de outro número não geram mensagem.
    Tolerante: formato inesperado devolve lista vazia em vez de levantar.
    """
    found: list[InboundMessage] = []
    for entry in _items(payload, "entry"):
        for change in _items(entry, "changes"):
            value = change.get("value")
            if not isinstance(value, dict):
                continue
            metadata = value.get("metadata")
            if not isinstance(metadata, dict) or metadata.get("phone_number_id") != phone_number_id:
                continue
            for raw in _items(value, "messages"):
                message = _message(raw)
                if message is not None:
                    found.append(message)
    return found


def _items(container: object, key: str) -> list[dict[str, Any]]:
    items = container.get(key) if isinstance(container, dict) else None
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def _message(raw: dict[str, Any]) -> InboundMessage | None:
    message_id, sender, kind = raw.get("id"), raw.get("from"), raw.get("type")
    if not (isinstance(message_id, str) and isinstance(sender, str) and isinstance(kind, str)):
        return None
    if not (sender.isdigit() and 5 <= len(sender) <= 20) or not message_id or len(message_id) > 256:
        return None
    if kind in _IGNORED_KINDS:
        return None
    text: str | None = None
    if kind == "text":
        body = raw.get("text")
        text = body.get("body") if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip():
            return None
    return InboundMessage(message_id, sender, kind, text)


class WhatsAppChannel:
    """Processa mensagens do cliente e as ações de operador sobre conversas em handoff."""

    def __init__(
        self,
        service: ChatService,
        sender: MessageSender,
        store: HandoffStore,
        dedup: MessageDeduplicator,
        limiter: RateLimiter,
        *,
        pepper: str,
        max_message_chars: int,
        handoff_ttl_seconds: float,
        workers: int = 4,
        queue_size: int = 64,
        conversation_queue: int = 10,
        now: Callable[[], float] = time.time,
    ) -> None:
        """Cria o canal.

        Args:
            service: Serviço de conversa do agente.
            sender: Saída de mensagens (Graph API).
            store: Estado de handoff e fila.
            dedup: Idempotência por id de mensagem.
            limiter: Limite por número (chave = id opaco do contato).
            pepper: Pepper do HMAC do telefone.
            max_message_chars: Teto do texto aceito do cliente.
            handoff_ttl_seconds: Janela sem ação do operador até voltar ao bot.
            workers: Threads dedicadas ao processamento (teto; o pool do servidor fica livre).
            queue_size: Mensagens aceitas além dos workers; acima disso descarta com log.
            conversation_queue: Teto de mensagens em andamento por conversa (rodando + na fila);
                o excedente de um número é descartado antes de ocupar a fila global.
            now: Relógio de parede (epoch); injetável nos testes.
        """
        self._service, self._sender, self._store = service, sender, store
        self._dedup, self._limiter, self._pepper = dedup, limiter, pepper
        self._max_chars, self._ttl, self._now = max_message_chars, handoff_ttl_seconds, now
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="wa-worker")
        self._capacity = workers + queue_size
        self._per_conversation = max(1, conversation_queue)
        self._closed = False
        self._mutex = threading.Lock()
        self._idle = threading.Condition(self._mutex)
        self._inflight = 0  # aceitas e ainda não terminadas (rodando + na fila)
        # conversa -> mensagens que chegaram enquanto o worker dela estava ocupado; a presença
        # da chave é o "trinco" da conversa (sem bloquear thread nenhuma esperando por ela).
        self._waiting: dict[str, deque[InboundMessage]] = {}

    def conversation_id(self, phone: str) -> str:
        """Id opaco do contato."""
        return contact_id(self._pepper, phone)

    # --- entrada assíncrona (pool dedicado, fila limitada) --------------------------------
    def submit(self, message: InboundMessage) -> bool:
        """Agenda a mensagem num worker sem bloquear quem chama; ``False`` se descartada.

        Antes de ocupar vaga, a mensagem passa por deduplicação e limite por número (quem
        repete ou abusa não consome a fila dos outros). Mensagens da mesma conversa são
        processadas em ordem, uma por vez, pelo mesmo worker; as que chegam no meio ficam numa
        fila da conversa (com teto), sem prender thread. Com ``workers + queue_size`` mensagens
        em andamento, as excedentes são descartadas com log. Devolve ``True`` também para
        reentrega ignorada (já tratada), pois não há o que perder.
        """
        conv_id = self.conversation_id(message.sender)
        if (rejected := self._screen(conv_id, message)) is not None:
            return rejected == "duplicate"
        slot = self._reserve(conv_id, message)
        if slot is None:
            return False
        if slot == "queued":
            return True
        try:
            self._pool.submit(self._drain, conv_id, message)
        except RuntimeError:  # pool já encerrado (shutdown)
            self._finish(conv_id, discard=True)
            log.warning("canal encerrando; mensagem descartada contato=%s", conv_id[:12])
            return False
        return True

    def _reserve(self, conv_id: str, message: InboundMessage) -> str | None:
        """Reserva vaga: ``start`` (livre), ``queued`` (fila da conversa) ou ``None``."""
        with self._mutex:
            waiting = self._waiting.get(conv_id)
            full = waiting is not None and len(waiting) + 1 >= self._per_conversation
            if self._closed or full or self._inflight >= self._capacity:
                reason = "encerrando" if self._closed else "fila cheia"
                log.warning("mensagem descartada (%s) contato=%s", reason, conv_id[:12])
                return None
            self._inflight += 1
            if waiting is None:
                self._waiting[conv_id] = deque()
                return "start"
            waiting.append(message)  # o worker da conversa pega em seguida
            return "queued"

    def _screen(self, conv_id: str, message: InboundMessage) -> str | None:
        """Deduplicação e limite por número, ANTES de ocupar vaga; ``None`` = segue.

        Devolve o motivo da rejeição (``duplicate``, ``rate_limited`` ou ``error``).
        """
        try:
            if not self._dedup.first_seen(message.message_id, self._now()):
                log.info("reentrega ignorada contato=%s", conv_id[:12])
                return "duplicate"
            self._limiter.check(conv_id)
        except RateLimitedError:
            log.warning("limite por número excedido contato=%s", conv_id[:12])
            return "rate_limited"
        except Exception:
            log.exception("falha ao triar mensagem contato=%s", conv_id[:12])
            return "error"
        return None

    def _drain(self, conv_id: str, first: InboundMessage) -> None:
        """Roda no worker: processa a mensagem e depois as que se acumularam na conversa."""
        current: InboundMessage | None = first
        while current is not None:
            try:
                self.handle(current)
            except Exception as exc:  # defesa: nada pode deixar a conversa presa
                log.error("falha inesperada no worker tipo=%s", type(exc).__name__)
            finally:
                current = self._finish(conv_id)

    def _finish(self, conv_id: str, *, discard: bool = False) -> InboundMessage | None:
        """Dá baixa em uma mensagem; devolve a próxima da conversa ou libera o trinco."""
        with self._mutex:
            self._inflight -= 1
            waiting = self._waiting[conv_id]
            if self._closed or discard:  # as que sobraram na fila da conversa não vão rodar
                self._inflight -= len(waiting)
                waiting.clear()
            nxt = waiting.popleft() if waiting else None
            if nxt is None:
                del self._waiting[conv_id]
            if self._inflight == 0:
                self._idle.notify_all()
            return nxt

    def drain(self, timeout: float = 30.0) -> bool:
        """Espera o que está em andamento terminar (testes e shutdown); ``False`` no timeout."""
        with self._idle:
            return self._idle.wait_for(lambda: self._inflight == 0, timeout)

    def close(self) -> None:
        """Encerra o pool: o que está rodando termina, o que está na fila é descartado."""
        with self._mutex:
            self._closed = True  # novas entradas recusadas; workers param de esvaziar filas
        self._pool.shutdown(wait=True, cancel_futures=True)
        with self._mutex:  # tarefas canceladas nunca rodaram _finish: zera a contabilidade
            self._inflight = 0
            self._waiting.clear()
            self._idle.notify_all()

    # --- cliente -> bot ou fila ---------------------------------------------------------
    def handle(self, message: InboundMessage) -> None:
        """Processa uma mensagem já triada; nunca levanta (roda depois do 200 ao webhook)."""
        conv_id = self.conversation_id(message.sender)
        try:
            self._route(conv_id, message)
        except Exception:
            log.exception("falha ao processar mensagem contato=%s", conv_id[:12])
            self._safe_send(message.sender, FAILURE_REPLY, conv_id)

    def _route(self, conv_id: str, message: InboundMessage) -> None:
        now = self._now()
        if self._store.is_human(conv_id, now):  # o bot PARA: só enfileira para o operador
            text = message.text or MEDIA_PLACEHOLDER.format(kind=message.kind)
            self._store.append(conv_id, "in", text, now)
            log.info("mensagem enfileirada contato=%s", conv_id[:12])
            return
        if message.text is None:
            self._safe_send(message.sender, MEDIA_REPLY, conv_id)
            return
        if len(message.text) > self._max_chars:
            self._safe_send(message.sender, TOO_LONG_REPLY, conv_id)
            return
        session_id = f"wa-{conv_id}"
        result = self._service.chat(session_id, message.text, uuid.uuid4().hex)
        answer = result.answer
        if self._wants_human(result.handoff, session_id) and self._store.enter(
            conv_id, message.sender, now, now + self._ttl
        ):
            self._store.append(conv_id, "in", message.text, now)
            answer = f"{answer}\n\n{HANDOFF_NOTICE}".strip()
            log.info("handoff iniciado contato=%s", conv_id[:12])
        self._sender.send_text(message.sender, answer)

    def _wants_human(self, handoff: bool, session_id: str) -> bool:
        """Handoff do grafo (falhas, guardas, verificação) ou chamado ``human`` aberto no turno."""
        if handoff:
            return True
        traces = self._service.snapshot(session_id).get("tool_traces") or []
        return any(
            t["name"] == "open_ticket" and t["status"] == "ok" and t["args"].get("kind") == "human"
            for t in traces
        )

    def _safe_send(self, to: str, body: str, conv_id: str) -> None:
        try:
            self._sender.send_text(to, body)
        except MessageDeliveryError:
            log.exception("envio falhou contato=%s", conv_id[:12])

    # --- operador -----------------------------------------------------------------------
    def queue(self) -> list[HandoffConversation]:
        """Conversas em handoff, da que espera há mais tempo à mais recente."""
        return self._store.list_active(self._now())

    @staticmethod
    def preview(conversation: HandoffConversation) -> str:
        """Última mensagem do cliente, truncada e sem PII (a lista não vaza dado pessoal)."""
        inbound = [m.text for m in conversation.messages if m.direction == "in"]
        return mask_pii(inbound[-1])[:_PREVIEW_CHARS] if inbound else ""

    def detail(self, conv_id: str) -> HandoffConversation:
        """Conversa com o histórico.

        Raises:
            HandoffNotFoundError: Não está em handoff.
        """
        conversation = self._store.get(conv_id, self._now())
        if conversation is None:
            raise HandoffNotFoundError(conv_id)
        return conversation

    def claim(self, conv_id: str, operator: str) -> HandoffConversation:
        """Operador assume a conversa (e renova a expiração).

        Raises:
            HandoffNotFoundError: Não está em handoff.
            HandoffConflictError: Outro operador já assumiu.
        """
        now = self._now()
        outcome = self._store.claim(conv_id, operator, now, now + self._ttl)
        if outcome is ClaimResult.NOT_FOUND:
            raise HandoffNotFoundError(conv_id)
        if outcome is ClaimResult.TAKEN:
            raise HandoffConflictError("conversa já assumida por outro operador")
        log.info("handoff assumido contato=%s", conv_id[:12])
        return self.detail(conv_id)

    def reply(self, conv_id: str, operator: str, text: str) -> None:
        """Operador responde ao cliente pelo WhatsApp.

        Raises:
            HandoffNotFoundError: Não está em handoff.
            HandoffConflictError: A conversa não é deste operador.
            MessageDeliveryError: A Graph API não aceitou o envio (nada é registrado).
        """
        conversation = self.detail(conv_id)
        now = self._now()
        if not self._store.extend(conv_id, operator, now + self._ttl, now):
            raise HandoffConflictError("assuma a conversa antes de responder")
        self._sender.send_text(conversation.phone, text)
        self._store.append(conv_id, "out", text, now)
        log.info("resposta de operador enviada contato=%s", conv_id[:12])

    def release(self, conv_id: str, operator: str) -> None:
        """Devolve a conversa ao bot e avisa o cliente.

        Raises:
            HandoffNotFoundError: Não está em handoff.
            HandoffConflictError: A conversa é de outro operador.
        """
        result, conversation = self._store.release(conv_id, operator, self._now())
        if result is ClaimResult.TAKEN:
            raise HandoffConflictError("conversa assumida por outro operador")
        if conversation is None:
            raise HandoffNotFoundError(conv_id)
        log.info("handoff devolvido ao bot contato=%s", conv_id[:12])
        self._safe_send(conversation.phone, RETURN_NOTICE, conv_id)
