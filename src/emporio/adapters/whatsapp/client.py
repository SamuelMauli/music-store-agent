"""Envio de texto pela Graph API do WhatsApp Cloud (``/{phone_number_id}/messages``)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import httpx

from emporio.domain.errors import MessageDeliveryError

log = logging.getLogger("emporio.whatsapp")

GRAPH_BASE_URL = "https://graph.facebook.com"
MAX_TEXT_CHARS = 4096  # limite do corpo de texto do WhatsApp
# 429 e 503 são recusas explícitas (nada foi processado); 502/504 são ambíguos como o timeout
# de leitura (o gateway pode ter repassado a mensagem), então não se repete.
_RETRY_STATUS = frozenset({429, 503})
# Só falhas em que a requisição comprovadamente não chegou à Meta: um timeout de leitura é
# ambíguo (a mensagem pode ter saído) e repetir duplicaria o texto para o cliente.
_RETRY_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)


class GraphApiSender:
    """:class:`~emporio.ports.MessageSender` sobre a Graph API.

    O token vai no cabeçalho ``Authorization`` e nunca em log; o destinatário só aparece
    em log depois de mascarado pelo chamador (aqui nem o hash é conhecido).
    """

    def __init__(
        self,
        access_token: str,
        phone_number_id: str,
        *,
        api_version: str = "v21.0",
        base_url: str = GRAPH_BASE_URL,
        timeout: float = 10.0,
        max_attempts: int = 3,
        backoff: float = 0.5,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Cria o cliente.

        Args:
            access_token: Token de acesso (System User) da Cloud API.
            phone_number_id: Id do número remetente na Meta.
            api_version: Versão da Graph API (configurável; premissa, não verificada ao vivo).
            base_url: Raiz da Graph API.
            timeout: Limite por tentativa, em segundos.
            max_attempts: Tentativas totais em falha transitória.
            backoff: Espera base entre tentativas (dobra a cada uma).
            transport: Transporte httpx (testes injetam um falso, sem rede).
            sleep: Espera injetável.
        """
        self._url = f"{base_url}/{api_version}/{phone_number_id}/messages"
        self._attempts, self._backoff, self._sleep = max_attempts, backoff, sleep
        self._http = httpx.Client(
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {access_token}"},
        )

    def close(self) -> None:
        """Fecha o cliente HTTP."""
        self._http.close()

    def send_text(self, to: str, body: str) -> None:
        """Envia o texto (cortado em 4096 caracteres).

        Raises:
            MessageDeliveryError: Rejeição definitiva (4xx) ou tentativas esgotadas.
        """
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"preview_url": False, "body": body[:MAX_TEXT_CHARS]},
        }
        reason = "sem tentativa"
        for attempt in range(1, self._attempts + 1):
            try:
                response = self._http.post(self._url, json=payload)
            except _RETRY_ERRORS as exc:
                reason = type(exc).__name__
            except httpx.HTTPError as exc:
                raise MessageDeliveryError(f"falha de transporte: {type(exc).__name__}") from exc
            else:
                if response.is_success:
                    return
                reason = f"HTTP {response.status_code}"
                if response.status_code not in _RETRY_STATUS:
                    raise MessageDeliveryError(f"Graph API recusou o envio ({reason})")
            if attempt < self._attempts:
                log.warning("envio WhatsApp falhou (%s); tentativa %s", reason, attempt)
                self._sleep(self._backoff * 2 ** (attempt - 1))
        raise MessageDeliveryError(
            f"Graph API indisponível após {self._attempts} tentativas ({reason})"
        )
