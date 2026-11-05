"""Logs estruturados: JSON de uma linha por evento, sem mensagem de cliente nem PII.

O ``ChatService`` registra ``mensagem sessao=... texto=<mascarado>``: em JSON esse evento é
descartado (o texto do cliente nunca sai do processo) e todo ``sessao=<id>`` vira hash curto.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import IO, Any

request_id_var: ContextVar[str] = ContextVar("request_id", default="")
_MARK = "_emporio_handler"
_SESSION = re.compile(r"sessao=(\S+)")
_DROP = ("mensagem sessao=",)
_EXTRA = ("event", "method", "route", "status", "latency_ms", "session")


def session_hash(session_id: str) -> str:
    """Hash curto e estável do id de sessão (correlaciona logs sem expor o id)."""
    return hashlib.sha256(session_id.encode()).hexdigest()[:12]


def _redact(text: str) -> str:
    return _SESSION.sub(lambda m: f"sessao={session_hash(m.group(1))}", text)


class _DropClientText(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not str(record.msg).startswith(_DROP)


class JsonFormatter(logging.Formatter):
    """Uma linha JSON por registro, com ``request_id`` do contexto e campos extras de allowlist."""

    def format(self, record: logging.LogRecord) -> str:
        """Serializa o registro."""
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": _redact(record.getMessage()),
        }
        if rid := request_id_var.get():
            payload["request_id"] = rid
        for key in _EXTRA:
            if (value := getattr(record, key, None)) is not None:
                payload[key] = value
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(fmt: str, level: str, stream: IO[str] | None = None) -> None:
    """Instala (idempotente) o handler do formato escolhido no logger raiz."""
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _MARK, False)]:
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    setattr(handler, _MARK, True)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
        handler.addFilter(_DropClientText())
        # O access log do uvicorn duplicaria o do app (que já traz request_id e latência).
        logging.getLogger("uvicorn.access").disabled = True
        for name in ("uvicorn", "uvicorn.error"):
            logging.getLogger(name).handlers.clear()
            logging.getLogger(name).propagate = True
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)
    root.setLevel(level.upper())
