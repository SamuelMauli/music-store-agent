"""Fatos de política que vêm da ferramenta e não podem depender da paráfrase do modelo."""

from __future__ import annotations

import json
import re
from typing import Any, Final

from emporio.domain.text import normalize_text

_YES: Final = re.compile(
    r"\b(da para devolver|pode devolver|elegivel|e possivel devolver|tem direito)\b"
)
_NO: Final = re.compile(
    r"\b(nao e elegivel|nao da para devolver|nao pode devolver|fora do prazo)\b"
)


def _last_return(traces: list[dict[str, Any]]) -> dict[str, Any] | None:
    for trace in reversed(traces):
        if trace["name"] == "check_return_eligibility" and trace["status"] == "ok":
            data = json.loads(trace["payload"]).get("data")
            return data if isinstance(data, dict) else None
    return None


def _statement(data: dict[str, Any]) -> str:
    if not data["eligible"]:
        return f"Pela regra {data['rule']}, esse caso não é elegível para devolução."
    parts = [f"Pela regra {data['rule']}, dá para devolver."]
    if data.get("window"):
        parts.append(f"Você está no {data['window']}.")
    if data.get("deadline"):
        parts.append(f"Prazo até {data['deadline']}.")
    return " ".join(parts)


def with_return_verdict(answer: str, traces: list[dict[str, Any]]) -> str:
    """Garante veredito e janela (``dia X de Y``) na resposta, copiados do resultado da tool.

    Se o modelo já afirmou o veredito e citou a janela, a resposta segue intacta; senão, a
    frase determinística vem na frente.
    """
    data = _last_return(traces)
    if data is None:
        return answer
    text = normalize_text(answer)
    said_verdict = bool((_YES if data["eligible"] else _NO).search(text))
    said_window = not data.get("window") or normalize_text(str(data["window"])) in text
    if said_verdict and said_window:
        return answer
    return f"{_statement(data)} {answer}".strip()
