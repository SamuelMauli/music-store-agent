"""Fatos que vêm da ferramenta e não podem depender da paráfrase do modelo."""

from __future__ import annotations

import json
import re
from typing import Any, Final

from emporio.domain.text import normalize_text

_YES: Final = re.compile(
    r"\b(da (sim )?para devolver|pode (sim )?devolver|pode (sim )?ser devolvid\w*|elegivel|"
    r"e possivel devolver|tem direito|dentro do prazo)\b"
)
_NO: Final = re.compile(
    r"\b(nao (e|eh) elegivel|nao da para devolver|nao pode devolver|nao pode ser devolvid\w*|"
    r"nao e possivel devolver|fora do prazo)\b"
)
_WINDOW: Final = re.compile(r"dia (\d+) de (\d+)")
_ENDED: Final = re.compile(
    r"\b(encerrad\w*|encerrou|nao esta (mais )?(valendo|vigente|ativa|em promocao)|"
    r"nao (ha|existe|tem) (mais )?(a )?promocao|nao vale mais|acabou|terminou|finalizad\w*)\b"
)


_CLOSING_MAX_CHARS: Final = 90


def _append(answer: str, sentence: str) -> str:
    """Acrescenta ``sentence`` ao corpo da resposta, antes da pergunta de fechamento.

    O modelo costuma encerrar com um parágrafo curto do tipo "Posso ajudar em mais alguma
    coisa?". Um fato acrescentado depois dele fica fora de ordem para o cliente, então o fato
    entra no fim do parágrafo anterior; sem fechamento, vai para o fim da resposta.

    Args:
        answer: Resposta já redigida.
        sentence: Frase determinística a acrescentar.

    Returns:
        A resposta com a frase no lugar certo.
    """
    body, sep, closing = answer.rstrip().rpartition("\n\n")
    is_closing = closing.endswith("?") and len(closing) <= _CLOSING_MAX_CHARS
    if not sep or not body.strip() or not is_closing:
        return f"{answer} {sentence}".strip()
    return f"{body.rstrip()} {sentence}\n\n{closing}"


def _last_ok(traces: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for trace in reversed(traces):
        if trace["name"] == name and trace["status"] == "ok":
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


def _said_window(data: dict[str, Any], text: str) -> bool:
    """A janela vem literal ("dia 4 de 7") ou em forma equivalente ("restam 3 dias")."""
    window = str(data.get("window") or "")
    if not window or normalize_text(window) in text:
        return True
    if (m := _WINDOW.search(normalize_text(window))) is None:
        return False
    left = int(m.group(2)) - int(m.group(1))
    return bool(
        re.search(rf"\b(restam|resta|faltam|falta) {left} dias?\b|\b{left} dias? restantes?", text)
    )


def with_return_verdict(answer: str, traces: list[dict[str, Any]]) -> str:
    """Garante veredito e janela (``dia X de Y``) na resposta, copiados do resultado da tool.

    Reconhece formulações equivalentes do modelo ("pode ser devolvido", "restam 3 dias") para
    não duplicar a informação. Se o veredito faltar, a frase determinística vem na frente; se
    só a janela faltar, só ela é acrescentada ao fim.
    """
    data = _last_ok(traces, "check_return_eligibility")
    if data is None:
        return answer
    text = normalize_text(answer)
    if data["eligible"]:
        said_verdict = bool(_YES.search(text)) and not _NO.search(text)
    else:
        said_verdict = bool(_NO.search(text))
    if not said_verdict:
        return f"{_statement(data)} {answer}".strip()
    if _said_window(data, text):
        return answer
    extra = [f"Você está no {data['window']}."]
    if data.get("deadline") and str(data["deadline"]) not in answer:
        extra.append(f"Prazo até {data['deadline']}.")
    return _append(answer, " ".join(extra))


def ended_promotions(traces: list[dict[str, Any]]) -> list[str]:
    """Campanhas encerradas do produto consultado (``get_product``) sem promoção ativa."""
    data = _last_ok(traces, "get_product")
    if data is None or data.get("promotion"):
        return []
    return [str(n) for n in data.get("ended_promotions") or []]


def with_promotion_status(answer: str, traces: list[dict[str, Any]]) -> str:
    """Se a tool indica promoção encerrada, a resposta diz isso explicitamente."""
    ended = ended_promotions(traces)
    text = re.sub(r"nao (foi |esta )?encerrad\w*", " ", normalize_text(answer))
    if not ended or _ENDED.search(text):
        return answer
    names = " e ".join(ended)
    note = f"A promoção {names} deste produto foi encerrada e não está mais valendo."
    return _append(answer, note)
