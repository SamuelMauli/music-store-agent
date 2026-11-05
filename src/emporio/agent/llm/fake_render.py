"""Respostas do LLM fake: texto montado só com o que as tools devolveram (nada inventado)."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Final

Payload = dict[str, Any]
CLOSING: Final = "Posso ajudar em mais alguma coisa?"
_MAX_POLICY_CHARS: Final = 320


def _lines(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items)


def _product_line(p: Payload) -> str:
    if p.get("original_price"):
        return (
            f"{p['name']}: de {p['original_price']} por {p['price']} ({p['discount']} off), "
            f"{p['availability']}"
        )
    return f"{p['name']}: {p['price']} ({p['availability']})"


def _search_products(d: Payload) -> str:
    items, total = d["items"], d["total_matches"]
    if not items:
        return d.get("note") or "Não achei nada com esses filtros."
    head = f"Encontrei {total} {'opção' if total == 1 else 'opções'} nesse filtro."
    if total > len(items):
        head += f" Estas são as {len(items)} mais em conta:"
    body = _lines([_product_line(p) for p in items])
    tail = "Quer afinar por cordas, captação ou faixa de preço?" if total > len(items) else ""
    return "\n".join(part for part in (head, body, tail) if part)


def _get_product(d: Payload) -> str:
    parts = [f"O {d['name']} sai por {d['price']}."]
    if d.get("original_price"):
        parts.append(
            f"Está em promoção ({d.get('promotion', 'campanha ativa')}): de {d['original_price']} "
            f"por {d['price']}, {d['discount']} de desconto."
        )
    else:
        parts.append("Sem promoção ativa no momento.")
    stock = d["stock_quantity"]
    parts.append(f"Estoque: {stock} em estoque." if stock else f"Estoque: {d['availability']}.")
    parts.append(f"No PIX: {d['pix_price']}." if d.get("pix_price") else d.get("pix_note", ""))
    parts.append(f"Parcelamento: {d['max_installments']}.")
    return " ".join(p for p in parts if p)


def _compare(d: Payload) -> str:
    rows = [f"{p['name']}: {p['price']}, {p['availability']}" for p in d["products"]]
    return "\n".join(["Comparando:", _lines(rows), d.get("note") or ""]).strip()


def _alternatives(d: Payload) -> str:
    if not d["items"]:
        return str(d.get("note", "Sem alternativas disponíveis nessa faixa."))
    return f"Como alternativas disponíveis perto de {d['reference_price']}:\n" + _lines(
        [_product_line(p) for p in d["items"]]
    )


def _promotions(d: Payload) -> str:
    if not d["promotions"]:
        return "Não temos promoções ativas agora."
    rows = [
        f"{p['name']}: de {p['original_price']} por {p['final_price']} ({p['discount']} off)"
        for p in d["promotions"]
    ]
    return "\n".join(["Promoções ativas agora:", _lines(rows), d.get("note", "")]).strip()


def _payment(d: Payload) -> str:
    parts = [f"Total: {d['total']}."]
    parts.append(f"No PIX: {d['pix_total']}." if d.get("pix_total") else d.get("pix_note", ""))
    parts.append(f"No cartão: até {d['max_installments']} sem juros.")
    parts.append(f"Combinar formas de pagamento: {d['combined_payment']}.")
    return " ".join(p for p in parts if p)


def _shipping(d: Payload) -> str:
    if d["region"] == "metropolitana":
        fee = d.get("fee")
        if fee:
            cost = "é grátis" if fee == "grátis" else f"é {fee}"
            return f"Para a região de Campo Grande o frete {cost}. Prazo: {d['delivery']}."
        return f"Em Campo Grande: {d['rule']}. Prazo: {d['delivery']}."
    modes = [f"{m['name']}: {m['deadline']}" for m in d["modes"]]
    return "\n".join(
        ["Para outras cidades enviamos por:", _lines(modes), d.get("note", "")]
    ).strip()


def _store_info(d: Payload) -> str:
    topic = d["topic"]
    if topic == "address":
        return f"Ficamos na {d['address']}."
    if topic == "contacts":
        return f"Telefone {d['phone']}, WhatsApp {d['whatsapp']}, e-mail {d['email']}."
    if topic == "hours":
        return f"Nosso horário: {d['hours']}"
    if d["open_now"]:
        return f"Estamos abertos agora ({d['now']})."
    return f"Estamos fechados agora ({d['now']}). Voltamos {d['next_opening']}."


def _policies(d: Payload) -> str:
    out = []
    for s in d["sections"][:2]:
        text = re.sub(r"\s+", " ", s["text"]).strip()
        if len(text) > _MAX_POLICY_CHARS:
            text = text[:_MAX_POLICY_CHARS].rsplit(" ", 1)[0] + "..."
        out.append(f"Pelo manual da loja ({s['section']} {s['title']}): {text}")
    return "\n".join(out)


def _order(d: Payload) -> str:
    parts = [f"Seu pedido {d['order_id']} está {d['status']}."]
    if d.get("estimated_delivery"):
        parts.append(f"Previsão de entrega: {d['estimated_delivery']}.")
    if d.get("tracking_code"):
        parts.append(f"Rastreio: {d['tracking_code']}.")
    if d["is_delayed"]:
        parts.append(
            f"Sinto muito: ele está {d['delay_note']}. Quer que eu abra um chamado para a equipe?"
        )
    return " ".join(parts)


def _return(d: Payload) -> str:
    if not d["eligible"]:
        return f"Pela regra {d['rule']}, esse caso não é elegível. {d.get('detail', '')}".strip()
    parts = [f"Pela regra {d['rule']}, dá para devolver."]
    if d.get("window"):
        parts.append(f"Você está no {d['window']}.")
    if d.get("deadline"):
        parts.append(f"Prazo até {d['deadline']}.")
    if d.get("days_left"):
        parts.append(f"{d['days_left'].capitalize()}.")
    if d.get("refund"):
        parts.append(f"Reembolso: {d['refund']}.")
    if d.get("freight"):
        parts.append(f"Frete da devolução: {d['freight']}.")
    if d.get("estimate_note"):
        parts.append(f"Obs.: {d['estimate_note']}.")
    return " ".join(parts)


def _ticket(d: Payload) -> str:
    again = " (já estava aberto)" if d.get("already_opened") else ""
    return (
        f"Registrei seu chamado{again}. Protocolo {d['protocol']}. "
        f"A equipe retorna em até {d['sla']}."
    )


_OK: Final[dict[str, Callable[[Payload], str]]] = {
    "search_products": _search_products,
    "get_product": _get_product,
    "compare_products": _compare,
    "find_alternatives": _alternatives,
    "list_active_promotions": _promotions,
    "simulate_payment": _payment,
    "estimate_shipping": _shipping,
    "get_store_info": _store_info,
    "search_policies": _policies,
    "get_order_status": _order,
    "check_return_eligibility": _return,
    "open_ticket": _ticket,
}


def _status_text(payload: Payload) -> str:
    status = payload["status"]
    if status == "not_found":
        similar = payload.get("similar") or []
        names = f" Parecidos: {', '.join(s['name'] for s in similar)}." if similar else ""
        return f"{payload['message']}{names}"
    if status == "ambiguous":
        options = payload.get("options") or []
        return str(payload["ask"]) + (f" ({' ou '.join(options)})" if options else "")
    if status == "policy_blocked":
        options = payload.get("options") or []
        extra = f" Posso te ajudar com: {', '.join(options)}." if options else ""
        return f"{payload['reason'].capitalize()}.{extra}"
    return "Tive um problema ao consultar agora. Quer que eu abra um chamado para a equipe?"


def render_tool(name: str, payload: Payload) -> str:
    """Texto de uma tool, a partir do payload exato que o LLM recebeu."""
    if payload.get("status") == "ok":
        return _OK[name](payload["data"])
    return _status_text(payload)


def render_answer(results: list[tuple[str, Payload]], *, complaint: bool = False) -> str:
    """Junta as tools do turno em uma resposta curta e termina oferecendo ajuda."""
    body = "\n\n".join(render_tool(name, payload) for name, payload in results)
    asks = any(p.get("status") in {"ambiguous", "policy_blocked"} for _, p in results)
    prefix = "Poxa, sinto muito por isso. " if complaint else ""
    return f"{prefix}{body}" if asks else f"{prefix}{body}\n\n{CLOSING}"
