"""Roteamento determinístico por palavras-chave, usado pelo LLM fake (CI e avaliação offline)."""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Final

from emporio.agent.intents import Entities, Intent, RouteDecision
from emporio.agent.tools.categories import resolve_category
from emporio.domain.text import normalize_text

I = Intent  # noqa: E741 - alias local para a tabela de regras
_SPLIT: Final = re.compile(
    r"[?!;]|\.(?=\s|$)|\s+e\s+(?=voc[eê]|quant|qual|quais|tem\b|abre|onde|como|pode|quero|aceit|existe)",
    re.IGNORECASE,
)
_MODEL: Final = re.compile(r"^(?=.*[a-z])(?=.*\d)[a-z0-9-]+$")
_INSTALLMENTS: Final = re.compile(r"^\d+x$")
_STOP: Final = frozenset(
    [
        "quanto",
        "custa",
        "preco",
        "valor",
        "o",
        "a",
        "os",
        "as",
        "um",
        "uma",
        "do",
        "da",
        "de",
        "no",
        "na",
        "tem",
        "estoque",
        "esta",
        "em",
        "e",
        "voces",
        "pix",
        "parcelas",
        "parcela",
        "ainda",
        "disponivel",
        "promocao",
        "para",
        "pra",
        "sai",
        "fica",
        "quero",
        "queria",
        "saber",
    ]
)
_STORE_HINTS: Final = re.compile(
    r"instrument|loja|musica|musical|tocar|som\b|banda|comprar|compra|produto|modelo"
)

# (intent, regex sobre o texto normalizado), do mais ao menos prioritário.
_RULES: Final[tuple[tuple[Intent, str], ...]] = (
    (I.HUMAN_HANDOFF, r"\b(atendente|humano|pessoa de verdade|falar com alguem|gerente)\b"),
    (I.PRIVACY_REQUEST, r"(excluir|apagar|deletar|remover) (meus |os )?dados|\blgpd\b|privacidade"),
    (
        I.COMPLAINT,
        r"\b(reclama\w*|absurdo|pessimo|horrivel|inaceitavel|indignad\w*|revoltad\w*|descaso)\b",
    ),
    (
        I.RETURN_EXCHANGE,
        r"\b(devolv\w*|devolucao|arrepend\w*|trocar|troca|estorno|reembolso|defeito|defeituos\w*"
        r"|quebrou|nao funciona)\b",
    ),
    (I.ORDER_STATUS, r"\b(pedido|rastre\w*|encomenda)\b"),
    (I.ACTIVE_PROMOTIONS, r"\b(promo\w*|oferta\w*|desconto\w*)\b"),
    (I.WARRANTY, r"\bgarantia\b"),
    (I.SHIPPING, r"\b(frete|envio|entregam|enviam|correios|prazo de entrega|entrega para)\b"),
    (
        I.STORE_INFO,
        r"\b(endereco|onde fica|localiza\w*|horario|abre|abrem|aberto|aberta|abertos|fecha|fecham"
        r"|telefone|whatsapp|contato|e-?mail)\b",
    ),
    (
        I.PAYMENT_SIMULATION,
        r"\b(pix|parcel\w*|\d+x|cartao|credito|debito|boleto|pagar|pagamento|a vista)\b",
    ),
    (
        I.RECOMMENDATION,
        r"\b(recomend\w*|indica\w*|sugest\w*|qual (o )?melhor|iniciante|presente|comecar)\b",
    ),
)
_ACCESSORY: Final = re.compile(
    r"\b(cabos?|palhetas?|pedais?|pedaleiras?|amplificador\w*|cases?|capas?|afinador\w*"
    r"|encordoamentos?|cordas?)\b"
)
_ALWAYS_ACCESSORY: Final = re.compile(
    r"\b(cabos?|palhetas?|pedais?|pedaleiras?|amplificador\w*|encordoamentos?)\b"
)
_GREETING: Final = re.compile(r"^(oi|ola|bom dia|boa tarde|boa noite|e ai|tudo bem|opa)\b")
_MAX_PRICE: Final = re.compile(
    r"(?:ate|menos de|abaixo de|no maximo|maximo de)\s*(?:r\$)?\s*"
    r"(\d[\d.]*)(?:,(\d+))?(?!\d*\s*x\b)"
)
_MIN_PRICE: Final = re.compile(
    r"(?:a partir de|acima de|mais de|minimo de)\s*(?:r\$)?\s*(\d[\d.]*)(?:,(\d+))?(?!\d*\s*x\b)"
)
_ORDER_ID: Final = re.compile(r"(?:pedido|ped)\s*(?:numero|num|n|no|#)?\s*(\d{1,6})\b|#(\d{1,6})\b")
_CITY: Final = re.compile(
    r"\b(?:para|pra|em|de)\s+((?:[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]*)(?:\s+(?:d[aeo]s?\s+)?[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]*)*)"
)
_FOCUS_LINE: Final = re.compile(r"id=(\d+) \| (.+?) \| (R\$ [\d.,]+)")
_DEICTIC: Final = re.compile(r"\b(esse|essa|este|esta|ele|dele|mais barat|mais car)")
_ORDINALS: Final = (("primeir", 0), ("segund", 1), ("terceir", 2))


@dataclass(frozen=True, slots=True)
class Focus:
    """Produto em foco da conversa."""

    product_id: int
    name: str
    price: float


def parse_focus(text: str) -> list[Focus]:
    """Lê as linhas ``id=95 | nome | R$ 2.199,00`` do contexto do prompt."""
    return [Focus(int(i), name, _money(price)) for i, name, price in _FOCUS_LINE.findall(text)]


def _money(text: str) -> float:
    return float(text.replace("R$", "").strip().replace(".", "").replace(",", "."))


def _amount(whole: str, cents: str | None) -> float:
    return float(whole.replace(".", "") + (f".{cents}" if cents else ""))


def _is_model(token: str, brands: Collection[str]) -> bool:
    if _INSTALLMENTS.match(token):
        return False
    return token in brands or bool(_MODEL.match(token))


def extract_product_name(clause: str, brands: Collection[str]) -> str | None:
    """Trecho do modelo citado: da primeira marca/modelo até uma palavra comum."""
    tokens = re.findall(r"[\wÀ-ÿ-]+", clause)
    for start, token in enumerate(tokens):
        if _is_model(normalize_text(token), brands):
            picked: list[str] = []
            for word in tokens[start : start + 4]:
                if normalize_text(word) in _STOP:
                    break
                picked.append(word)
            return " ".join(picked)
    return None


def _category_word(clause: str) -> str | None:
    for token in re.findall(r"[\wÀ-ÿ]+", clause):
        if resolve_category(token) is not None:
            return str(token)
    return None


def _reference(norm: str, focus: Sequence[Focus]) -> tuple[list[int], bool]:
    """Resolve "esse", "o mais barato", "o segundo" contra os produtos em foco."""
    if not focus:
        return [], False
    if re.search(r"\bmais barat", norm):
        return [min(focus, key=lambda f: f.price).product_id], False
    if re.search(r"\bmais car[oa]", norm):
        return [max(focus, key=lambda f: f.price).product_id], False
    for stem, index in _ORDINALS:
        if stem in norm and index < len(focus):
            return [focus[index].product_id], False
    if len(focus) == 1:
        return [focus[0].product_id], False
    return [], True


def _first_intents(norm: str, category: str | None, model: str | None) -> list[Intent]:
    for intent, pattern in _RULES:
        if re.search(pattern, norm):
            return [intent]
    if _ALWAYS_ACCESSORY.search(norm) or (_ACCESSORY.search(norm) and category is None):
        return [Intent.OUT_OF_SCOPE_ACCESSORY]
    if model and re.search(r"\b(quanto|preco|valor|custa|estoque|tem|disponivel)\b", norm):
        return [Intent.PRODUCT_DETAIL]
    if category or re.search(r"\b(mostre|mostra|quais|lista|opcoes|catalogo)\b", norm):
        return [Intent.CATALOG_SEARCH]
    if model:
        return [Intent.PRODUCT_DETAIL]
    return []


def _price_entities(norm: str) -> tuple[float | None, float | None]:
    high = _MAX_PRICE.search(norm)
    low = _MIN_PRICE.search(norm)
    return (
        _amount(low.group(1), low.group(2)) if low else None,
        _amount(high.group(1), high.group(2)) if high else None,
    )


def _return_reason(norm: str) -> str | None:
    if re.search(r"arrepend|desisti|nao quero mais|nao gostei", norm):
        return "regret"
    if re.search(r"defeito|quebrou|nao funciona|danific|riscad", norm):
        return "defect"
    if re.search(r"trocar|troca|outro modelo|outra cor|prefer", norm):
        return "preference"
    return None


def _channel(norm: str) -> str | None:
    if re.search(r"\b(online|site|internet|app)\b", norm):
        return "online"
    if re.search(r"na loja|presencial|loja fisica|fisicamente", norm):
        return "store"
    return None


def _city(clause: str) -> str | None:
    match = _CITY.search(clause)
    if match and not any(c.isdigit() for c in match.group(1)):
        return match.group(1)
    return None


def _clause_entities(clause: str, norm: str, brands: Collection[str]) -> dict[str, object]:
    low, high = _price_entities(norm)
    order = _ORDER_ID.search(norm)
    strings = "nylon" if "nylon" in norm else ("aço" if re.search(r"\baco\b", norm) else None)
    electric = (
        True
        if re.search(r"eletroacust|eletric|captacao", norm)
        else (False if "acustico" in norm else None)
    )
    found: dict[str, object] = {
        "category": _category_word(clause),
        "min_price": low,
        "max_price": high,
        "strings": strings,
        "electric": electric,
        "product_name": extract_product_name(clause, brands),
        "order_id": int(next(g for g in order.groups() if g)) if order else None,
        "city": _city(clause),
        "channel": _channel(norm),
        "return_reason": _return_reason(norm),
    }
    return {k: v for k, v in found.items() if v is not None}


def analyze(
    text: str,
    *,
    brands: Collection[str] = (),
    focus: Sequence[Focus] = (),
    order_id: int | None = None,
) -> RouteDecision:
    """Classifica a mensagem por regras, cláusula a cláusula (multi-intent).

    Args:
        text: Mensagem do cliente.
        brands: Primeiras palavras dos nomes do catálogo, normalizadas.
        focus: Produtos citados antes, para resolver referências.
        order_id: Pedido em atendimento.

    Returns:
        Decisão com confiança alta quando alguma regra casou, baixa caso contrário.
    """
    intents: list[Intent] = []
    entities: dict[str, object] = {}
    ambiguous_reference = False
    clauses = [c.strip() for c in _SPLIT.split(text) if c and c.strip()]
    for clause in clauses:
        norm = normalize_text(clause)
        found = _clause_entities(clause, norm, brands)
        clause_intents = _first_intents(
            norm, str(found.get("category") or "") or None, str(found.get("product_name") or "")
        )
        if not clause_intents and _GREETING.match(norm):
            continue
        for key, value in found.items():
            entities.setdefault(key, value)
        needs_product = set(clause_intents) & {
            Intent.PRODUCT_DETAIL,
            Intent.PAYMENT_SIMULATION,
            Intent.SHIPPING,
        }
        if needs_product and "product_name" not in found and "category" not in found:
            ids, ambiguous = _reference(norm, focus)
            if ids:
                entities["referenced_product_ids"] = ids
            ambiguous_reference = ambiguous_reference or (ambiguous and bool(focus))
        elif not clause_intents and re.search(
            r"\b(esse|essa|este|esta|ele|dele|mais barat|mais car)", norm
        ):
            ids, ambiguous = _reference(norm, focus)
            clause_intents = [Intent.PRODUCT_DETAIL]
            if ids:
                entities["referenced_product_ids"] = ids
            ambiguous_reference = ambiguous_reference or ambiguous
        intents.extend(clause_intents)
    if order_id is not None:
        entities.setdefault("order_id", order_id)
    return _decide(text, intents, entities, ambiguous_reference)


def _decide(
    text: str, intents: list[Intent], entities: dict[str, object], ambiguous_reference: bool
) -> RouteDecision:
    if ambiguous_reference:
        return RouteDecision(
            intents=intents or [Intent.PRODUCT_DETAIL],
            confidence=0.4,
            clarification="De qual desses modelos você está falando?",
        )
    if intents:
        return RouteDecision(
            intents=intents, entities=Entities.model_validate(entities)
        ).normalized()
    norm = normalize_text(text)
    if _GREETING.match(norm):
        return RouteDecision(confidence=0.95)
    if _STORE_HINTS.search(norm):
        return RouteDecision(
            confidence=0.3,
            clarification="Quer ver instrumentos, saber de um pedido ou tirar uma dúvida da loja?",
        )
    return RouteDecision(intents=[Intent.OUT_OF_SCOPE_GENERAL], confidence=0.9)
