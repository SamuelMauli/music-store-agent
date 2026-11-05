"""``scope_gate``: recusas e esclarecimentos por template, sem chamar o modelo grande."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from emporio.agent.intents import Intent, RouteDecision
from emporio.agent.tools.categories import CATEGORY_NAMES, resolve_category

# Categorias que existem no cadastro mas não têm nenhum produto (D1).
EMPTY_CATEGORIES: Final = frozenset({6, 7, 8})
OFFER: Final = "violões, guitarras, baixos, baterias, teclados e ukuleles"

GENERAL: Final = (
    "Aqui eu só consigo ajudar com a Empório da Música: instrumentos, preços, pedidos, "
    "trocas e entrega. Quer que eu ajude com algum desses?"
)
ACCESSORY: Final = (
    "A Empório trabalha só com instrumentos musicais, então não temos cordas, cabos, palhetas, "
    f"pedais, amplificadores nem cases. Por aqui você encontra {OFFER}. "
    "Quer ver alguma opção?"
)
CLARIFY: Final = "Não tenho certeza se entendi. Pode me contar um pouco mais do que você precisa?"


@dataclass(frozen=True, slots=True)
class ScopeOutcome:
    """Resultado do portão de escopo.

    Attributes:
        answer: Resposta final por template; ``None`` se ainda há trabalho para o agente.
        notes: Avisos que precedem a resposta do agente (intents parcialmente fora do escopo).
        intents: Intents que seguem para o agente.
    """

    answer: str | None = None
    notes: list[str] = field(default_factory=list)
    intents: list[Intent] = field(default_factory=list)


def empty_category_text(category_text: str | None) -> str | None:
    """Mensagem para categoria sem produtos; ``None`` se a categoria tem catálogo."""
    category_id = resolve_category(category_text) if category_text else None
    if category_id not in EMPTY_CATEGORIES:
        return None
    name = CATEGORY_NAMES[category_id].split(" (")[0].lower()
    return (
        f"No momento não temos {name} no catálogo. "
        f"O que temos hoje: {OFFER}. Quer que eu mostre alguma opção?"
    )


def scope_gate(route: RouteDecision) -> ScopeOutcome:
    """Decide o que responde por template e o que segue para o agente.

    Args:
        route: Decisão do roteador.

    Returns:
        ``answer`` preenchido quando nada resta para o agente; caso contrário, ``notes``
        com avisos e ``intents`` com o que continua.
    """
    if route.low_confidence:
        return ScopeOutcome(answer=route.clarification or CLARIFY)
    notes: list[str] = []
    remaining: list[Intent] = []
    for intent in route.intents:
        if intent == Intent.OUT_OF_SCOPE_GENERAL:
            notes.append(GENERAL)
        elif intent == Intent.OUT_OF_SCOPE_ACCESSORY:
            notes.append(ACCESSORY)
        elif intent in (Intent.CATALOG_SEARCH, Intent.RECOMMENDATION) and (
            text := empty_category_text(route.entities.category)
        ):
            notes.append(text)
        else:
            remaining.append(intent)
    if not remaining and notes:
        return ScopeOutcome(answer=" ".join(dict.fromkeys(notes)))
    return ScopeOutcome(notes=list(dict.fromkeys(notes)), intents=remaining)
