"""Intents, entidades e a decisão do roteador (saída estruturada)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, Field


class Intent(StrEnum):
    """As 15 intents do agente."""

    CATALOG_SEARCH = "catalog_search"
    PRODUCT_DETAIL = "product_detail"
    RECOMMENDATION = "recommendation"
    ACTIVE_PROMOTIONS = "active_promotions"
    PAYMENT_SIMULATION = "payment_simulation"
    ORDER_STATUS = "order_status"
    RETURN_EXCHANGE = "return_exchange"
    SHIPPING = "shipping"
    WARRANTY = "warranty"
    STORE_INFO = "store_info"
    COMPLAINT = "complaint"
    PRIVACY_REQUEST = "privacy_request"
    OUT_OF_SCOPE_ACCESSORY = "out_of_scope_accessory"
    OUT_OF_SCOPE_GENERAL = "out_of_scope_general"
    HUMAN_HANDOFF = "human_handoff"


CONFIDENCE_THRESHOLD: Final = 0.6

VERIFICATION_INTENTS: Final = frozenset({Intent.ORDER_STATUS, Intent.RETURN_EXCHANGE})
SCOPE_INTENTS: Final = frozenset({Intent.OUT_OF_SCOPE_ACCESSORY, Intent.OUT_OF_SCOPE_GENERAL})

# Só as tools da intent chegam ao modelo (menos tokens e menos chance de escolha errada).
# ``verify_customer`` é executada pelo nó de verificação, nunca pelo nó ReAct.
TOOLS_BY_INTENT: Final[dict[Intent, tuple[str, ...]]] = {
    Intent.CATALOG_SEARCH: ("search_products", "get_product"),
    Intent.PRODUCT_DETAIL: ("get_product", "find_alternatives", "search_products"),
    Intent.RECOMMENDATION: ("search_products", "compare_products", "get_product"),
    Intent.ACTIVE_PROMOTIONS: ("list_active_promotions",),
    Intent.PAYMENT_SIMULATION: ("simulate_payment", "get_product"),
    Intent.ORDER_STATUS: ("verify_customer", "get_order_status"),
    Intent.RETURN_EXCHANGE: (
        "verify_customer",
        "check_return_eligibility",
        "search_policies",
        "open_ticket",
    ),
    Intent.SHIPPING: ("estimate_shipping", "get_product"),
    Intent.WARRANTY: ("search_policies",),
    Intent.STORE_INFO: ("get_store_info",),
    Intent.COMPLAINT: ("open_ticket",),
    Intent.PRIVACY_REQUEST: ("open_ticket", "search_policies"),
    Intent.OUT_OF_SCOPE_ACCESSORY: (),
    Intent.OUT_OF_SCOPE_GENERAL: (),
    Intent.HUMAN_HANDOFF: ("open_ticket",),
}

POLICY_INTENTS: Final = frozenset(
    {
        Intent.RETURN_EXCHANGE,
        Intent.WARRANTY,
        Intent.COMPLAINT,
        Intent.PRIVACY_REQUEST,
        Intent.PAYMENT_SIMULATION,
        Intent.SHIPPING,
    }
)


class Entities(BaseModel):
    """Entidades extraídas da mensagem; vazio quando o cliente não disse."""

    category: str | None = Field(default=None, description="violão, guitarra, baixo, sax...")
    min_price: float | None = Field(default=None, description="Preço mínimo em reais.")
    max_price: float | None = Field(default=None, description="Preço máximo em reais.")
    strings: Literal["nylon", "aço"] | None = None
    electric: bool | None = None
    product_name: str | None = Field(default=None, description="Modelo citado, ex.: GD20.")
    referenced_product_ids: list[int] = Field(
        default_factory=list,
        description="IDs de produtos em foco a que 'esse', 'o mais barato' se referem.",
    )
    order_id: int | None = None
    city: str | None = None
    channel: Literal["online", "store"] | None = None
    return_reason: Literal["regret", "defect", "preference"] | None = None


class RouteDecision(BaseModel):
    """Decisão do roteador: o que o cliente quer e com que dados."""

    intents: list[Intent] = Field(
        default_factory=list,
        description="Todas as intents da mensagem; vazio para saudação ou conversa social.",
    )
    entities: Entities = Field(default_factory=Entities)
    needs_verification: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    clarification: str | None = Field(
        default=None, description="Pergunta curta quando a confiança é baixa."
    )

    def normalized(self) -> RouteDecision:
        """Remove duplicatas e recalcula ``needs_verification`` pelo código, não pelo modelo."""
        unique = list(dict.fromkeys(self.intents))
        return self.model_copy(
            update={
                "intents": unique,
                "needs_verification": any(i in VERIFICATION_INTENTS for i in unique),
            }
        )

    @property
    def low_confidence(self) -> bool:
        """Confiança abaixo do limiar: perguntar antes de agir."""
        return self.confidence < CONFIDENCE_THRESHOLD


def tools_for(intents: list[Intent]) -> list[str]:
    """União ordenada das tools das intents, sem ``verify_customer``."""
    names: dict[str, None] = {}
    for intent in intents:
        for name in TOOLS_BY_INTENT[intent]:
            if name != "verify_customer":
                names[name] = None
    return list(names)
