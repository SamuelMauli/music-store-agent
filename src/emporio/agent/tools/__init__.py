"""Registro das 14 tools do agente."""

from __future__ import annotations

import threading
from collections.abc import Collection
from typing import Any

from emporio.agent.tools.base import Tool, ToolContext, ToolExecution, ToolRegistry
from emporio.agent.tools.catalog import (
    CompareProducts,
    FindAlternatives,
    GetProduct,
    SearchProducts,
)
from emporio.agent.tools.commerce import ListActivePromotions, SimulatePayment
from emporio.agent.tools.info import EstimateShipping, GetStoreInfo, SearchPolicies
from emporio.agent.tools.orders import CheckReturnEligibility, GetOrderStatus, VerifyCustomer
from emporio.agent.tools.shipping_quote import QuoteShipping
from emporio.agent.tools.tickets import OpenTicket
from emporio.domain.calendar import DEFAULT_CALENDAR, HolidayCalendar
from emporio.ports import (
    Clock,
    CustomerRepository,
    OrderRepository,
    PolicyRetriever,
    ProductRepository,
    PromotionRepository,
    TicketRepository,
)


def build_registry(
    *,
    products: ProductRepository,
    orders: OrderRepository,
    customers: CustomerRepository,
    promotions: PromotionRepository,
    retriever: PolicyRetriever,
    tickets: TicketRepository,
    clock: Clock,
    metro_cities: Collection[str],
    lock: threading.RLock | None = None,
    calendar: HolidayCalendar = DEFAULT_CALENDAR,
) -> ToolRegistry:
    """Monta o ``ToolRegistry`` com as 14 tools (composition root das tools).

    Args:
        products: Repositório do catálogo.
        orders: Repositório de pedidos.
        customers: Repositório de clientes.
        promotions: Repositório de promoções.
        retriever: Estratégia de recuperação do manual.
        tickets: Repositório de chamados.
        clock: Relógio injetado.
        metro_cities: Cidades da região metropolitana.
        lock: Trava compartilhada para acesso ao DuckDB.
        calendar: Feriados nacionais para contar dias úteis (reembolso, prazo de entrega).

    Returns:
        Registro pronto para uso pelo grafo.
    """
    tools: list[Tool[Any, Any]] = [
        SearchProducts(products),
        GetProduct(products, promotions),
        CompareProducts(products),
        FindAlternatives(products),
        ListActivePromotions(promotions, products),
        SimulatePayment(products),
        VerifyCustomer(orders, customers),
        GetOrderStatus(orders, products, clock),
        CheckReturnEligibility(orders, products, clock, calendar),
        EstimateShipping(products, metro_cities),
        QuoteShipping(products, clock, calendar),
        GetStoreInfo(clock, calendar),
        SearchPolicies(retriever),
        OpenTicket(tickets, clock),
    ]
    return ToolRegistry(tools, lock=lock)


__all__ = ["Tool", "ToolContext", "ToolExecution", "ToolRegistry", "build_registry"]
