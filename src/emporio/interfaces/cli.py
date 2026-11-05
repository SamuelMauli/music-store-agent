"""CLI de conversa (``make chat``): Typer + Rich, com ``/debug`` e ``/reset``."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import date, datetime, time
from typing import Annotated
from zoneinfo import ZoneInfo

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from emporio.agent.service import ChatResult, ChatService
from emporio.bootstrap import build_container
from emporio.guardrails.rate_limit import RateLimitedError
from emporio.settings import Settings

HELP = "Comandos: /debug liga/desliga o painel técnico, /reset zera a conversa, /sair encerra."
app = typer.Typer(add_completion=False, help="Conversa com o Tom, atendente da Empório da Música.")


def debug_table(result: ChatResult) -> Table:
    """Painel técnico do turno: intent, tools, trechos citados, tokens e custo."""
    table = Table(title=f"debug · turno {result.turn}", show_header=False, box=None)
    rows = {
        "intents": ", ".join(result.intents) or "-",
        "tools": ", ".join(result.tools) or "-",
        "fontes citadas": ", ".join(result.sources) or "-",
        "tokens": f"{result.usage.input_tokens} in / {result.usage.output_tokens} out",
        "chamadas LLM": str(result.usage.calls),
        "custo do turno": f"US$ {result.usage.cost_usd:.6f}",
        "latência": f"{result.latency_ms} ms",
        "handoff": "sim" if result.handoff else "não",
        "bloqueado": "sim" if result.blocked else "não",
        "prompt": result.prompt_hash,
        "request_id": result.request_id,
    }
    for key, value in rows.items():
        table.add_row(key, value)
    return table


def run_chat(
    service: ChatService,
    session_id: str,
    read: Callable[[], str],
    console: Console,
    *,
    debug: bool = False,
) -> None:
    """Laço de conversa; ``read`` devolve a próxima linha ou levanta ``EOFError`` ao acabar."""
    console.print(f"[dim]{HELP}[/dim]")
    while True:
        try:
            line = read().strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "/sair":
            break
        if line == "/reset":
            service.reset(session_id)
            console.print("[dim]Conversa reiniciada.[/dim]")
            continue
        if line == "/debug":
            debug = not debug
            console.print(f"[dim]debug {'ligado' if debug else 'desligado'}[/dim]")
            continue
        try:
            result = service.chat(session_id, line)
        except RateLimitedError as exc:
            console.print(f"[yellow]{exc}[/yellow]")
            continue
        console.print(Panel(result.answer, title="Tom", border_style="cyan"))
        if debug:
            console.print(debug_table(result))


def _reference(day: date, zone: str) -> datetime:
    return datetime.combine(day, time(10, 0), tzinfo=ZoneInfo(zone))


@app.command()
def chat(
    session: Annotated[str | None, typer.Option(help="ID da sessão (retoma a memória).")] = None,
    reference_date: Annotated[
        datetime | None,
        typer.Option(formats=["%Y-%m-%d"], help="Data simulada de 'hoje' (ex.: 2026-03-25)."),
    ] = None,
    debug: Annotated[bool, typer.Option(help="Começa com o painel /debug ligado.")] = False,
) -> None:
    """Abre a conversa no terminal."""
    settings = Settings()
    if reference_date is not None:
        settings = settings.model_copy(
            update={"reference_datetime": _reference(reference_date.date(), settings.timezone)}
        )
    container = build_container(settings)
    console = Console()
    try:
        run_chat(
            container.service,
            session or uuid.uuid4().hex[:8],
            lambda: console.input("[bold green]você>[/bold green] "),
            console,
            debug=debug,
        )
    finally:
        container.close()


if __name__ == "__main__":
    app()
