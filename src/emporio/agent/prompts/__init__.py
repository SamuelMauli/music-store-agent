"""Prompts versionados (Jinja) com hash para o trace."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPT_DIR = Path(__file__).parent
SUMMARY_MARKER = "RESUMO DE CONVERSA"

_env = Environment(
    loader=FileSystemLoader(PROMPT_DIR),
    undefined=StrictUndefined,
    autoescape=False,  # noqa: S701 - prompt de texto puro, não HTML
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)


def render(name: str, **context: Any) -> str:
    """Renderiza ``<name>.jinja``.

    Args:
        name: Nome do template sem extensão: ``system``, ``router`` ou ``summary``.
        **context: Variáveis do template (``StrictUndefined`` acusa as que faltam).

    Returns:
        Texto do prompt.
    """
    return _env.get_template(f"{name}.jinja").render(**context).strip()


def prompt_hash() -> str:
    """Hash curto de todos os prompts, exposto em cada trace para correlacionar com métricas."""
    digest = hashlib.sha256()
    for path in sorted(PROMPT_DIR.glob("*.jinja")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]
