"""Formato dos casos de avaliação (YAML) validado por Pydantic."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

EVALS_DIR = Path("evals")
SUITES = ("golden", "adversarial")


class ToolExpect(BaseModel):
    """Tool que deve ter sido chamada, com um subconjunto dos argumentos."""

    model_config = ConfigDict(extra="forbid")

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Expect(BaseModel):
    """O que o caso exige; todas as checagens são determinísticas, exceto ``judge_min``.

    As tools e intents valem para a conversa inteira; menções valem para a última resposta.
    """

    model_config = ConfigDict(extra="forbid")

    intents: list[str] = Field(default_factory=list)
    tool_calls: list[ToolExpect] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    must_mention: list[str] = Field(default_factory=list)
    must_mention_any: list[str] = Field(default_factory=list)
    must_not_mention: list[str] = Field(default_factory=list)
    grounded_numbers: bool = False
    no_pii: bool = False
    handoff: bool | None = None
    judge_min: float | None = Field(default=None, ge=1, le=5)

    @property
    def has_check(self) -> bool:
        """Há pelo menos uma checagem determinística."""
        return bool(
            self.intents
            or self.tool_calls
            or self.forbidden_tools
            or self.must_mention
            or self.must_mention_any
            or self.must_not_mention
            or self.grounded_numbers
            or self.no_pii
            or self.handoff is not None
        )


class Case(BaseModel):
    """Uma conversa (um ou mais turnos do cliente) e suas expectativas."""

    model_config = ConfigDict(extra="forbid")

    id: str
    turns: list[str] = Field(min_length=1)
    expect: Expect
    reference_date: date | None = None
    live: bool = False
    fake: bool = True
    tags: list[str] = Field(default_factory=list)
    suite: str = ""

    @model_validator(mode="after")
    def _has_a_check(self) -> Case:
        if not self.expect.has_check:
            raise ValueError(f"caso {self.id} sem nenhuma checagem")
        return self


def load_cases(suite: str, directory: Path = EVALS_DIR) -> list[Case]:
    """Carrega ``<suite>.yaml``; ids repetidos são erro.

    Raises:
        ValueError: Se houver id duplicado.
    """
    rows: list[dict[str, Any]] = (
        yaml.safe_load((directory / f"{suite}.yaml").read_text(encoding="utf-8")) or []
    )
    cases = [Case.model_validate({**row, "suite": suite}) for row in rows]
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"ids repetidos em {suite}.yaml")
    return cases
