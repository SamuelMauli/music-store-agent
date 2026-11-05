"""Data contracts (subconjunto do ODCS v3) e validação linha a linha."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from emporio.ingestion.errors import IngestionError

_TRUE = {"1", "true", "t", "yes", "sim"}
_FALSE = {"0", "false", "f", "no", "nao", "não"}


class ContractProperty(BaseModel):
    """Coluna de uma tabela no contrato."""

    model_config = ConfigDict(extra="allow")

    name: str
    logicalType: str  # noqa: N815 - nome do campo no padrão ODCS
    required: bool = False
    unique: bool = False
    minimum: Decimal | None = None
    maximum: Decimal | None = None
    enum: list[str] | None = None
    references: str | None = None


class ContractTable(BaseModel):
    """Tabela descrita pelo contrato."""

    name: str
    properties: list[ContractProperty]


class DataContract(BaseModel):
    """Contrato de dados de um data product (campos ODCS relevantes)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    version: str
    owner: str
    description: str
    tables: list[ContractTable] = Field(alias="schema")

    @property
    def table(self) -> ContractTable:
        """Tabela única descrita por este contrato."""
        return self.tables[0]


def load_contract(path: Path) -> DataContract:
    """Lê e valida um arquivo de contrato YAML.

    Args:
        path: Caminho do ``.yaml``.

    Returns:
        O contrato validado.

    Raises:
        IngestionError: Se o arquivo não existir ou não seguir o formato.
    """
    try:
        return DataContract.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise IngestionError(f"contrato inválido em {path}: {exc}") from exc


def _parse_int(raw: str) -> int:
    return int(raw)


def _parse_number(raw: str) -> Decimal:
    value = Decimal(raw)
    if not value.is_finite():
        raise ValueError(raw)
    return value


def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(raw)


_PARSERS: dict[str, Callable[[str], object]] = {
    "integer": _parse_int,
    "number": _parse_number,
    "date": date.fromisoformat,
    "boolean": _parse_bool,
    "string": str.strip,
}


def _coerce(prop: ContractProperty, raw: str) -> tuple[object | None, str | None]:
    parser = _PARSERS.get(prop.logicalType)
    if parser is None:
        return None, f"{prop.name}: tipo lógico desconhecido {prop.logicalType!r}"
    try:
        return parser(raw), None
    except (ValueError, InvalidOperation):
        return None, f"{prop.name}: valor {raw!r} não é {prop.logicalType}"


def _check_bounds(prop: ContractProperty, value: object) -> str | None:
    if not isinstance(value, int | Decimal) or isinstance(value, bool):
        return None
    if prop.minimum is not None and value < prop.minimum:
        return f"{prop.name}: {value} abaixo do mínimo {prop.minimum}"
    if prop.maximum is not None and value > prop.maximum:
        return f"{prop.name}: {value} acima do máximo {prop.maximum}"
    return None


@dataclass
class RowValidator:
    """Valida linhas de uma tabela contra o contrato, mantendo estado de unicidade.

    Attributes:
        table: Definição da tabela no contrato.
        references: Valores aceitos nas tabelas-pai, chaveados por ``tabela.coluna``.
    """

    table: ContractTable
    references: Mapping[str, set[object]] = field(default_factory=dict)
    _seen: dict[str, set[object]] = field(default_factory=dict, init=False)

    def validate(self, row: Mapping[str, str | None]) -> tuple[dict[str, object], list[str]]:
        """Valida e tipa uma linha.

        Args:
            row: Linha crua (strings ou ``None``) vinda da bronze.

        Returns:
            Par ``(valores tipados, motivos de rejeição)``. Só é seguro usar os
            valores quando a lista de motivos está vazia.
        """
        typed: dict[str, object] = {}
        reasons: list[str] = []
        for prop in self.table.properties:
            value, problem = self._check(prop, row.get(prop.name))
            typed[prop.name] = value
            if problem:
                reasons.append(problem)
        if not reasons:
            self._remember_unique(typed)
        return typed, reasons

    def _check(self, prop: ContractProperty, raw: str | None) -> tuple[object | None, str | None]:
        if raw is None or not raw.strip():
            return None, (f"{prop.name}: obrigatório ausente" if prop.required else None)
        value, problem = _coerce(prop, raw)
        return value, problem or self._check_value(prop, value)

    def _check_value(self, prop: ContractProperty, value: object) -> str | None:
        if prop.enum is not None and value not in prop.enum:
            return f"{prop.name}: {value!r} fora do enum {prop.enum}"
        bounds = _check_bounds(prop, value)
        if bounds:
            return bounds
        if prop.unique and value in self._seen.get(prop.name, set()):
            return f"{prop.name}: duplicado ({value})"
        if prop.references and value not in self.references.get(prop.references, set()):
            return f"{prop.name}: referência inexistente em {prop.references} ({value})"
        return None

    def _remember_unique(self, typed: Mapping[str, object]) -> None:
        for prop in self.table.properties:
            if prop.unique and typed[prop.name] is not None:
                self._seen.setdefault(prop.name, set()).add(typed[prop.name])


def contract_for(table: str, contracts_dir: Path) -> DataContract:
    """Carrega ``<contracts_dir>/<table>.yaml``.

    Raises:
        IngestionError: Se o contrato não existir ou for inválido.
    """
    return load_contract(contracts_dir / f"{table}.yaml")
