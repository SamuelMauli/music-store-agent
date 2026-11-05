"""Esquemas de ``specs`` por categoria, em união discriminada Pydantic.

O CSV guarda ``specs`` como JSON cujo formato varia por categoria (e o lote
135-144 usa outra formatação). Aqui cada categoria ganha um modelo estrito;
o discriminador ``kind`` é injetado a partir de ``category_id``.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from emporio.ingestion.errors import IngestionError

CATEGORY_KIND: dict[int, str] = {
    1: "guitar",
    2: "bass",
    3: "drums",
    4: "keyboard",
    5: "acoustic_guitar",
    9: "ukulele",
}


class _Specs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GuitarSpecs(_Specs):
    """Guitarra elétrica."""

    kind: Literal["guitar"] = "guitar"
    body: str
    neck: str
    strings: int
    color: str


class BassSpecs(_Specs):
    """Baixo elétrico."""

    kind: Literal["bass"] = "bass"
    body: str
    neck: str
    strings: int
    color: str


class DrumSpecs(_Specs):
    """Bateria."""

    kind: Literal["drums"] = "drums"
    shells: str
    pieces: int
    hardware: str
    color: str


class KeyboardSpecs(_Specs):
    """Teclado ou piano."""

    kind: Literal["keyboard"] = "keyboard"
    keys: int
    type: str
    polyphony: int
    color: str


class _AcousticSpecs(_Specs):
    top: str
    back_sides: str
    neck: str
    strings: Literal["nylon", "steel"]
    scale_mm: int = Field(alias="scale")
    electronics: bool
    color: str

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @field_validator("scale_mm", mode="before")
    @classmethod
    def _strip_mm(cls, value: Any) -> Any:
        return value.removesuffix("mm") if isinstance(value, str) else value

    @field_validator("electronics", mode="before")
    @classmethod
    def _yes_no(cls, value: Any) -> Any:
        return {"yes": True, "no": False}.get(value, value) if isinstance(value, str) else value


class AcousticGuitarSpecs(_AcousticSpecs):
    """Violão."""

    kind: Literal["acoustic_guitar"] = "acoustic_guitar"


class UkuleleSpecs(_AcousticSpecs):
    """Ukulele."""

    kind: Literal["ukulele"] = "ukulele"


ProductSpecs = Annotated[
    GuitarSpecs | BassSpecs | DrumSpecs | KeyboardSpecs | AcousticGuitarSpecs | UkuleleSpecs,
    Field(discriminator="kind"),
]
_ADAPTER: TypeAdapter[ProductSpecs] = TypeAdapter(ProductSpecs)


def parse_specs(category_id: int, raw_json: str) -> ProductSpecs:
    """Valida o JSON de specs contra o esquema da categoria.

    Args:
        category_id: Categoria do produto (define o esquema).
        raw_json: Texto JSON da coluna ``specs``.

    Returns:
        O modelo tipado da categoria.

    Raises:
        IngestionError: Se a categoria não tem esquema, o JSON é inválido ou
            as chaves/valores não batem com o esquema.
    """
    kind = CATEGORY_KIND.get(category_id)
    if kind is None:
        raise IngestionError(f"sem esquema de specs para a categoria {category_id}")
    try:
        payload = json.loads(raw_json)
        if not isinstance(payload, dict):
            raise IngestionError("specs deve ser um objeto JSON")
        return _ADAPTER.validate_python({**payload, "kind": kind})
    except (json.JSONDecodeError, ValidationError) as exc:
        raise IngestionError(f"specs inválidas para {kind}: {exc}") from exc


def flatten_specs(specs: ProductSpecs) -> dict[str, str | int | bool]:
    """Specs como dicionário plano, sem o discriminador, com ``scale_mm`` canônico."""
    return specs.model_dump(mode="json", exclude={"kind"})
