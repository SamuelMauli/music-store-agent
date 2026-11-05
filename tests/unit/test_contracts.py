from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from emporio.ingestion.contracts import ContractTable, RowValidator, load_contract
from emporio.ingestion.errors import IngestionError
from emporio.ingestion.specs import flatten_specs, parse_specs

CONTRACTS = Path(__file__).resolve().parents[2] / "data" / "contracts"


@pytest.fixture
def products_table() -> ContractTable:
    return load_contract(CONTRACTS / "products.yaml").table


def good_row() -> dict[str, str | None]:
    return {
        "product_id": "1",
        "name": "Violão X",
        "price_brl": "199.9",
        "stock_quantity": "3",
        "status": "active",
        "category_id": "5",
        "description": None,
        "specs": "{}",
        "created_at": "2024-01-31",
    }


def validator(table: ContractTable) -> RowValidator:
    return RowValidator(table, references={"categories.category_id": {5}})


def test_valid_row_is_typed(products_table: ContractTable) -> None:
    typed, reasons = validator(products_table).validate(good_row())
    assert reasons == []
    assert typed["price_brl"] == Decimal("199.9")
    assert typed["product_id"] == 1
    assert typed["description"] is None


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("price_brl", "0", "abaixo do mínimo"),
        ("price_brl", "abc", "não é number"),
        ("price_brl", "NaN", "não é number"),
        ("stock_quantity", "-1", "abaixo do mínimo"),
        ("stock_quantity", "1.5", "não é integer"),
        ("status", "sold_out", "fora do enum"),
        ("category_id", "99", "referência inexistente"),
        ("name", "  ", "obrigatório"),
        ("created_at", "31/01/2024", "não é date"),
    ],
)
def test_violations(products_table: ContractTable, field: str, value: str, fragment: str) -> None:
    row = good_row()
    row[field] = value
    _, reasons = validator(products_table).validate(row)
    assert any(fragment in r for r in reasons), reasons


def test_duplicate_primary_key_rejected_after_first(products_table: ContractTable) -> None:
    v = validator(products_table)
    assert v.validate(good_row())[1] == []
    assert any("duplicado" in r for r in v.validate(good_row())[1])


def test_rejected_row_does_not_reserve_its_key(products_table: ContractTable) -> None:
    v = validator(products_table)
    bad = good_row() | {"price_brl": "0"}
    assert v.validate(bad)[1]
    assert v.validate(good_row())[1] == []


def test_boolean_and_maximum() -> None:
    table = load_contract(CONTRACTS / "promotions.yaml").table
    v = RowValidator(table, references={"products.product_id": {1}})
    row: dict[str, str | None] = {
        "promotion_id": "1",
        "product_id": "1",
        "discount_percent": "101",
        "description": "x",
        "is_active": "talvez",
    }
    reasons = v.validate(row)[1]
    assert any("acima do máximo" in r for r in reasons)
    assert any("is_active" in r for r in reasons)


def test_invalid_contract_file_raises(tmp_path: Path) -> None:
    broken = tmp_path / "x.yaml"
    broken.write_text("id: [", encoding="utf-8")
    with pytest.raises(IngestionError):
        load_contract(broken)
    with pytest.raises(IngestionError):
        load_contract(tmp_path / "missing.yaml")


def test_every_contract_loads() -> None:
    for path in sorted(CONTRACTS.glob("*.yaml")):
        contract = load_contract(path)
        assert contract.table.name == path.stem
        assert contract.owner


def test_unknown_logical_type_is_reported() -> None:
    table = ContractTable.model_validate(
        {"name": "t", "properties": [{"name": "a", "logicalType": "blob"}]}
    )
    _, reasons = RowValidator(table).validate({"a": "x"})
    assert "tipo lógico desconhecido" in reasons[0]


class TestSpecs:
    def test_acoustic_normalizes_scale_and_electronics(self) -> None:
        raw = (
            '{"top":"Cedro","back_sides":"Sapele","neck":"Mogno","strings":"steel",'
            '"scale":"650mm","electronics":"no","color":"Natural"}'
        )
        specs = parse_specs(5, raw)
        assert specs.kind == "acoustic_guitar"
        assert flatten_specs(specs)["scale_mm"] == 650
        assert flatten_specs(specs)["electronics"] is False

    def test_spaced_json_with_numeric_strings_is_accepted(self) -> None:
        raw = '{"body": "Ash", "neck": "Maple", "strings": "4", "color": "Red"}'
        assert flatten_specs(parse_specs(2, raw))["strings"] == 4

    @pytest.mark.parametrize(
        ("category", "raw"),
        [
            (6, "{}"),
            (2, "[1]"),
            (2, "{not json"),
            (2, '{"body": "Ash"}'),
            (2, '{"body":"Ash","neck":"M","strings":4,"color":"R","extra":1}'),
            (
                5,
                '{"top":"a","back_sides":"b","neck":"c","strings":"nylon6",'
                '"scale":"650mm","electronics":"no","color":"d"}',
            ),
        ],
    )
    def test_invalid_specs(self, category: int, raw: str) -> None:
        with pytest.raises(IngestionError):
            parse_specs(category, raw)
