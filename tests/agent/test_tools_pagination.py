from __future__ import annotations

import json

from emporio.agent.tools.results import Ok
from tests.agent.conftest import Caller


def test_search_pagination_walks_all_pages(call: Caller) -> None:
    first = call("search_products", category="violão", page_size=5)
    assert isinstance(first.result, Ok)
    total = first.result.value.total_matches
    pages = -(-total // 5)
    assert (first.result.value.page, first.result.value.total_pages) == (1, pages)
    assert first.result.value.has_more is (pages > 1)
    seen = [i.name for i in first.result.value.items]
    for page in range(2, pages + 1):
        nxt = call("search_products", category="violão", page_size=5, page=page)
        assert isinstance(nxt.result, Ok)
        assert nxt.result.value.has_more is (page < pages)
        seen += [i.name for i in nxt.result.value.items]
    assert len(seen) == total
    assert len(set(seen)) == total


def test_search_page_size_up_to_ten_and_default_unchanged(call: Caller) -> None:
    default = call("search_products", category="violão")
    wide = call("search_products", category="violão", page_size=10)
    assert isinstance(default.result, Ok)
    assert isinstance(wide.result, Ok)
    assert len(default.result.value.items) == 5
    assert len(wide.result.value.items) == 10
    assert default.result.value.page == 1


def test_search_page_beyond_the_end_is_empty_with_note(call: Caller) -> None:
    result = call("search_products", category="violão", page=99)
    assert isinstance(result.result, Ok)
    assert result.result.value.items == []
    assert result.result.value.has_more is False
    assert "não existe" in (result.result.value.note or "")
    assert result.result.value.total_matches > 0


def test_search_empty_has_zero_pages(call: Caller) -> None:
    result = call("search_products", category="saxofone")
    assert isinstance(result.result, Ok)
    assert (result.result.value.total_pages, result.result.value.has_more) == (0, False)


def test_search_pagination_bounds_are_enforced(call: Caller) -> None:
    for kwargs in ({"page": 0}, {"page_size": 11}, {"page_size": 0}):
        assert json.loads(call("search_products", **kwargs).payload)["status"] == "error"
