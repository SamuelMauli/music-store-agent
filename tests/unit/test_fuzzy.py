from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from emporio.adapters.persistence.fuzzy import THRESHOLD, pick, rank_candidates

CANDIDATES = [
    (1, "takamine gd20 dreadnought natural", ["gd20", "gd 20", "takamine gd20"]),
    (2, "giannini gf 3d violao nylon", ["gf3d", "gf 3d"]),
    (3, "giannini gf 1d violao aco", ["gf1d", "gf 1d"]),
    (4, "fender stratocaster sunburst", []),
]


def test_alias_matches_with_full_score() -> None:
    assert rank_candidates("GD20", CANDIDATES)[0] == (1, 100.0)


def test_word_order_and_accents_are_ignored() -> None:
    assert rank_candidates("Violão Giannini", CANDIDATES)[0][1] >= THRESHOLD


def test_gibberish_and_blank_have_no_candidates() -> None:
    assert rank_candidates("xyzzy plugh", CANDIDATES) == []
    assert rank_candidates("   ", CANDIDATES) == []


def test_tie_returns_every_top_candidate_up_to_three() -> None:
    ranked = rank_candidates("giannini", CANDIDATES)
    assert [pid for pid, _ in pick(ranked, [])] == [2, 3]


def test_tie_is_capped_at_three_options() -> None:
    many = [(i, f"violao modelo {i}", []) for i in range(1, 6)]
    assert len(pick(rank_candidates("violao", many), [])) == 3


def test_unique_exact_name_breaks_a_tie() -> None:
    ranked = [(1, 100.0), (2, 100.0)]
    assert pick(ranked, [2]) == [(2, 100.0)]


def test_pick_of_nothing_is_empty() -> None:
    assert pick([], []) == []


@given(st.text(max_size=40))
def test_ranking_is_total_and_deterministic(query: str) -> None:
    first = rank_candidates(query, CANDIDATES)
    assert first == rank_candidates(query, CANDIDATES)
    assert all(score >= THRESHOLD for _, score in first)
