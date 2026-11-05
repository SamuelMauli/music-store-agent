"""Ranqueamento de nomes por similaridade (puro, sem I/O)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from rapidfuzz import fuzz

from emporio.domain.lookup import MAX_OPTIONS
from emporio.domain.text import normalize_text

THRESHOLD = 85.0

Candidate = tuple[int, str, Sequence[str]]


def rank_candidates(query: str, candidates: Iterable[Candidate]) -> list[tuple[int, float]]:
    """Ranqueia produtos pelo melhor entre nome normalizado e aliases.

    ``token_set_ratio`` ignora ordem e palavras extras, o que casa "gd 20 takamine"
    com "takamine gd20 dreadnought natural". O custo é aceitar consultas genéricas
    ("violão") como subconjunto de muitos nomes, que viram empate e, portanto,
    pergunta de desambiguação.

    Args:
        query: Texto do cliente.
        candidates: Tuplas ``(product_id, name_norm, aliases)``.

    Returns:
        ``(product_id, score)`` com score >= :data:`THRESHOLD`, do maior ao menor
        score e, no empate, por ID.
    """
    needle = normalize_text(query)
    if not needle:
        return []
    ranked = []
    for product_id, name_norm, aliases in candidates:
        score = max(fuzz.token_set_ratio(needle, text) for text in (name_norm, *aliases))
        if score >= THRESHOLD:
            ranked.append((product_id, float(score)))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def pick(ranked: list[tuple[int, float]], exact: list[int]) -> list[tuple[int, float]]:
    """Decide quantos candidatos sobrevivem à desambiguação.

    Args:
        ranked: Saída de :func:`rank_candidates`.
        exact: IDs cujo nome normalizado é idêntico à consulta.

    Returns:
        Um item quando há vencedor único (por score ou por nome exato); caso
        contrário, até :data:`MAX_OPTIONS` empatados no topo; vazio sem candidatos.
    """
    if not ranked:
        return []
    if len(exact) == 1:
        return [item for item in ranked if item[0] == exact[0]]
    top = ranked[0][1]
    tied = [item for item in ranked if item[1] == top]
    return tied[:MAX_OPTIONS] if len(tied) > 1 else tied
