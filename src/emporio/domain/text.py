"""Normalização de texto para busca por nome."""

from __future__ import annotations

import re
import unicodedata

_SPACES = re.compile(r"\s+")
_LETTER_DIGIT = re.compile(r"(?<=[a-z])(?=\d)")


def normalize_text(text: str) -> str:
    """Remove acentos, passa para minúsculas e colapsa espaços.

    Args:
        text: Texto livre (nome de produto ou consulta do cliente).

    Returns:
        Texto normalizado, ex.: ``"Violão Não"`` vira ``"violao nao"``.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _SPACES.sub(" ", stripped.lower()).strip()


def _is_model_token(token: str) -> bool:
    # Exige letra e dígito ("gd20", "ac-39"); "7" ou "100" sozinhos casariam com tudo.
    return any(c.isdigit() for c in token) and any(c.isalpha() for c in token)


def _model_variants(token: str) -> set[str]:
    compact = token.replace("-", "")
    return {token, compact, _LETTER_DIGIT.sub(" ", compact)}


def build_aliases(name_norm: str) -> list[str]:
    """Gera apelidos de busca a partir do nome normalizado.

    Para um token de modelo como ``gd20`` produz ``gd20``, ``gd 20`` e as
    combinações com a marca (primeiro token) e com o token anterior, como
    ``takamine gd20`` e ``takamine gd 20``.

    Args:
        name_norm: Nome já normalizado por :func:`normalize_text`.

    Returns:
        Apelidos ordenados e sem repetição, sem incluir o próprio nome.
    """
    tokens = name_norm.split()
    aliases: set[str] = set()
    for index, token in enumerate(tokens):
        if not _is_model_token(token):
            continue
        heads = {""}
        if index > 0:
            heads |= {tokens[0], tokens[index - 1]}
        for variant in _model_variants(token):
            aliases |= {f"{head} {variant}".strip() for head in heads}
    aliases.discard(name_norm)
    return sorted(aliases)
