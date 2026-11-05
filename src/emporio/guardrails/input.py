"""Guarda de entrada: normaliza o texto e detecta tentativa óbvia de injeção de prompt.

Esta é a camada heurística (padrões PT/EN sobre o texto e sobre variantes desofuscadas). O
classificador treinado e a política de combinação ficam em ``guardrails/injection.py``.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from typing import Final

from emporio.domain.text import normalize_text

MAX_INPUT_CHARS: Final = 2000
REFUSAL: Final = (
    "Não consigo mudar regras, preços nem descontos por aqui, e não compartilho minhas "
    "instruções. Posso te ajudar com instrumentos, preços, pedidos e trocas. "
    "O que você procura?"
)
_INVISIBLE: Final = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
_INJECTION: Final = tuple(
    re.compile(p)
    for p in (
        r"\b(ignore|ignora|desconsidere|esqueca|desobedeca|descarte)\b.{0,40}"
        r"\b(regras?|instrucoes|instrucao|prompt|politicas?|orientacoes)\b",
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(rules?|instructions?|prompt|polic)",
        r"\b(revele|mostre|exiba|repita|vaze|imprima)\b.{0,30}\b(prompt|instrucoes|system)\b",
        r"\b(reveal|show|print|repeat)\b.{0,30}\b(system prompt|instructions)\b",
        r"\bsystem prompt\b|\bprompt do sistema\b",
        r"\b(modo|mode)\s+(desenvolvedor|developer|admin|deus|god)\b",
        r"\b(voce agora e|a partir de agora voce|finja que voce|aja como se)\b",
        r"\b(jailbreak|dan mode)\b",
    )
)
# Ofuscação comum: leet ("1gn0r3"), homóglifos cirílicos/gregos (I, o, e trocados), letras separadas
# ("i-g-n-o-r-e", "s y s t e m") e payload curto em base64.
_LEET: Final = str.maketrans("013457@$!|", "oieastasii")
_HOMOGLYPHS: Final = str.maketrans(
    "аеорсхуіїјѕԁһԛԝАВЕКМНОРСТХІЈЅΑΒΕΖΗΙΚΜΝΟΡΤΥΧοιⅠⅰ",
    "aeopcxyiijsdhqwABEKMHOPCTXIJSABEZHIKMNOPTYXoiIi",
)
_LEETABLE: Final = re.compile(r"\b(?=\S*[a-z])(?=\S*[013457@$!|])\S+")
_SPACED_RUN: Final = re.compile(r"(?:\b[a-z]\b[\s.\-_*]+){3,}\b[a-z]\b")
_BASE64: Final = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
_MAX_BASE64_TOKENS: Final = 3
# Frases-assinatura procuradas no texto "espremido" (só letras), quando há letras separadas.
_SQUASHED: Final = (
    "ignoreasregras",
    "ignoreasinstrucoes",
    "ignoresuasinstrucoes",
    "reveleoprompt",
    "promptdosistema",
    "ignoreprevious",
    "ignoreallprevious",
    "ignoretherules",
    "disregardtherules",
    "revealyourprompt",
    "systemprompt",
)


def normalize(text: str) -> str:
    """NFKC, sem caracteres invisíveis ou de controle, espaços colapsados, limitado a 2000.

    Args:
        text: Mensagem crua do cliente.

    Returns:
        Texto limpo.
    """
    text = unicodedata.normalize("NFKC", _INVISIBLE.sub("", text))
    text = "".join(c if unicodedata.category(c)[0] != "C" else " " for c in text)
    return " ".join(text.split())[:MAX_INPUT_CHARS]


def _decoded_base64(text: str) -> list[str]:
    """Trechos base64 que decodificam para texto UTF-8 imprimível (no máximo 3)."""
    found: list[str] = []
    for token in _BASE64.findall(text)[:_MAX_BASE64_TOKENS]:
        try:
            raw = base64.b64decode(token + "=" * (-len(token) % 4), validate=True)
            decoded = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if decoded.isprintable():
            found.append(decoded)
    return found


def deobfuscated_variants(text: str) -> list[str]:
    """Versões desofuscadas da mensagem que diferem do texto normalizado.

    Args:
        text: Mensagem já normalizada por :func:`normalize`.

    Returns:
        Variantes (homóglifos e leet trocados, base64 decodificado), sem repetição; vazio se
        nada mudou. O texto original não entra na lista.
    """
    base = normalize_text(text)
    folded = normalize_text(text.translate(_HOMOGLYPHS))
    folded = _LEETABLE.sub(lambda m: m.group().translate(_LEET), folded)
    candidates = [folded, *(normalize_text(d) for d in _decoded_base64(text))]
    out: list[str] = []
    for candidate in candidates:
        if candidate and candidate != base and candidate not in out:
            out.append(candidate)
    return out


def _matches(norm: str) -> bool:
    if any(p.search(norm) for p in _INJECTION):
        return True
    if not _SPACED_RUN.search(norm):
        return False
    squashed = re.sub(r"[^a-z]", "", norm)
    return any(sig in squashed for sig in _SQUASHED)


def looks_like_injection(text: str) -> bool:
    """Heurística de padrões conhecidos (PT/EN) sobre o texto e suas variantes desofuscadas.

    Args:
        text: Mensagem já normalizada.

    Returns:
        ``True`` se o texto ou alguma variante casar com "ignore as regras", "revele o prompt"
        e afins.
    """
    return any(_matches(v) for v in (normalize_text(text), *deobfuscated_variants(text)))
