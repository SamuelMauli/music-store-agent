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
# Leet só em palavra "de letra": >= 2 caracteres leet e sem formato de identificador (SKU,
# modelo, rastreio, UUID, e-mail, chave pix), para "GF-3D" ou "QB123456789BR" não virarem texto.
_LEETABLE: Final = re.compile(r"(?<![\w@.\-])(?=\S*[a-z])(?=(?:\S*[013457@$!|]){2})[^\s]+")
_ID_LIKE: Final = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+"  # e-mail
    r"|(?=\S*\d)\w+(?:[-_/.]\w+)+"  # modelo/SKU/UUID com separador (GF-3D, FG-830)
    r"|[a-z]{0,3}\d{4,}[a-z]{0,3}"  # rastreio, pedido, CPF (QB123456789BR)
    r"|[0-9a-f]{8,}"  # hash, UUID sem hífen
)
_HYPHENATED: Final = re.compile(r"\b([a-z]{2,})-([a-z]{2,})\b")
_SPACED_RUN: Final = re.compile(r"(?:\b[a-z]\b[\s.\-_*]+){3,}\b[a-z]\b")
_BASE64: Final = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
# Limites contam trechos que DECODIFICAM (não matches da regex) e o total de bytes, para que
# trechos-isca na frente não escondam o payload; os maiores decodificados têm prioridade.
_MAX_BASE64_DECODED: Final = 4
_MAX_BASE64_BYTES: Final = 2048
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
    """Trechos base64 que decodificam para texto UTF-8 imprimível, maiores primeiro.

    Percorre todos os candidatos (a entrada já vem limitada a ``MAX_INPUT_CHARS``), para no
    teto de bytes decodificados e devolve no máximo ``_MAX_BASE64_DECODED`` trechos.
    """
    found: list[str] = []
    budget = _MAX_BASE64_BYTES
    for token in _BASE64.findall(text):
        try:
            raw = base64.b64decode(token + "=" * (-len(token) % 4), validate=True)
            decoded = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        budget -= len(raw)
        if budget < 0:
            break
        if decoded.strip() and decoded.isprintable():
            found.append(decoded)
    return sorted(found, key=len, reverse=True)[:_MAX_BASE64_DECODED]


def _unleet(text: str) -> str:
    def swap(match: re.Match[str]) -> str:
        token = match.group()
        return token if _ID_LIKE.fullmatch(token.strip(".,;:!?")) else token.translate(_LEET)

    return _LEETABLE.sub(swap, text)


def _dedupe(candidates: list[str], base: str) -> list[str]:
    out: list[str] = []
    for candidate in candidates:
        if candidate and candidate != base and candidate not in out:
            out.append(candidate)
    return out


def classifier_variants(text: str) -> list[str]:
    """Variantes que o classificador também pontua: homóglifos trocados e base64 decodificado.

    Leet, hífen e letras separadas ficam de fora: reescrevem identificadores legítimos (SKU,
    rastreio, UUID, e-mail) em texto que o embedding confunde com ataque.

    Args:
        text: Mensagem já normalizada por :func:`normalize`.

    Returns:
        Variantes diferentes do texto normalizado, sem repetição.
    """
    folded = normalize_text(text.translate(_HOMOGLYPHS))
    decoded = [normalize_text(d) for d in _decoded_base64(text)]
    return _dedupe([folded, *decoded], normalize_text(text))


def deobfuscated_variants(text: str) -> list[str]:
    """Versões desofuscadas da mensagem, para a heurística de padrões.

    Args:
        text: Mensagem já normalizada por :func:`normalize`.

    Returns:
        Variantes (homóglifos, leet fora de identificadores, hífen intra-palavra removido,
        base64 decodificado), sem repetição; vazio se nada mudou. O texto original não entra.
    """
    folded = normalize_text(text.translate(_HOMOGLYPHS))
    unleet = _unleet(folded)
    joined = _HYPHENATED.sub(r"\1\2", unleet)
    decoded = [normalize_text(d) for d in _decoded_base64(text)]
    return _dedupe([unleet, joined, *decoded], normalize_text(text))


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
