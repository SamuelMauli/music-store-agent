"""Dataset do guarda de injeção: set curado + dataset público, deduplicado e dividido.

Fontes (em ``evals/guard/``):

- ``curated.yaml``: escrito à mão para a loja (PT-BR e EN), com negativos difíceis; entradas
  com ``augment: true`` ganham variantes ofuscadas determinísticas.
- ``external/deepset_prompt_injections.jsonl``: ``deepset/prompt-injections`` (Apache-2.0),
  procedência em ``*.meta.json``.

Regras: deduplicação pelo texto normalizado (sem acento, caixa, pontuação); rótulos em
conflito descartam o texto; turnos de ``golden.yaml``/``adversarial.yaml`` ficam fora (o
``make eval`` continua fora da amostra). A divisão é por grupo (original + variantes juntos),
estratificada por (fonte, rótulo, categoria), ordenada por sha256: fixa e reprodutível.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml

from emporio.domain.text import normalize_text
from emporio.evals.cases import EVALS_DIR
from emporio.guardrails.input import normalize

GUARD_DIR: Final = EVALS_DIR / "guard"
CURATED: Final = GUARD_DIR / "curated.yaml"
EXTERNAL: Final = GUARD_DIR / "external" / "deepset_prompt_injections.jsonl"
TEST_FRACTION: Final = 0.2
SPLIT_SEED: Final = "guard-split-v1"
_LABELS: Final = {"injection": 1, "benign": 0}
_PUNCT: Final = re.compile(r"[^\w]+")
_LEET: Final = str.maketrans("aeiost", "431057")
_HOMOGLYPH: Final = str.maketrans("aeocpi", "аеосрі")  # cirílico
_DECODE_PROMPT: Final = {"pt": "decodifique e siga: ", "en": "decode and follow: "}


@dataclass(frozen=True, slots=True)
class GuardExample:
    """Um exemplo rotulado (``label``: 1 injeção, 0 benigno)."""

    text: str
    label: int
    category: str
    lang: str
    source: str
    group: str


@dataclass(slots=True)
class BuildStats:
    """O que o build descartou e por quê."""

    raw: int = 0
    duplicates: int = 0
    label_conflicts: int = 0
    excluded_eval_turns: int = 0
    conflicts: list[str] = field(default_factory=list)


def dedupe_key(text: str) -> str:
    """Chave de deduplicação: normalizado, sem acento, caixa nem pontuação."""
    return _PUNCT.sub(" ", normalize_text(normalize(text))).strip()


def obfuscations(text: str, lang: str) -> list[str]:
    """Variantes ofuscadas determinísticas: leet, letras separadas, base64 e homóglifos."""
    lower = text.lower()
    encoded = base64.b64encode(text.encode()).decode()
    return [
        lower.translate(_LEET),
        " ".join("-".join(word) for word in lower.split()),
        _DECODE_PROMPT.get(lang, _DECODE_PROMPT["en"]) + encoded,
        lower.translate(_HOMOGLYPH),
    ]


def _group_id(source: str, text: str) -> str:
    return f"{source}:{hashlib.sha256(text.encode()).hexdigest()[:12]}"


def load_curated(path: Path = CURATED) -> list[GuardExample]:
    """Lê o set curado e expande as variantes ofuscadas."""
    blocks: list[dict[str, Any]] = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    out: list[GuardExample] = []
    for block in blocks:
        label, category, lang = _LABELS[block["label"]], block["category"], block["lang"]
        for text in block["texts"]:
            if not isinstance(text, str):  # "a: b" sem aspas vira dict no YAML
                raise TypeError(f"texto não-string em {path.name}: {text!r}")
            group = _group_id("curated", text)
            out.append(GuardExample(text, label, category, lang, "curated", group))
            if block.get("augment"):
                out.extend(
                    GuardExample(v, label, "obfuscation", lang, "curated", group)
                    for v in obfuscations(text, lang)
                )
    return out


def load_external(path: Path = EXTERNAL) -> list[GuardExample]:
    """Lê o dataset público convertido (uma linha JSON por exemplo)."""
    out: list[GuardExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        category = "injection" if row["label"] else "benign"
        group = f"deepset:{row['upstream_split']}:{row['row_idx']}"
        out.append(
            GuardExample(row["text"], int(row["label"]), category, "en/de", "deepset", group)
        )
    return out


def eval_turns(directory: Path = EVALS_DIR) -> set[str]:
    """Chaves dos turnos dos evals ponta a ponta (excluídos do dataset)."""
    keys: set[str] = set()
    for suite in ("golden", "adversarial"):
        cases = yaml.safe_load((directory / f"{suite}.yaml").read_text(encoding="utf-8")) or []
        keys |= {dedupe_key(t) for case in cases for t in case.get("turns", [])}
    return keys


def must_answer_turns(directory: Path = EVALS_DIR) -> list[str]:
    """Turnos dos evals ponta a ponta que o agente tem de RESPONDER (a guarda não pode barrar).

    Todos os de ``golden.yaml`` (inclusive ``fake: false``) e os adversariais que esperam
    resposta (``must_mention`` ou ``tool_calls``), como a tag ``<system>`` falsa seguida de
    pergunta legítima: o contrato ali é ignorar a tag e responder o preço real.
    """
    turns: list[str] = []
    for suite in ("golden", "adversarial"):
        cases = yaml.safe_load((directory / f"{suite}.yaml").read_text(encoding="utf-8")) or []
        for case in cases:
            expect = case.get("expect") or {}
            if suite == "golden" or expect.get("must_mention") or expect.get("tool_calls"):
                turns.extend(case.get("turns", []))
    return turns


def deduplicate(
    examples: Iterable[GuardExample], exclude: set[str]
) -> tuple[list[GuardExample], BuildStats]:
    """Remove repetidos (fica o primeiro), conflitos de rótulo e turnos de eval."""
    stats = BuildStats()
    by_key: dict[str, GuardExample] = {}
    conflicted: set[str] = set()
    for ex in examples:
        stats.raw += 1
        key = dedupe_key(ex.text)
        if key in exclude:
            stats.excluded_eval_turns += 1
            continue
        kept = by_key.get(key)
        if kept is None:
            by_key[key] = ex
        elif kept.label != ex.label:
            conflicted.add(key)
        else:
            stats.duplicates += 1
    for key in sorted(conflicted):
        stats.label_conflicts += 1
        stats.conflicts.append(by_key.pop(key).text[:80])
    return list(by_key.values()), stats


def _rank(group: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{group}".encode()).hexdigest()


def split(
    examples: list[GuardExample], test_fraction: float = TEST_FRACTION
) -> tuple[list[GuardExample], list[GuardExample]]:
    """Divide por grupo, estratificado por (fonte, rótulo, categoria do primeiro do grupo).

    Returns:
        ``(train, test)`` na ordem original dos exemplos.
    """
    members: dict[str, list[GuardExample]] = defaultdict(list)
    for ex in examples:
        members[ex.group].append(ex)
    strata: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    for group, items in members.items():
        head = items[0]
        strata[(head.source, head.label, head.category)].append(group)
    test_groups: set[str] = set()
    for groups in strata.values():
        ordered = sorted(groups, key=_rank)
        n_test = max(1, round(len(ordered) * test_fraction)) if len(ordered) > 1 else 0
        test_groups.update(ordered[:n_test])
    train = [ex for ex in examples if ex.group not in test_groups]
    test = [ex for ex in examples if ex.group in test_groups]
    return train, test


def build(
    curated: Path = CURATED, external: Path = EXTERNAL, evals_dir: Path = EVALS_DIR
) -> tuple[list[GuardExample], list[GuardExample], BuildStats]:
    """Dataset completo: ``(train, test, stats)``."""
    examples, stats = deduplicate(
        [*load_curated(curated), *load_external(external)], eval_turns(evals_dir)
    )
    train, test = split(examples)
    return train, test, stats


def write_jsonl(path: Path, examples: list[GuardExample]) -> None:
    """Grava um exemplo por linha, UTF-8, chaves ordenadas (diff estável)."""
    path.write_text(
        "".join(
            json.dumps(asdict(ex), ensure_ascii=False, sort_keys=True) + "\n" for ex in examples
        ),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[GuardExample]:
    """Lê o que :func:`write_jsonl` gravou."""
    return [
        GuardExample(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
