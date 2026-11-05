"""Seções do manual de políticas: a unidade de recuperação e de citação."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PolicySection:
    """Uma seção ou subseção numerada do manual.

    Attributes:
        section_id: Numeração do manual, ex.: ``"4.1"``.
        title: Título sem a numeração, ex.: ``"Direito de Arrependimento (Compras Online)"``.
        parent_id: Seção-mãe (``"4"`` para ``"4.1"``); ``None`` nas de nível 1.
        page: Página onde a seção começa (1-based).
        text: Corpo da seção, sem o das filhas; tabelas viram pares chave-valor.
        breadcrumb: Título hierárquico, ex.: ``"4. Trocas › 4.1 Arrependimento"``.
        keywords: Sinônimos de domínio que ajudam busca lexical e semântica.
    """

    section_id: str
    title: str
    parent_id: str | None
    page: int
    text: str
    breadcrumb: str
    keywords: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """Rótulo de citação, ex.: ``"§4.1 Direito de Arrependimento (Compras Online)"``."""
        return f"§{self.section_id} {self.title}"

    @property
    def search_text(self) -> str:
        """Texto indexado: hierarquia, palavras-chave e corpo."""
        parts = (self.breadcrumb, " ".join(self.keywords), self.text)
        return "\n".join(part for part in parts if part)
