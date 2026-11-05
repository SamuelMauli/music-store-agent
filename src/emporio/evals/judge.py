"""LLM-juiz opcional: rubrica 1 a 5 para tom, clareza e completude (nunca decide fatos)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from emporio.agent.llm.types import LLMProvider, Usage

RUBRIC = (
    "Você avalia a resposta de um atendente virtual de loja de instrumentos musicais. "
    "Dê notas de 1 a 5 para: tom (cordial, direto, sem exagero e sem emoji em reclamação), "
    "clareza (fácil de entender, sem rodeio) e completude (responde o que foi perguntado). "
    "Não julgue fatos, preços ou prazos: isso é checado por código. Seja rigoroso."
)


class JudgeScore(BaseModel):
    """Notas do juiz."""

    tone: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)

    @property
    def mean(self) -> float:
        """Média das três notas."""
        return (self.tone + self.clarity + self.completeness) / 3


def judge(llm: LLMProvider, question: str, answer: str) -> tuple[JudgeScore, Usage]:
    """Pontua uma resposta com o modelo do juiz (família diferente da do gerador)."""
    messages = [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": f"Pergunta do cliente:\n{question}\n\nResposta:\n{answer}"},
    ]
    return llm.structured(messages, JudgeScore, tier="small")
