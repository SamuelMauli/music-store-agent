"""Nó ``summarize``: mantém só as últimas mensagens na janela e o resto num resumo curto."""

from __future__ import annotations

import logging
from typing import Any, Final

from emporio.agent.llm.types import LLMError
from emporio.agent.nodes.deps import HISTORY_WINDOW, Deps, add_usage
from emporio.agent.prompts import render
from emporio.agent.state import ChatState

log = logging.getLogger("emporio.memory")
SUMMARY_BATCH: Final = 4  # só resume quando há ao menos isto fora da janela


def make_summarizer(deps: Deps) -> Any:
    """Cria o nó que incorpora ao resumo as mensagens que saíram da janela de 12."""

    def summarize(state: ChatState) -> dict[str, Any]:
        messages = state.get("messages", [])
        cutoff = len(messages) - HISTORY_WINDOW
        done = state.get("summarized_upto", 0)
        if cutoff - done < SUMMARY_BATCH:
            return {}
        prompt = render(
            "summary", previous=state.get("summary", ""), messages=messages[done:cutoff]
        )
        try:
            response = deps.llm.complete(
                [{"role": "system", "content": prompt}], tier="small", temperature=0.0
            )
        except LLMError:
            log.warning("resumo do histórico falhou; mantendo o anterior")
            return {}
        return {
            "summary": response.text.strip(),
            "summarized_upto": cutoff,
            "usage": add_usage(state, response.usage),
        }

    return summarize
