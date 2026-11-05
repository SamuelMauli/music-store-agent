"""Compatibilidade: o adaptador OpenAI-compatível virou o genérico :class:`LangChainLLM`."""

from __future__ import annotations

from emporio.agent.llm.chat import ChatFactory, LangChainLLM
from emporio.agent.llm.messages import to_langchain

OpenAICompatibleLLM = LangChainLLM

__all__ = ["ChatFactory", "OpenAICompatibleLLM", "to_langchain"]
