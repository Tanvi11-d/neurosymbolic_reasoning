"""Structured I/O models (Pydantic) for LLM stages — independent of transport (OpenRouter, etc.).

Layout::

    nre/schemas/
        task.py      # LLMTaskItem (shared)
        decompose.py
        check.py
        order.py
        match.py

Typical import::

    from nre.schemas import LLMTaskItem, LLMDecomposeOutput

Stage-specific::

    from nre.schemas.task import LLMTaskItem
"""

from __future__ import annotations

from .check import LLMCheckRemediationOutput
from .decompose import LLMDecomposeOutput
from .match import LLMMatchFillParams, LLMMatchFuncOutput
from .order import LLMSiblingOrderOutput
from .task import LLMTaskItem

__all__ = [
    "LLMCheckRemediationOutput",
    "LLMDecomposeOutput",
    "LLMMatchFillParams",
    "LLMMatchFuncOutput",
    "LLMSiblingOrderOutput",
    "LLMTaskItem",
]
