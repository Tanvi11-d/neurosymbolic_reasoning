"""MATCH-FUNCS-AND-PARAMS LLM outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMMatchFuncOutput(BaseModel):
    """Pick one registry tool or null."""

    tool_name: str | None = Field(
        default=None,
        description="Exact registry tool name, or null",
    )
    reasoning: str = Field(default="", description="Brief rationale")


class LLMMatchFillParams(BaseModel):
    """Fill parameters not resolved from τ / scratchpad."""

    values: dict[str, Any] = Field(
        default_factory=dict,
        description="Map param name → value; omit keys you cannot ground",
    )
