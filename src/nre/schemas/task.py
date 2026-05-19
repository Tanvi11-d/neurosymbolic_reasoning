"""Shared unit-task shape (DECOMPOSE rows, CHECK remediation payload, etc.)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMTaskItem(BaseModel):
    """One unit task / tool call."""

    id: str = Field(description="Unique task id, e.g. t1, t2")
    description: str = Field(
        default="",
        description="Short human-readable description of the unit task",
    )
    tool_name: str = Field(description="Exact tool name from the registry")
    depends_on: list[str] = Field(
        default_factory=list,
        description="Ids of tasks that must complete before this one",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter names to values explicit in the user message only",
    )
