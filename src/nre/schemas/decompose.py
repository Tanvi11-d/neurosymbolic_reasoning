"""DECOMPOSE (sub-process 1) structured output."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .task import LLMTaskItem


class LLMDecomposeOutput(BaseModel):
    tasks: list[LLMTaskItem] = Field(description="Unit tasks, each maps to one tool call")
    config_keys: dict[str, Any] = Field(
        default_factory=dict,
        description="Scratchpad seed keys extracted from gamma and the request",
    )
    reasoning: str = Field(default="", description="Brief decomposition rationale")

    def to_raw_decompose_dict(self) -> dict[str, Any]:
        turns: list[dict[str, Any]] = []
        for t in self.tasks:
            turns.append({
                "id": t.id,
                "text": t.description,
                "depends_on": list(t.depends_on),
                "tool_name": t.tool_name,
                "parameters": dict(t.parameters),
            })
        return {
            "turns": turns,
            "config": dict(self.config_keys),
            "tools": {},
            "reasoning": self.reasoning,
        }
