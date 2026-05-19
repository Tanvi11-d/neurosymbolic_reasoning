"""GET-ORDER (sibling batch) structured output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMSiblingOrderOutput(BaseModel):
    """Permutation of sibling task ids in one Kahn wave."""

    ordered_task_ids: list[str] = Field(
        description="Permutation of the batch ids; best workflow order given context",
    )
    reasoning: str = Field(
        default="",
        description="Brief rationale for this ordering",
    )
