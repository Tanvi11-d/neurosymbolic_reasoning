"""CHECK-PREREQUISITES LLM remediation output."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .task import LLMTaskItem


class LLMCheckRemediationOutput(BaseModel):
    """Optional remediation unit task t′ when a prerequisite is Absent or ¬Met."""

    task: LLMTaskItem | None = Field(
        default=None,
        description="Concrete sub-task to run first (retrieve value, satisfy predicate); null if none",
    )
    reasoning: str = Field(default="", description="Why this remediation fits the CHECK outcome")
