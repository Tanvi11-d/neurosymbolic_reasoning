"""Convert LLM structured task rows into kernel :class:`UnitTask` (symbolic side)."""

from __future__ import annotations

from nre.schemas import LLMTaskItem

from .deterministic import UnitTask


def llm_task_item_to_unit_task(item: LLMTaskItem) -> UnitTask:
    return UnitTask(
        id=item.id,
        text=item.description,
        depends_on=tuple(item.depends_on),
        tool_name=item.tool_name,
        parameters=dict(item.parameters),
    )
