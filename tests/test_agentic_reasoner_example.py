"""Smoke tests for :mod:`nre.examples.agentic_reasoner` with mocked OpenRouter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from nre.examples.agentic_reasoner import IOTA_1, build_engine
from nre.schemas import LLMDecomposeOutput, LLMTaskItem


def _decompose_turn1() -> LLMDecomposeOutput:
    return LLMDecomposeOutput(
        tasks=[
            LLMTaskItem(
                id="t1",
                description="nav",
                tool_name="cd",
                depends_on=[],
                parameters={},
            ),
            LLMTaskItem(
                id="t2",
                description="mkdir archive",
                tool_name="mkdir",
                depends_on=["t1"],
                parameters={},
            ),
            LLMTaskItem(
                id="t3",
                description="move report",
                tool_name="mv",
                depends_on=["t2"],
                parameters={},
            ),
        ],
        config_keys={
            "cwd": "/",
            "cwd_is_workspace": False,
            "archive_exists": False,
        },
    )


def _decompose_turn2() -> LLMDecomposeOutput:
    return LLMDecomposeOutput(
        tasks=[
            LLMTaskItem(
                id="u1",
                description="cd archive",
                tool_name="cd",
                depends_on=[],
                parameters={},
            ),
            LLMTaskItem(
                id="u2",
                description="grep revenue",
                tool_name="grep",
                depends_on=["u1"],
                parameters={},
            ),
        ],
        config_keys={},
    )


def _make_turn2_tools(scratchpad: dict, log: list) -> dict:
    def cd(folder: str):
        log.append({"name": "cd", "kwargs": {"folder": folder}})
        scratchpad["cwd"] = "/workspace/archive"
        return {}

    def grep(file_name: str, pattern: str):
        log.append({
            "name": "grep",
            "kwargs": {"file_name": file_name, "pattern": pattern},
        })
        return {"hits": 1}

    return {
        "cd": {"description": "Tool cd", "params": ["folder"], "execute": cd},
        "grep": {
            "description": "Tool grep",
            "params": ["file_name", "pattern"],
            "execute": grep,
        },
    }


@patch("nre.examples.agentic_reasoner.OpenRouterClient")
def test_two_turn_session_mocked_llm_executes_tools(mock_cls: MagicMock) -> None:
    mock_inst = MagicMock()
    mock_cls.return_value = mock_inst
    mock_inst.chat.side_effect = [_decompose_turn1(), _decompose_turn2()]

    engine = build_engine()
    state: dict = {}
    log: list = []
    from nre.examples.agentic_reasoner import IOTA_2, _make_turn1_tools

    tools1 = _make_turn1_tools(state, log)
    out1 = engine.run(
        "agent_turn",
        iota=IOTA_1,
        gamma={},
        tools=tools1,
        scratchpad_carry=state,
        turn_context_by_task_id={
            "t1": {"folder": "workspace"},
            "t2": {"dir_name": "archive"},
            "t3": {"source": "report.txt", "destination": "archive"},
        },
    )
    assert out1["cycle"] is False
    assert [s.get("status") for s in out1["steps"]] == ["success"] * 3

    tools2 = _make_turn2_tools(state, log)
    out2 = engine.run(
        "agent_turn",
        iota=IOTA_2,
        gamma={},
        tools=tools2,
        scratchpad_carry=state,
        turn_context_by_task_id={
            "u1": {"folder": "archive"},
            "u2": {"file_name": "report.txt", "pattern": "revenue"},
        },
    )
    assert out2["cycle"] is False
    assert [s.get("status") for s in out2["steps"]] == ["success"] * 2
    assert len(log) == 5


@patch("nre.examples.agentic_reasoner.OpenRouterClient")
def test_agent_turn_mocked_llm_matches_single_turn(mock_cls: MagicMock) -> None:
    mock_cls.return_value.chat.return_value = _decompose_turn1()
    engine = build_engine()
    state: dict = {}
    log: list = []
    from nre.examples.agentic_reasoner import _make_turn1_tools

    tools = _make_turn1_tools(state, log)
    out = engine.run(
        "agent_turn",
        iota=IOTA_1,
        gamma={},
        tools=tools,
        scratchpad_carry=state,
        turn_context_by_task_id={
            "t1": {"folder": "workspace"},
            "t2": {"dir_name": "archive"},
            "t3": {"source": "report.txt", "destination": "archive"},
        },
    )
    assert out["cycle"] is False
    assert out["ordered_task_ids"] == ["t1", "t2", "t3"]
