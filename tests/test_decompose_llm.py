"""DECOMPOSE path when config provides llm_client (mocked)."""

from unittest.mock import MagicMock

from nre.schemas import LLMDecomposeOutput, LLMTaskItem
from nre.primitives.deterministic import UnitTask
from nre.primitives.hybrid import DecomposeInput


def test_decompose_uses_llm_client_and_normalizes():
    parsed = LLMDecomposeOutput(
        tasks=[
            LLMTaskItem(
                id="t1",
                description="enter workspace",
                tool_name="cd",
                depends_on=[],
                parameters={"folder": "workspace"},
            ),
            LLMTaskItem(
                id="t2",
                description="make archive",
                tool_name="mkdir",
                depends_on=["t1"],
                parameters={"dir_name": "archive"},
            ),
        ],
        config_keys={"seed": 1},
        reasoning="two steps",
    )
    client = MagicMock()
    client.chat.return_value = parsed

    out = DecomposeInput(llm_client=client)(
        "Create archive under workspace",
        {"GorillaFileSystem": {"root": {}}},
        {
            "cd": {"params": ["folder"], "description": "cd"},
            "mkdir": {"params": ["dir_name"], "description": "mkdir"},
        },
    )

    client.chat.assert_called_once()
    call_kw = client.chat.call_args.kwargs
    assert call_kw["response_model"] is LLMDecomposeOutput

    assert out["reasoning"] == "two steps"
    assert out["config"] == {"seed": 1}
    assert out["tools"] == {}
    assert out["turns"] == [
        UnitTask(
            id="t1",
            text="enter workspace",
            depends_on=(),
            tool_name="cd",
            parameters={"folder": "workspace"},
        ),
        UnitTask(
            id="t2",
            text="make archive",
            depends_on=("t1",),
            tool_name="mkdir",
            parameters={"dir_name": "archive"},
        ),
    ]
