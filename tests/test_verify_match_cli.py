"""CLI smoke: ``nre-verify-match`` presets (subprocess, no install required)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nre.schemas import LLMMatchFillParams, LLMMatchFuncOutput

_REPO = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO / "src")
    return env


@pytest.mark.parametrize(
    "preset",
    [
        "scratchpad",
        "tau_wins",
        "tool_name_prefers",
        "bound",
        "match_func",
        "task_params",
    ],
)
def test_verify_match_cli_preset_smoke(preset: str):
    r = subprocess.run(
        [sys.executable, "-m", "nre.cli.verify_match", preset],
        cwd=str(_REPO),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert '"status"' in r.stdout


def test_verify_match_cli_benchmark_mocked_openrouter(monkeypatch):
    """benchmark subcommand loads cases; patch OpenRouterClient to avoid network."""

    from nre.cli import verify_match

    mock_client = MagicMock()

    def chat(messages, *, response_model=None, **kwargs):
        if response_model is LLMMatchFuncOutput:
            text = messages[-1]["content"].lower()
            if "pickme" in text:
                return LLMMatchFuncOutput(tool_name="pickme", reasoning="")
            return LLMMatchFuncOutput(tool_name="add", reasoning="")
        if response_model is LLMMatchFillParams:
            return LLMMatchFillParams(values={"a": 2, "b": 3})
        raise AssertionError(response_model)

    mock_client.chat.side_effect = chat

    monkeypatch.setattr(verify_match, "OpenRouterClient", lambda *a, **k: mock_client)
    monkeypatch.setattr(verify_match, "OpenRouterSettings", lambda: MagicMock())

    monkeypatch.setattr(sys, "argv", ["nre-verify-match", "benchmark", "-j", "1"])
    verify_match.main()
    assert mock_client.chat.call_count == 3
