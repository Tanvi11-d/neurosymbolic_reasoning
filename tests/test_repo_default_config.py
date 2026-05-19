"""Repo-root ``config.yaml`` is the shipped default.

This config file is committed at the repo root, baked into the Docker
image (``deployment/gcp/Dockerfile``), and is auto-discovered at runtime
because ``Path.cwd() / 'config.yaml'`` is the first entry in
:data:`nre.llm.config.DEFAULT_CONFIG_PATHS`.

These tests pin the shipped defaults so an accidental delete, rename, or
mis-edit is caught immediately instead of propagating through a deploy.

If you need to *change* a shipped default intentionally, update the YAML
and the assertion together.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nre.llm.config import load_config


REPO_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def test_repo_config_yaml_exists_at_repo_root():
    """Required for Dockerfile ``COPY config.yaml`` and for server
    auto-discovery from the repo-root working directory."""
    assert REPO_CONFIG.is_file(), (
        f"shipped default config is missing from {REPO_CONFIG}; the "
        f"Dockerfile and runtime auto-discovery both depend on it"
    )


def test_repo_config_yaml_parses():
    """Syntax + loader compatibility — malformed YAML would break every
    server boot. Fail fast in tests rather than in production."""
    cfg = load_config(REPO_CONFIG)
    # Smoke assertion — any model string present.
    assert isinstance(cfg.model, str) and cfg.model


def test_repo_config_pins_expected_defaults():
    """Pin the shipped routing choices. Change these together with the YAML
    when you intentionally switch the default model or provider.
    """
    cfg = load_config(REPO_CONFIG)
    assert cfg.model == "z-ai/glm-4.7", (
        f"shipped default model changed unexpectedly: {cfg.model!r}"
    )
    assert cfg.provider_order == ("cerebras/fp16",), (
        f"shipped provider pin changed unexpectedly: {cfg.provider_order!r}"
    )
    # Reasoning block must be a mapping (enabled true/false, effort, etc).
    assert isinstance(cfg.reasoning, dict), (
        f"shipped reasoning block must be a mapping: {cfg.reasoning!r}"
    )
    assert cfg.tool_completion is not None
    assert cfg.tool_completion.model == "minimax/minimax-m2.7", (
        f"shipped tool_completion model changed: {cfg.tool_completion.model!r}"
    )
    assert cfg.tool_completion.provider_order == ("sambanova",), (
        f"shipped tool_completion provider changed: {cfg.tool_completion.provider_order!r}"
    )
    assert cfg.routing is not None and cfg.routing.enabled
    assert cfg.routing.direct_model == "anthropic/claude-opus-4.7"
    assert cfg.routing.classifier is not None
    assert cfg.routing.classifier.model == "openai/gpt-oss-120b"
    assert cfg.routing.classifier.provider_order == ("cerebras/fp16",)
