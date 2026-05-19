"""YAML config loader for LLM settings.

Allows users to pick a model (e.g. ``z-ai/glm-4.7``) and provider routing
(e.g. ``cerebras/fp16``) without touching code. Entirely opt-in: when no
config file is found, :class:`OpenRouterSettings` defaults are unchanged.

Resolution precedence (highest → lowest):
1. Explicit constructor args.
2. Environment variables (``OPENROUTER_API_KEY``).
3. YAML config file (discovered via ``NRE_CONFIG`` / ``./config.yaml`` /
   ``~/.config/nre/config.yaml``).
4. Hardcoded defaults in :class:`OpenRouterSettings`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nre.llm.config import (
    DEFAULT_CONFIG_PATHS,
    NreConfig,
    discover_config_path,
    load_config,
)
from nre.llm.openrouter import OpenRouterSettings


# ── YAML loading ─────────────────────────────────────────────────────

def test_load_config_parses_minimal_yaml(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "llm:\n"
        "  model: z-ai/glm-4.7\n",
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert isinstance(loaded, NreConfig)
    assert loaded.model == "z-ai/glm-4.7"


def test_load_config_full_yaml_block(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
llm:
  model: z-ai/glm-4.7
  openrouter:
    base_url: https://openrouter.ai/api/v1
    temperature: 0.1
    max_tokens: 16384
    max_retries: 5
    timeout_seconds: 60.0
    provider_order:
      - cerebras/fp16
    reasoning:
      enabled: false
""".strip(),
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert loaded.model == "z-ai/glm-4.7"
    assert loaded.base_url == "https://openrouter.ai/api/v1"
    assert loaded.temperature == 0.1
    assert loaded.max_tokens == 16384
    assert loaded.max_retries == 5
    assert loaded.timeout_seconds == 60.0
    assert loaded.provider_order == ("cerebras/fp16",)
    assert loaded.reasoning == {"enabled": False}


def test_load_config_empty_yaml_returns_empty_config(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("", encoding="utf-8")
    loaded = load_config(cfg)
    assert loaded.model is None
    assert loaded.provider_order is None


def test_load_config_yml_extension_also_accepted(tmp_path: Path):
    """``.yml`` is just as valid as ``.yaml``."""
    cfg = tmp_path / "config.yml"
    cfg.write_text("llm:\n  model: test-model\n", encoding="utf-8")
    loaded = load_config(cfg)
    assert loaded.model == "test-model"


def test_load_config_yaml_with_comments(tmp_path: Path):
    """Comments are a real YAML advantage over TOML for inline tables."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
# Top-level config for the NRE kernel.
llm:
  # Pick any OpenRouter-hosted model.
  model: z-ai/glm-4.7

  openrouter:
    # Pin Cerebras for highest throughput.
    provider_order:
      - cerebras/fp16
    # Disable reasoning for structured-JSON kernel calls.
    reasoning:
      enabled: false  # GLM 4.7 otherwise burns tokens on chain-of-thought
""".strip(),
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert loaded.model == "z-ai/glm-4.7"
    assert loaded.provider_order == ("cerebras/fp16",)
    assert loaded.reasoning == {"enabled": False}


def test_load_config_tool_completion_block(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
llm:
  model: z-ai/glm-4.7
  openrouter:
    provider_order:
      - cerebras/fp16
  tool_completion:
    model: minimax/minimax-m2.7
    openrouter:
      provider_order:
        - sambanova
      reasoning:
        enabled: false
""".strip(),
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert loaded.model == "z-ai/glm-4.7"
    assert loaded.tool_completion is not None
    assert loaded.tool_completion.model == "minimax/minimax-m2.7"
    assert loaded.tool_completion.provider_order == ("sambanova",)
    assert loaded.tool_completion.reasoning == {"enabled": False}


def test_load_config_routing_block(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
llm:
  model: z-ai/glm-4.7
  routing:
    enabled: true
    direct_model: anthropic/claude-opus-4.7
  openrouter:
    provider_order:
      - cerebras/fp16
""".strip(),
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert loaded.routing is not None
    assert loaded.routing.enabled is True
    assert loaded.routing.direct_model == "anthropic/claude-opus-4.7"


def test_load_config_routing_classifier_block(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
llm:
  model: z-ai/glm-4.7
  routing:
    enabled: true
    direct_model: anthropic/claude-opus-4.7
    classifier:
      model: openai/gpt-oss-120b
      openrouter:
        provider_order:
          - cerebras/fp16
  openrouter:
    provider_order:
      - cerebras/fp16
""".strip(),
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert loaded.routing is not None
    assert loaded.routing.classifier is not None
    assert loaded.routing.classifier.model == "openai/gpt-oss-120b"
    assert loaded.routing.classifier.provider_order == ("cerebras/fp16",)


def test_load_config_routing_disabled(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
llm:
  model: z-ai/glm-4.7
  routing:
    enabled: false
    direct_model: anthropic/claude-opus-4.7
""".strip(),
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert loaded.routing is not None
    assert loaded.routing.enabled is False


def test_load_config_missing_file_raises(tmp_path: Path):
    cfg = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(cfg)


def test_load_config_malformed_yaml_raises(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("llm: [invalid: yaml: here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML"):
        load_config(cfg)


def test_load_config_unknown_keys_ignored(tmp_path: Path):
    """Unknown keys are silently ignored so newer configs don't break older code."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
llm:
  model: z-ai/glm-4.7
  future_key: ok
  openrouter:
    unknown_tunable: 42
""".strip(),
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert loaded.model == "z-ai/glm-4.7"


def test_load_config_empty_provider_order_means_auto_route(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
llm:
  openrouter:
    provider_order: []
""".strip(),
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert loaded.provider_order == ()


# ── Reasoning-model knob ─────────────────────────────────────────────

def test_reasoning_block_parsed_from_yaml(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
llm:
  openrouter:
    reasoning:
      enabled: false
""".strip(),
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert loaded.reasoning == {"enabled": False}


def test_reasoning_effort_variant(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
llm:
  openrouter:
    reasoning:
      effort: low
""".strip(),
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert loaded.reasoning == {"effort": "low"}


def test_reasoning_rejects_non_mapping(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
llm:
  openrouter:
    reasoning: "invalid"
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reasoning"):
        load_config(cfg)


# ── discover_config_path: respects env + fallback locations ──────────

def test_discover_config_respects_nre_config_env(tmp_path: Path, monkeypatch):
    target = tmp_path / "custom.yaml"
    target.write_text("llm:\n  model: x\n", encoding="utf-8")
    monkeypatch.setenv("NRE_CONFIG", str(target))
    assert discover_config_path() == target


def test_discover_config_picks_local_config_yaml(tmp_path: Path, monkeypatch):
    local = tmp_path / "config.yaml"
    local.write_text("llm:\n  model: x\n", encoding="utf-8")
    monkeypatch.delenv("NRE_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "_empty_home"))
    assert discover_config_path() == local


def test_discover_config_yml_extension_also_found(tmp_path: Path, monkeypatch):
    yml = tmp_path / "config.yml"
    yml.write_text("llm:\n  model: yml_model\n", encoding="utf-8")
    monkeypatch.delenv("NRE_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "_empty_home"))
    assert discover_config_path() == yml


def test_discover_config_yaml_beats_yml_when_both_present(
    tmp_path: Path, monkeypatch
):
    """``config.yaml`` is preferred over ``config.yml`` when both exist."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("llm:\n  model: yaml_wins\n", encoding="utf-8")
    yml_file = tmp_path / "config.yml"
    yml_file.write_text("llm:\n  model: yml_loses\n", encoding="utf-8")
    monkeypatch.delenv("NRE_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "_empty_home"))
    assert discover_config_path() == yaml_file


def test_discover_config_returns_none_when_nothing_found(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("NRE_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "_empty_home"))
    assert discover_config_path() is None


def test_default_paths_contains_known_locations():
    """Sanity — the documented fallback paths must include a local YAML
    and a user-scoped config so discovery isn't surprising."""
    paths = [str(p) for p in DEFAULT_CONFIG_PATHS]
    assert any("config.yaml" in p for p in paths), f"paths={paths}"
    assert any(".config/nre" in p for p in paths), f"paths={paths}"
    # Strictly YAML now — no TOML fallback.
    assert not any(".toml" in p for p in paths), f"paths={paths}"


# ── OpenRouterSettings integration — end-to-end ──────────────────────

def test_settings_picks_up_yaml_config_model_when_env_unset(
    tmp_path: Path, monkeypatch
):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("llm:\n  model: z-ai/glm-4.7\n", encoding="utf-8")
    monkeypatch.setenv("NRE_CONFIG", str(cfg))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    settings = OpenRouterSettings()
    assert settings.model == "z-ai/glm-4.7"
    assert settings.api_key == "test-key"


def test_settings_explicit_constructor_beats_config(
    tmp_path: Path, monkeypatch
):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("llm:\n  model: z-ai/glm-4.7\n", encoding="utf-8")
    monkeypatch.setenv("NRE_CONFIG", str(cfg))
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    settings = OpenRouterSettings(model="custom/model")
    assert settings.model == "custom/model"


def test_settings_provider_order_from_yaml_is_applied(
    tmp_path: Path, monkeypatch
):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
llm:
  openrouter:
    provider_order:
      - cerebras/fp16
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("NRE_CONFIG", str(cfg))
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    settings = OpenRouterSettings()
    assert settings.extra_body == {"provider": {"order": ["cerebras/fp16"]}}


def test_settings_reasoning_flows_into_extra_body(
    tmp_path: Path, monkeypatch
):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
llm:
  openrouter:
    provider_order:
      - cerebras/fp16
    reasoning:
      enabled: false
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("NRE_CONFIG", str(cfg))
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    settings = OpenRouterSettings()
    assert settings.extra_body == {
        "provider": {"order": ["cerebras/fp16"]},
        "reasoning": {"enabled": False},
    }


def test_settings_back_compat_when_no_config_present(monkeypatch):
    """No config file, no NRE_CONFIG → hardcoded MiniMax defaults."""
    monkeypatch.delenv("NRE_CONFIG", raising=False)
    monkeypatch.setenv("HOME", "/nonexistent_home_for_test")
    monkeypatch.chdir("/tmp")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    settings = OpenRouterSettings()
    assert settings.model == "minimax/minimax-m2.7"
    assert settings.extra_body == {"provider": {"order": ["minimax/fp8"]}}
