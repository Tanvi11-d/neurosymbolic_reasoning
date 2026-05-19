"""YAML config loader for LLM transport settings.

User-facing config file that lets operators pick an OpenRouter model and
provider routing without touching code. The file is optional — if none
found, :class:`nre.llm.openrouter.OpenRouterSettings` keeps its hardcoded
defaults.

**Resolution precedence** (highest wins):

1. Explicit ``OpenRouterSettings(...)`` constructor args.
2. Environment variables (``OPENROUTER_API_KEY``).
3. YAML config file discovered via :func:`discover_config_path`.
4. Hardcoded defaults in :class:`OpenRouterSettings`.

**Example** (``./config.yaml``):

.. code-block:: yaml

    llm:
      model: z-ai/glm-4.7
      openrouter:
        provider_order:
          - cerebras/fp16
        reasoning:
          enabled: false
      # Optional: HTTP server only — OpenRouter call that returns ``tool_calls``
      # after the kernel. Primitives keep using ``llm`` above.
      tool_completion:
        model: minimax/minimax-m2.7
        openrouter:
          provider_order:
            - sambanova

**Discovery order**:

1. ``$NRE_CONFIG`` (explicit path; any readable YAML file).
2. ``./config.yaml``
3. ``./config.yml``
4. ``~/.config/nre/config.yaml``
5. ``~/.config/nre/config.yml``
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ToolCompletionConfig:
    """OpenRouter overrides for the post-kernel tool-calling completion only.

    Parsed from ``llm.tool_completion`` in YAML. If ``model`` is unset, the
    server uses the main ``llm`` settings for that call (backward compatible).
    Any field left ``None`` falls back to the kernel :class:`OpenRouterSettings`
    when building the dedicated client.
    """

    model: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_retries: int | None = None
    timeout_seconds: float | None = None
    # ``None`` = omit ``provider.order`` (OpenRouter auto-routes); ``()`` = same.
    provider_order: tuple[str, ...] | None = None
    reasoning: dict[str, object] | None = None


@dataclass(frozen=True)
class RouteClassifierConfig:
    """OpenRouter overrides for the HTTP route classifier only (structured JSON)."""

    model: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_retries: int | None = None
    timeout_seconds: float | None = None
    provider_order: tuple[str, ...] | None = None
    reasoning: dict[str, object] | None = None


@dataclass(frozen=True)
class RoutingConfig:
    """HTTP-only: classify tool requests; non-agentic completions use Opus (or
    ``direct_model``) instead of the neurosymbolic kernel."""

    enabled: bool = True
    direct_model: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_retries: int | None = None
    timeout_seconds: float | None = None
    provider_order: tuple[str, ...] | None = None
    reasoning: dict[str, object] | None = None
    classifier: RouteClassifierConfig | None = None


@dataclass(frozen=True)
class NreConfig:
    """Parsed config. All fields optional; ``None`` means "not set".

    Unknown keys are silently ignored so older code can read newer config
    files without surprises.
    """

    # ── llm block ────────────────────────────────────────────────────
    model: str | None = None

    # ── llm.openrouter block ─────────────────────────────────────────
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_retries: int | None = None
    timeout_seconds: float | None = None
    # Tuple because dataclasses(frozen=True) forbids mutable defaults.
    # ``None`` = "use built-in default"; ``()`` = "auto-route (no provider pin)".
    provider_order: tuple[str, ...] | None = None
    # Reasoning-model controls (forwarded to OpenRouter ``extra_body.reasoning``).
    # Common shapes:
    #   {"enabled": False}    → disable reasoning (fastest for structured JSON)
    #   {"effort": "low"}     → light reasoning
    #   {"max_tokens": 1024}  → cap reasoning tokens
    reasoning: dict[str, object] | None = None

    # ── llm.routing (HTTP server only) ───────────────────────────────
    routing: RoutingConfig | None = None

    # ── llm.tool_completion ───────────────────────────────────────────
    tool_completion: ToolCompletionConfig | None = None


def discover_config_path() -> Path | None:
    """
        Find the config file. Checks $NRE_CONFIG first, then the default locations.
        Returns None if nothing is found. Missing $NRE_CONFIG file is ignored, not an error.
    """

    env = os.environ.get("NRE_CONFIG", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    
    cwd = Path.cwd()
    home = Path.home() / ".config" / "nre"
    default_paths = (
        cwd / "config.yaml",
        cwd / "config.yml",
        home / "config.yaml",
        home / "config.yml",
    )
    for candidate in default_paths:
        if candidate.is_file():
            return candidate
    return None

def load_config(path: Path) -> NreConfig:
    """Load and parse a YAML config file. Raises FileNotFoundError if missing, ValueError if invalid."""
    if not path.is_file():
        raise FileNotFoundError(str(path))

    data = _parse_yaml(path)

    # Empty YAML file is fine — just treat it as empty config.
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"expected top-level mapping in {path}, got {type(data).__name__}")

    # ── Read top-level llm + openrouter sections ──────────────────────────

    llm = data.get("llm") or {}
    if not isinstance(llm, dict):
        raise ValueError(f"`llm` must be a mapping, got {type(llm).__name__}")

    openrouter = llm.get("openrouter") or {}
    if not isinstance(openrouter, dict):
        raise ValueError(f"`llm.openrouter` must be a mapping, got {type(openrouter).__name__}")

    main_provider_order = _parse_provider_order(openrouter, "llm.openrouter.provider_order")
    main_reasoning      = _parse_reasoning(openrouter, "llm.openrouter.reasoning")

    # ── tool_completion section (optional) ────────────────────────────────

    tool_completion = None
    tc_raw = llm.get("tool_completion")

    if tc_raw is not None:
        if not isinstance(tc_raw, dict):
            raise ValueError(f"`llm.tool_completion` must be a mapping, got {type(tc_raw).__name__}")

        tc_openrouter = tc_raw.get("openrouter") or {}
        if not isinstance(tc_openrouter, dict):
            raise ValueError(f"`llm.tool_completion.openrouter` must be a mapping, got {type(tc_openrouter).__name__}")

        tool_completion = ToolCompletionConfig(
            model           = _opt_str(tc_raw.get("model")),
            base_url        = _opt_str(tc_openrouter.get("base_url")),
            temperature     = _opt_float(tc_openrouter.get("temperature")),
            max_tokens      = _opt_int(tc_openrouter.get("max_tokens")),
            max_retries     = _opt_int(tc_openrouter.get("max_retries")),
            timeout_seconds = _opt_float(tc_openrouter.get("timeout_seconds")),
            provider_order  = _parse_provider_order(tc_openrouter, "llm.tool_completion.openrouter.provider_order"),
            reasoning       = _parse_reasoning(tc_openrouter, "llm.tool_completion.openrouter.reasoning"),
        )

    # ── routing section (optional) ────────────────────────────────────────

    routing = None
    routing_raw = llm.get("routing")

    if routing_raw is not None:
        if not isinstance(routing_raw, dict):
            raise ValueError(f"`llm.routing` must be a mapping, got {type(routing_raw).__name__}")

        # routing.enabled defaults to True if not set.
        enabled_raw = routing_raw.get("enabled")
        if enabled_raw is None:
            routing_enabled = True
        elif isinstance(enabled_raw, bool):
            routing_enabled = enabled_raw
        else:
            raise ValueError(f"llm.routing.enabled must be a boolean, got {type(enabled_raw).__name__}")

        routing_openrouter = routing_raw.get("openrouter") or {}
        if not isinstance(routing_openrouter, dict):
            raise ValueError(f"`llm.routing.openrouter` must be a mapping, got {type(routing_openrouter).__name__}")

        # ── classifier sub-section (optional) ────────────────────────────

        classifier = None
        classifier_raw = routing_raw.get("classifier")

        if classifier_raw is not None:
            if not isinstance(classifier_raw, dict):
                raise ValueError(f"`llm.routing.classifier` must be a mapping, got {type(classifier_raw).__name__}")

            clf_openrouter = classifier_raw.get("openrouter") or {}
            if not isinstance(clf_openrouter, dict):
                raise ValueError(f"`llm.routing.classifier.openrouter` must be a mapping, got {type(clf_openrouter).__name__}")

            classifier = RouteClassifierConfig(
                model           = _opt_str(classifier_raw.get("model")),
                base_url        = _opt_str(clf_openrouter.get("base_url")),
                temperature     = _opt_float(clf_openrouter.get("temperature")),
                max_tokens      = _opt_int(clf_openrouter.get("max_tokens")),
                max_retries     = _opt_int(clf_openrouter.get("max_retries")),
                timeout_seconds = _opt_float(clf_openrouter.get("timeout_seconds")),
                provider_order  = _parse_provider_order(clf_openrouter, "llm.routing.classifier.openrouter.provider_order"),
                reasoning       = _parse_reasoning(clf_openrouter, "llm.routing.classifier.openrouter.reasoning"),
            )

        routing = RoutingConfig(
            enabled         = routing_enabled,
            direct_model    = _opt_str(routing_raw.get("direct_model")),
            base_url        = _opt_str(routing_openrouter.get("base_url")),
            temperature     = _opt_float(routing_openrouter.get("temperature")),
            max_tokens      = _opt_int(routing_openrouter.get("max_tokens")),
            max_retries     = _opt_int(routing_openrouter.get("max_retries")),
            timeout_seconds = _opt_float(routing_openrouter.get("timeout_seconds")),
            provider_order  = _parse_provider_order(routing_openrouter, "llm.routing.openrouter.provider_order"),
            reasoning       = _parse_reasoning(routing_openrouter, "llm.routing.openrouter.reasoning"),
            classifier      = classifier,
        )

    # ── Build and return the final config object ──────────────────────────

    return NreConfig(
        model           = _opt_str(llm.get("model")),
        base_url        = _opt_str(openrouter.get("base_url")),
        temperature     = _opt_float(openrouter.get("temperature")),
        max_tokens      = _opt_int(openrouter.get("max_tokens")),
        max_retries     = _opt_int(openrouter.get("max_retries")),
        timeout_seconds = _opt_float(openrouter.get("timeout_seconds")),
        provider_order  = main_provider_order,
        reasoning       = main_reasoning,
        routing         = routing,
        tool_completion = tool_completion,
    )

def _parse_provider_order(source, field_path):
    """Parse provider_order from a dict. Returns None if missing, tuple if valid list."""
    raw = source.get("provider_order")
    if raw is None:
        return None
    if isinstance(raw, list):
        return tuple(str(x) for x in raw)
    raise ValueError(f"{field_path} must be a list, got {type(raw).__name__}")

def _parse_reasoning(source, field_path):
    """Parse reasoning from a dict. Returns None if missing, dict if valid."""
    raw = source.get("reasoning")
    if raw is None:
        return None
    if isinstance(raw, dict):
        return dict(raw)
    raise ValueError(f"{field_path} must be a mapping, got {type(raw).__name__}")


def load_discovered_config() -> NreConfig | None:
    """Discover and load the config file. Returns None if no config file is found."""
    path = discover_config_path()
    if path is None:
        return None
    return load_config(path)


# ── private parsers ──────────────────────────────────────────────────


def _parse_yaml(path: Path) -> object:
    try:
        import yaml
    except ImportError as e:  # pragma: no cover — pyyaml is in both [llm] and [server]
        raise ValueError(
            f"YAML config {path} requires pyyaml — install with `pip install pyyaml`"
        ) from e
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"invalid YAML in {path}: {e}") from e


def _opt_str(v: object) -> str | None:
    if v is None:
        return None
    if not isinstance(v, str):
        raise ValueError(f"expected string, got {type(v).__name__}: {v!r}")
    return v


def _opt_int(v: object) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"expected integer, got {type(v).__name__}: {v!r}")
    return v


def _opt_float(v: object) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError(f"expected number, got bool: {v!r}")
    if isinstance(v, (int, float)):
        return float(v)
    raise ValueError(f"expected number, got {type(v).__name__}: {v!r}")