# neurosymbolic-reasoning-engine

**Docs:** [Step-by-step pipeline & debugging](docs/DEBUGGING_PIPELINE_WALKTHROUGH.md)

## Developer setup

Python **3.11+** (see `pyproject.toml`). To install from source with `uv`:

```sh
git clone https://github.com/CoreThink-AI/neurosymbolic-reasoning-engine
cd neurosymbolic-reasoning-engine
uv venv -p 3.11
source .venv/bin/activate
uv pip install -e .
```

After an editable install, console scripts such as `nre-verify-decompose` are on your `PATH`. If you skip install, use `PYTHONPATH=src` or the `scripts/verify_*.py` wrappers (they prepend `src`).

## Tests

From the repo root:

```sh
# Full suite (pytest is configured with pythonpath = ["src"] in pyproject.toml)
pytest

# Quiet summary
pytest -q

# Match / harness only
pytest tests/test_match_harness.py tests/test_verify_match_cli.py -q
```

## Verification CLIs (kernel benchmarks & smoke)

These tools exercise **DECOMPOSE**, **GET-ORDER**, **CHECK-PREREQUISITES**, and **MATCH** against JSON fixtures. Several **benchmark** subcommands call **OpenRouter**; set `OPENROUTER_API_KEY` (and optional model overrides via `--model` where supported).

| Command | Role |
|--------|------|
| `nre-verify-decompose` | DECOMPOSE fixtures + `benchmark` (parallel `-j`) |
| `nre-verify-order` | GET-ORDER smoke + `benchmark` (OpenRouter sibling order) |
| `nre-verify-check` | CHECK smoke + `benchmark` (includes LLM remediation rows) |
| `nre-verify-match` | MATCH smoke + `benchmark` (demo tool registry; LLM on some rows) |

**Without install** (repo root):

```sh
PYTHONPATH=src python -m nre.cli.verify_decompose turn1
PYTHONPATH=src python -m nre.cli.verify_order parallel
PYTHONPATH=src python -m nre.cli.verify_check ready
PYTHONPATH=src python -m nre.cli.verify_match scratchpad
```

Or use the wrappers:

```sh
python scripts/verify_decompose.py turn1
python scripts/verify_order.py chain
python scripts/verify_check.py benchmark --jobs 8
python scripts/verify_match.py tau_wins
```

**After `pip install -e .` or `uv pip install -e .`:**

```sh
nre-verify-decompose turn2
nre-verify-decompose benchmark --jobs 8

nre-verify-order parallel
nre-verify-order benchmark --jobs 4

nre-verify-check ready
nre-verify-check benchmark --jobs 8

nre-verify-match scratchpad
nre-verify-match benchmark --jobs 4 --model <openrouter-model-id>
```

Use `--fixture path/to.json` on each command where the CLI supports it (see module docstrings under `src/nre/cli/`).

## HTTP server (`nre-serve`)

FastAPI server that runs the five kernel primitives, builds a **reasoning trace**, then calls a **base** OpenRouter model for `/v1/chat/completions` (`tool_calls` / `content`). Install optional deps:

```sh
uv pip install -e ".[server]"
# or: pip install -e ".[server]"
```

Set **`OPENROUTER_API_KEY`**: kernel primitives (DECOMPOSE / ORDER / CHECK / MATCH) and the base chat step all use OpenRouter. Default models come from `OpenRouterSettings` (see `src/nre/llm/openrouter.py`).

**After editable install:**

```sh
nre-serve --host 0.0.0.0 --port 8000
nre-serve --port 8000 --log-level debug
```

**Without changing `PATH` (repo root):**

```sh
python run_server.py --port 8000
```

**Without installing the package** (repo root, with `[server]` deps available):

```sh
PYTHONPATH=src python run_server.py --port 8000
```

**Useful endpoints:**

| Method | Path | Role |
|--------|------|------|
| `GET` | `/health` | Liveness |
| `GET` | `/v1/models` | Lists `nre-kernel` + configured OpenRouter model |
| `POST` | `/v1/chat/completions` | Kernel → trace → base LLM (OpenAI-style `messages` + `tools`) |

Implementation: `src/nre/server/` (see `app.py`, `schemas.py`). Pipeline logs use the `nre.*` loggers (e.g. `nre.decompose`, `nre.openrouter`) when `nre-serve` sets `--log-level`.
