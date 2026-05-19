# NRE on Google Cloud Run

Deploys the **Neurosymbolic Reasoning Engine** HTTP API (`nre.server.app:app`): kernel primitives via `/v1/chat/completions`, health, and models list.

Same overall flow as other CoreThink Cloud Run services (e.g. `dev-jay-glm5`): Docker → `gcr.io` → `gcloud run deploy`.

## Prerequisites

- `gcloud` authenticated, Cloud Run + Container Registry (or Artifact Registry) enabled  
- Docker  
- `OPENROUTER_API_KEY` (kernel and base chat use OpenRouter)

## Deploy

From this directory:

```bash
export OPENROUTER_API_KEY='sk-or-...'
chmod +x deploy_to_gcp.sh
./deploy_to_gcp.sh
```

Non-interactive (e.g. CI):

```bash
./deploy_to_gcp.sh --skip-confirm
```

Options: `--project-id`, `--service-name`, `--region`, `--tag`, `--openrouter-key`, `--help`.

## Endpoints (after deploy)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Service metadata + endpoint list |
| GET | `/health` | Liveness |
| GET | `/v1/models` | Lists `nre-kernel` + configured OpenRouter model |
| POST | `/v1/chat/completions` | Kernel trace → base model (OpenAI-style body) |

Responses can include `nre_kernel_trace`, `nre_primitive_log`, and `nre` (see main README / server schemas).

## Local Docker smoke test

```bash
# From repository root
docker build -f deployment/gcp/Dockerfile -t nre-kernel:local .
docker run --rm -p 8080:8080 -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" nre-kernel:local
curl -sS http://127.0.0.1:8080/health
```

## Cloud Run defaults

- Memory **4Gi**, CPU **2** (heavy scientific deps in the package)  
- Timeout **900s** (long kernel + LLM)  
- **Unauthenticated** ingress (`--allow-unauthenticated`); lock down with IAM / IAP for production  

Tune in `deploy_to_gcp.sh` if the image slimming removes optional deps later.

## Build context

The Dockerfile expects the **repository root** as build context. The script sets `REPO_ROOT` automatically.
