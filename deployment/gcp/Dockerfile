# NRE HTTP API (FastAPI) for Google Cloud Run
# Build from repository root:
#   docker build -f deployment/gcp/Dockerfile -t nre-kernel .

FROM python:3.12-slim-bookworm

WORKDIR /app

# RDKit / scientific wheels often need build tools; curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install package + server + OpenRouter client stack (see pyproject optional-deps)
COPY pyproject.toml README.md ./
COPY src ./src
# Ship the default kernel config at /app/config.yaml so the server's
# discover_config_path() picks it up automatically (cwd = /app at runtime).
# Override at deploy time by mounting a replacement file or by setting
# ``NRE_CONFIG=/path/to/config.yaml`` in the environment.
COPY config.yaml ./config.yaml
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir ".[server,llm]"

ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV HOST=0.0.0.0

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f "http://127.0.0.1:${PORT}/health" || exit 1

# Cloud Run sets PORT; match dev-jay-glm5 pattern
CMD ["sh", "-c", "exec uvicorn nre.server.app:app --host 0.0.0.0 --port ${PORT} --log-level info"]
