#!/bin/bash

# ============================================================================
# GCP Cloud Run — Neurosymbolic Reasoning Engine (NRE) HTTP API
# ============================================================================
#
# Deploys the FastAPI app from nre.server.app:app (kernel + /v1/chat/completions).
#
# Usage:
#   export OPENROUTER_API_KEY='sk-or-...'
#   ./deploy_to_gcp.sh
#
# Options:
#   --project-id PROJECT_ID     GCP project (default: corethink-reasoner-prod)
#   --service-name NAME         Cloud Run service (default: nre-kernel)
#   --region REGION             (default: us-central1)
#   --tag TAG                   Image tag (default: latest)
#   --openrouter-key KEY        Optional; else uses OPENROUTER_API_KEY
#   --skip-confirm              Deploy without interactive y/n
#   --help
#
# ============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

SERVICE_NAME="nre-kernel"
REGION="us-central1"
TAG="latest"
PROJECT_ID="corethink-reasoner-prod"
OPENROUTER_KEY=""
SKIP_CONFIRM="0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# deployment/gcp -> repo root is two levels up
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile"

while [[ $# -gt 0 ]]; do
    case $1 in
        --project-id) PROJECT_ID="$2"; shift 2 ;;
        --service-name) SERVICE_NAME="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --openrouter-key) OPENROUTER_KEY="$2"; shift 2 ;;
        --skip-confirm) SKIP_CONFIRM="1"; shift ;;
        --help)
            sed -n '1,30p' "$0" | tail -n +2
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

print_status() { echo -e "${BLUE}==>${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }

if [[ -z "${OPENROUTER_KEY}" ]]; then
    OPENROUTER_KEY="${OPENROUTER_API_KEY:-}"
fi
if [[ -z "${OPENROUTER_KEY}" ]]; then
    print_error "Set OPENROUTER_API_KEY or pass --openrouter-key"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  NRE — Neurosymbolic Reasoning Engine (Cloud Run)                            ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Project:       ${PROJECT_ID}"
echo "  Service:       ${SERVICE_NAME}"
echo "  Region:        ${REGION}"
echo "  Image tag:     ${TAG}"
echo "  Repo root:     ${REPO_ROOT}"
echo "  OpenRouter:    ${OPENROUTER_KEY:0:12}..."
echo ""

if [[ "${SKIP_CONFIRM}" != "1" ]]; then
    read -p "Proceed with deployment? (y/n): " -n 1 -r
    echo
    if [[ ! ${REPLY:-n} =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

print_status "Checking gcloud..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &>/dev/null; then
    print_error "Run: gcloud auth login"
    exit 1
fi
print_success "gcloud OK"

print_status "Setting project ${PROJECT_ID}..."
gcloud config set project "${PROJECT_ID}"

print_status "Docker auth for GCR..."
gcloud auth configure-docker gcr.io --quiet

IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:${TAG}"
print_status "Building ${IMAGE_NAME}..."
docker build -f "${DOCKERFILE}" -t "${IMAGE_NAME}" "${REPO_ROOT}"
print_success "Image built"

print_status "Pushing ${IMAGE_NAME}..."
docker push "${IMAGE_NAME}"
print_success "Image pushed"

print_status "Deploying Cloud Run service ${SERVICE_NAME}..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}" \
    --region "${REGION}" \
    --platform managed \
    --allow-unauthenticated \
    --port 8080 \
    --memory 4Gi \
    --cpu 2 \
    --timeout 900 \
    --max-instances 10 \
    --concurrency 40 \
    --set-env-vars "OPENROUTER_API_KEY=${OPENROUTER_KEY},LOG_LEVEL=INFO" \
    --quiet

print_success "Deployed"

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format='value(status.url)')"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  Deployment complete                                                         ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Service URL:  ${SERVICE_URL}"
echo ""
echo "  Service index:"
echo "    curl -sS '${SERVICE_URL}/' | python3 -m json.tool"
echo ""
echo "  Health:"
echo "    curl -sS '${SERVICE_URL}/health' | python3 -m json.tool"
echo ""
echo "  Chat completions (needs non-empty tools[] for full kernel path):"
echo "    curl -sS -X POST '${SERVICE_URL}/v1/chat/completions' \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"tools\":[]}'"
echo ""
