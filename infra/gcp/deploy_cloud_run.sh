#!/usr/bin/env bash
# Despliega la app Streamlit en Cloud Run.
set -euo pipefail

PROJECT_ID="${1:?Usage: $0 PROJECT_ID [REGION]}"
REGION="${2:-us-central1}"
IMAGE="${DOCKER_IMAGE:-docker.io/willianalbertorein/trabajofinalproyecto3:latest}"
SERVICE_NAME="${SERVICE_NAME:-sugarcane-app}"

gcloud config set project "$PROJECT_ID"

gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --region "$REGION" \
  --port 8501 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-secrets "SENDGRID_API_KEY=SENDGRID_API_KEY:latest,OPENAI_API_KEY=OPENAI_API_KEY:latest" \
  --set-env-vars "GCP_PROJECT_ID=${PROJECT_ID},BQ_DATASET=sugarcane,BQ_TABLE=predictions,ALERT_EMAIL=bebesowi@gmail.com,ALERT_FROM=noreply@sugarcane.local"

echo "Desplegado: $(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format='value(status.url)')"
