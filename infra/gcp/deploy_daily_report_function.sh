#!/usr/bin/env bash
# Despliega Cloud Function + Cloud Scheduler para informe diario 18:00 America/Bogota.
set -euo pipefail

PROJECT_ID="${1:?Usage: $0 PROJECT_ID [REGION]}"
REGION="${2:-us-central1}"
FUNCTION_NAME="${FUNCTION_NAME:-sugarcane-daily-report}"
SCHEDULER_NAME="${SCHEDULER_NAME:-sugarcane-daily-report-18h}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

gcloud config set project "$PROJECT_ID"

gcloud functions deploy "$FUNCTION_NAME" \
  --gen2 \
  --runtime python311 \
  --region "$REGION" \
  --source "$ROOT/infra/gcp/function_daily_report" \
  --entry-point daily_report_http \
  --trigger-http \
  --allow-unauthenticated \
  --set-secrets "SENDGRID_API_KEY=SENDGRID_API_KEY:latest" \
  --set-env-vars "GCP_PROJECT_ID=${PROJECT_ID},BQ_DATASET=sugarcane,BQ_TABLE=predictions,ALERT_EMAIL=bebesowi@gmail.com,ALERT_FROM=noreply@sugarcane.local"

FUNCTION_URL="$(gcloud functions describe "$FUNCTION_NAME" --region "$REGION" --gen2 --format='value(serviceConfig.uri)')"

gcloud scheduler jobs delete "$SCHEDULER_NAME" --location="$REGION" --quiet 2>/dev/null || true

gcloud scheduler jobs create http "$SCHEDULER_NAME" \
  --location="$REGION" \
  --schedule="0 18 * * *" \
  --time-zone="America/Bogota" \
  --uri="$FUNCTION_URL" \
  --http-method=POST \
  --oidc-service-account-email="$(gcloud iam service-accounts list --filter='displayName:Compute Engine default' --format='value(email)' | head -1)"

echo "Scheduler $SCHEDULER_NAME → $FUNCTION_URL (18:00 America/Bogota)"
