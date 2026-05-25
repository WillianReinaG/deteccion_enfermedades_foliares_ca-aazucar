# Despliegue en Google Cloud — SugarCane AI

Guía paso a paso para cuando tengas cuenta GCP con facturación activa.

## 1. Prerrequisitos

1. Crear proyecto en [Google Cloud Console](https://console.cloud.google.com).
2. Activar facturación.
3. Instalar [gcloud CLI](https://cloud.google.com/sdk/docs/install).
4. Autenticarse: `gcloud auth login` y `gcloud config set project TU_PROJECT_ID`.

### APIs a habilitar

```bash
gcloud services enable \
  run.googleapis.com \
  bigquery.googleapis.com \
  cloudscheduler.googleapis.com \
  cloudfunctions.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com
```

## 2. Modelo en Cloud Storage

Suba `models/best.pt` a un bucket (reemplace `TU_BUCKET`):

```bash
gsutil mb -l us-central1 gs://TU_BUCKET
gsutil cp models/best.pt gs://TU_BUCKET/models/best.pt
```

En Cloud Run puede descargar el modelo al arranque o montarlo vía script en `docker/entrypoint.sh`.

## 3. BigQuery

Cree dataset y tabla de predicciones:

```bash
bq mk --dataset --location=US TU_PROJECT_ID:sugarcane
```

Schema de `predictions`:

| Campo | Tipo |
|-------|------|
| predicted_at | TIMESTAMP |
| class_name | STRING |
| confidence | FLOAT |
| session_id | STRING |
| demo_mode | BOOL |
| model_id | STRING |
| source | STRING |

```bash
bq mk --table sugarcane.predictions \
  predicted_at:TIMESTAMP,class_name:STRING,confidence:FLOAT,session_id:STRING,demo_mode:BOOL,model_id:STRING,source:STRING
```

Scheduled query diaria (18:00 America/Bogota) — resumen últimas 24 h:

```sql
CREATE OR REPLACE TABLE `sugarcane.daily_summary` AS
SELECT
  DATE(predicted_at) AS day,
  class_name,
  COUNT(*) AS n,
  AVG(confidence) AS avg_conf
FROM `sugarcane.predictions`
WHERE predicted_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
GROUP BY day, class_name;
```

Configure en BigQuery → **Scheduled queries** → cron `0 18 * * *`, timezone `America/Bogota`.

## 4. Secret Manager

```bash
echo -n "SG.xxxx" | gcloud secrets create SENDGRID_API_KEY --data-file=-
echo -n "bebesowi@gmail.com" | gcloud secrets create ALERT_EMAIL --data-file=-
```

## 5. Cloud Run (app Streamlit)

Desde la raíz del repo:

```bash
bash infra/gcp/deploy_cloud_run.sh TU_PROJECT_ID us-central1
```

O manualmente:

```bash
gcloud run deploy sugarcane-app \
  --image docker.io/willianalbertorein/trabajofinalproyecto3:latest \
  --region us-central1 \
  --port 8501 \
  --allow-unauthenticated \
  --set-secrets SENDGRID_API_KEY=SENDGRID_API_KEY:latest,OPENAI_API_KEY=OPENAI_API_KEY:latest \
  --set-env-vars GCP_PROJECT_ID=TU_PROJECT_ID,BQ_DATASET=sugarcane,BQ_TABLE=predictions,ALERT_EMAIL=bebesowi@gmail.com
```

La URL pública será del tipo `https://sugarcane-app-xxxxx-uc.a.run.app`.

## 6. Informe diario 18:00 (Cloud Scheduler + Cloud Function)

Despliegue la función en `infra/gcp/function_daily_report/`:

```bash
bash infra/gcp/deploy_daily_report_function.sh TU_PROJECT_ID us-central1
```

Esto crea:

- Cloud Function `sugarcane-daily-report` (Python 3.11)
- Cloud Scheduler `sugarcane-daily-report-18h` — cron `0 18 * * *`, timezone `America/Bogota`

## 7. IAM mínimo

| Service account | Roles |
|-----------------|-------|
| Cloud Run | `roles/bigquery.dataEditor`, `roles/secretmanager.secretAccessor` |
| Cloud Function informe | `roles/bigquery.dataViewer`, `roles/secretmanager.secretAccessor` |

## 8. Costes estimados

- **Cloud Run**: free tier generoso para tráfico bajo; pago por uso de CPU/memoria.
- **BigQuery**: almacenamiento + consultas (pocas filas/día → coste bajo).
- **Cloud Scheduler**: 3 jobs gratis/mes.
- **SendGrid**: plan gratuito ~100 correos/día.

## 9. Verificación

1. Clasificar hoja enferma en la URL de Cloud Run → correo inmediato.
2. Revisar filas en `sugarcane.predictions`.
3. Ejecutar informe manual: `gcloud functions call sugarcane-daily-report --region us-central1`.
4. A las 18:00 (Bogotá) debe llegar el informe diario.

## Alternativa local (sin GCP)

```powershell
docker compose run --rm app daily-report
```

Requiere `.env` con `SMTP_USER`, `SMTP_APP_PASSWORD` y predicciones en `data/predictions/predictions.jsonl`.
