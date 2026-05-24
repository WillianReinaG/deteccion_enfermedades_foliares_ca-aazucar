# SugarCane AI Agent

Detección de enfermedades foliares en caña de azúcar (visión computacional + RAG + agente conversacional).

Repositorio: [deteccion_enfermedades_foliares_ca-aazucar](https://github.com/WillianReinaG/deteccion_enfermedades_foliares_ca-aazucar.git)

## Estructura del proyecto

```text
├── src/                    # Código fuente
│   ├── app/                # Clasificador, RAG, agente, UI Streamlit
│   ├── scripts/            # Utilidades y MLOps
│   └── main.py             # CLI del agente
├── data/                   # Conversaciones y reportes de monitoreo
├── tests/                  # Pruebas unitarias
├── docker/                 # Dockerfile y entrypoint
├── models/                 # Pesos best.pt (local, no van a Git)
├── artifacts/              # Leaderboards y metadatos
├── .github/workflows/      # CI, CD y MLOps
├── requirements.txt
├── requirements-dev.txt
└── docker-compose.yml
```

## Puesta en marcha completa

### Pre requisitos

- Python 3.11+
- Docker Desktop (recomendado)
- Git
- `models/best.pt` en la carpeta del proyecto (copia local del modelo entrenado)

### 1. Configurar IA generativa (OpenAI)

```powershell
cd "C:\Users\bebes\Documents\MIAA\3.SEMESTRE\3.proyecto_tres\el proyecto\DATA2\SugarCane ProyectoFinal"
copy .env.example .env
```

Edite `.env` y agregue su clave:

```env
OPENAI_API_KEY=sk-su_clave_aqui
OPENAI_MODEL=gpt-4o-mini
```

> `.env` no se sube a GitHub. Sin `OPENAI_API_KEY`, el chat usa modo RAG extractivo local.

### 2. Arranque con Docker (recomendado)

```powershell
docker compose build app
docker compose up app
```

Abrir **http://localhost:8501**

El sidebar debe mostrar **Motor de respuesta: OpenAI (gpt-4o-mini)** si la clave está configurada.

### 3. Arranque local (alternativa)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt
streamlit run src/app/ui/streamlit_app.py
```

### 4. Probar la aplicación

1. Subir imagen de hoja → **Clasificar hoja** (requiere `models/best.pt`).
2. Pregunta válida: *"Explícame la enfermedad detectada, síntomas y manejo en campo"*.
3. Pregunta fuera de dominio: *"¿Quién ganó el mundial?"* → debe rechazar educadamente.
4. Verificar sidebar: modelo cargado + motor OpenAI.

### 5. Verificar pruebas y CI/CD

```powershell
# Tests locales
python -m pytest -q

# Tests en contenedor (igual que GitHub Actions)
docker compose run --rm test
```

GitHub Actions: https://github.com/WillianReinaG/deteccion_enfermedades_foliares_ca-aazucar/actions

| Workflow | Disparador | Qué hace |
|----------|------------|----------|
| `ci.yml` | push / PR | Ruff, pytest, build Docker, tests en contenedor |
| `cd-publish.yml` | push a `main` | Build, smoke test, publica en **GHCR** |
| `cd-dockerhub.yml` | push a `main` | Build, smoke test, publica en **Docker Hub** |
| `mlops.yml` | manual / cron | train → validate → monitor → deploy |

## Modelo entrenado

Si ya tiene `models/best.pt` en su PC, no necesita scripts adicionales.

Opcional (solo si copia artefactos desde una carpeta de entrenamiento externa):

```powershell
python src/scripts/prepare_artifacts.py --exp-root "RUTA_A_carpeta_de_resultados"
```

## Docker — referencia rápida

```powershell
docker compose run --rm test    # tests
docker compose up app             # Streamlit en :8501
docker compose down               # detener
```

Si en Windows aparece `entrypoint.sh: no such file or directory`:

```powershell
docker compose build --no-cache app
```

## Imagen publicada (GHCR)

```text
ghcr.io/willianreinag/deteccion_enfermedades_foliares_ca-aazucar:latest
```

## Imagen publicada (Docker Hub)

Requiere secrets `DOCKERHUB_USERNAME` y `DOCKERHUB_TOKEN` en GitHub Actions.

```text
willianalbertorein/trabajofinalproyecto3:latest
```

Ejecutar desde cualquier PC (con `models/best.pt` local o descargado de GCS):

```powershell
docker login
docker pull willianalbertorein/trabajofinalproyecto3:latest
docker run -p 8501:8501 `
  -v "${PWD}/models:/app/models:ro" `
  -v "${PWD}/artifacts:/app/artifacts:ro" `
  -v "${PWD}/data/predictions:/app/data/predictions" `
  --env-file .env `
  willianalbertorein/trabajofinalproyecto3:latest streamlit
```

## Alertas e informes por correo

Configure en `.env`:

```env
SENDGRID_API_KEY=SG.xxxx
ALERT_EMAIL=bebesowi@gmail.com
ALERT_CONFIDENCE_MIN=0.5
```

- **Alerta inmediata**: al clasificar una hoja con enfermedad (clase ≠ Healthy) y confianza ≥ umbral.
- **Registro local**: `data/predictions/predictions.jsonl` (BigQuery cuando configure `GCP_PROJECT_ID`).
- **Informe diario manual**:

```powershell
docker compose run --rm app daily-report
```

Guía cloud completa: [docs/GCP_SETUP.md](docs/GCP_SETUP.md) (Cloud Run, BigQuery, Scheduler 18:00 America/Bogota).

## Agente agronómico (IA generativa)

El prompt en `src/app/rag/generator.py` instruye al agente a:

- Responder **solo** con evidencia RAG + histórico + predicción de imagen.
- Usar tono profesional de agrónomo especialista en caña de azúcar.
- Rechazar preguntas fuera de dominio o sin evidencia documental.
- No presentar la clasificación visual como diagnóstico definitivo.

Prioridad de motores: **OpenAI** → Ollama (si no hay OpenAI) → RAG extractivo local.

## MLOps (iteración)

Disparo manual: **Actions** → **MLOps Pipeline** → etapa `validate`, `monitor` o `all`.

## Nota

La clasificación por IA es apoyo a la decisión; no sustituye el criterio de un agrónomo.
