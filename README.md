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
├── models/                 # Pesos (.pt) — no van a Git
├── artifacts/              # Leaderboards y metadatos
├── .github/workflows/      # CI, CD y MLOps
├── requirements.txt
├── requirements-dev.txt
└── docker-compose.yml
```

## Desarrollo local

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
ruff check src tests
streamlit run src/app/ui/streamlit_app.py
```

### Copiar modelo entrenado

```powershell
python src/scripts/prepare_artifacts.py --exp-root "RUTA/sugarcane_multimodel_2026"
```

## Docker

```powershell
# Tests en contenedor
docker compose run --rm test

# App Streamlit (primera vez: build puede tardar varios minutos)
docker compose build app
docker compose up app
```

Abrir **http://localhost:8501**. Si en Windows ves `entrypoint.sh: no such file or directory`, reconstruir sin caché: `docker compose build --no-cache app`.

El `best.pt` debe estar en `models/` del host (se monta como volumen).

Entrypoint (`docker/entrypoint.sh`): `test` | `lint` | `streamlit` | `train` | `validate` | `monitor`

## CI/CD (GitHub Actions)

| Workflow | Disparador | Qué hace |
|----------|------------|----------|
| `ci.yml` | push / PR | Ruff, pytest, build Docker, tests en contenedor |
| `cd-publish.yml` | push a `main` | Build, smoke test, publica en **GHCR** |
| `mlops.yml` | manual / cron semanal | train → validate → monitor → deploy |

### Imagen publicada (GHCR)

```text
ghcr.io/willianreinag/deteccion_enfermedades_foliares_ca-aazucar:latest
```

```powershell
docker pull ghcr.io/willianreinag/deteccion_enfermedades_foliares_ca-aazucar:latest
docker run --rm -p 8501:8501 `
  -v "${PWD}/models:/app/models:ro" `
  -v "${PWD}/artifacts:/app/artifacts:ro" `
  ghcr.io/willianreinag/deteccion_enfermedades_foliares_ca-aazucar:latest streamlit
```

Tras el primer push a `main`, en GitHub: **Packages** → hacer el paquete **público** si quieres `docker pull` sin login.

## MLOps (iteración)

1. **Entrenamiento**: notebook local → `EXP_ROOT` en workflow o `python src/scripts/mlops/train.py`
2. **Validación**: comprueba artefactos y carga del predictor → `data/monitoring/validation_report.json`
3. **Monitoreo**: heurística de retraining → `data/monitoring/monitor_report.json`
4. **Despliegue**: `docker compose up app` o imagen GHCR con volúmenes `models/` y `artifacts/`

Disparo manual: **Actions** → **MLOps Pipeline** → elegir etapa (`train`, `validate`, `monitor`, `deploy`, `all`).

## Variables de entorno

Copie `.env.example` a `.env` (OpenAI / Ollama opcionales).

## Nota

La clasificación por IA es apoyo a la decisión; no sustituye el criterio de un agrónomo.
