# SugarCane AI Agent

Proyecto final funcional para **Proyecto III de Innovación Tecnológica para IA**: clasificación de enfermedades foliares en caña de azúcar mediante visión computacional, más un **frontend en Streamlit** con **RAG/Agente IA** para explicación agronómica.

## 1. ¿Qué hace?

- Permite cargar una imagen de hoja de caña de azúcar.
- Carga automáticamente el mejor modelo entrenado según `artifacts/leaderboard_final.csv`.
- Clasifica la hoja en clases como `Healthy`, `Mosaic`, `RedRot`, `Rust` y `Yellow`.
- Muestra top de predicciones y confianza.
- Permite conversar con un Agente IA sobre el diagnóstico, síntomas, mitigación y recomendaciones.
- Usa RAG local con documentos `.md`, `.txt` o `.pdf` ubicados en `app/knowledge_base/`.
- Puede responder con OpenAI, Ollama o modo local extractivo si no hay LLM configurado.

## 2. Estructura

```text
SugarCane-AI-Agent/
├── app/
│   ├── classifier/          # Carga del mejor modelo y predicción
│   ├── rag/                 # Recuperador TF-IDF + generador de respuestas
│   ├── agent/               # Agente que une predicción + RAG
│   ├── ui/streamlit_app.py  # Frontend principal
│   └── knowledge_base/      # Documentos agronómicos para RAG
├── artifacts/               # leaderboards y metadatos del notebook
├── models/                  # best.pt y metadata del mejor modelo
├── scripts/prepare_artifacts.py
├── requirements.txt
└── README.md
```

## 3. Instalación en Visual Studio Code

```bash
cd SugarCane-AI-Agent
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## 4. Copiar el mejor modelo entrenado

Desde el notebook `proyecto3warpcaña_corregido.ipynb`, la carpeta esperada es:

```text
sugarcane_multimodel_2026/
├── leaderboard_final.csv
├── full_experiment_summary.json
└── final/.../best.pt
```

Ejecuta:

```bash
python scripts/prepare_artifacts.py --exp-root "RUTA_A/sugarcane_multimodel_2026"
```

El script copia:

- `leaderboard_final.csv` a `artifacts/`
- `full_experiment_summary.json` a `artifacts/`
- el `best.pt` del mejor modelo a `models/best.pt`
- `model_metadata.json` a `models/`

Si no tienes la ruta completa, copia manualmente:

```text
best.pt              -> models/best.pt
leaderboard_final.csv -> artifacts/leaderboard_final.csv
class_names.json      -> artifacts/class_names.json
```

## 5. Ejecutar Streamlit

```bash
streamlit run app/ui/streamlit_app.py
```

## 6. Configurar IA generativa opcional

Copia `.env.example` como `.env`.

### Opción OpenAI

```env
OPENAI_API_KEY=tu_api_key
OPENAI_MODEL=gpt-4o-mini
```

### Opción Ollama local

Instala Ollama, descarga un modelo y ejecuta:

```bash
ollama pull llama3.2:3b
ollama serve
```

Configura:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

Si no configuras OpenAI ni Ollama, el agente funciona en modo RAG extractivo local.

## 7. Personalizar la base de conocimiento

Agrega documentos técnicos sobre caña de azúcar en:

```text
app/knowledge_base/
```

Formatos soportados: `.md`, `.txt`, `.pdf`.

## 8. Flujo recomendado para la sustentación

1. Explicar que primero se entrenaron y compararon 9 modelos.
2. Mostrar que el sistema selecciona el mejor por `macro_f1` o `top1`.
3. Cargar una imagen de hoja.
4. Mostrar predicción y confianza.
5. Preguntar al agente: “Explícame la enfermedad detectada y cómo mitigarla”.
6. Destacar que la respuesta combina predicción visual + recuperación de conocimiento agronómico.

## 9. Nota importante

El sistema no reemplaza el diagnóstico de un agrónomo. La clasificación por IA es una herramienta de apoyo para priorizar inspección, monitoreo y toma de decisiones.

## Mejora RAG + Agente IA con memoria conversacional

Esta versión incorpora un agente conversacional más robusto:

- Usa el resultado actual de clasificación de la imagen como contexto del chat.
- Recupera conocimiento desde varios archivos Markdown/PDF/TXT ubicados en `app/knowledge_base/`.
- Expande la consulta con sinónimos técnicos por enfermedad: Rust, RedRot, Mosaic, Yellow y Healthy.
- Guarda histórico local de conversación en `data/conversations/`.
- Permite crear nueva conversación, guardar, borrar y cargar conversaciones anteriores desde la barra lateral.
- Usa OpenAI si existe `OPENAI_API_KEY`; si no, intenta Ollama; si no hay LLM, responde con modo extractivo local.

### Agregar más conocimiento al RAG

Para fortalecer la base de conocimiento, copie archivos `.md`, `.txt` o `.pdf` dentro de:

```bash
app/knowledge_base/
```

Luego reinicie Streamlit:

```bash
streamlit cache clear
streamlit run app/ui/streamlit_app.py
```

### Memoria de conversación

Los históricos quedan en:

```bash
data/conversations/
```

Cada conversación se guarda como un archivo `.json` con preguntas, respuestas, predicción de imagen y fuentes recuperadas.
