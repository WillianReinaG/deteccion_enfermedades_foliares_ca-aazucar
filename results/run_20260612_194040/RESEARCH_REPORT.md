# Informe de evaluación RAG — Métricas oficiales

## Métricas evaluadas

| Métrica | Descripción |
|---------|-------------|
| **Faithfulness** | Sustento de la respuesta en contextos recuperados |
| **Answer Relevance** | Alineación pregunta-respuesta (léxica + semántica) |
| **nDCG@5** | Ranking de recuperación con relevancia semántica (embeddings) |
| **Hallucination Rate** | 1 − Faithfulness |

## Configuración

- **Retrieval:** embeddings semánticos (`paraphrase-multilingual-mpnet-base-v2` + FAISS)
- **nDCG:** similitud coseno entre embeddings de consulta y fragmento
- **Mejor método en comparación:** `semantic`

## Tabla comparativa

| Método   |   N |   Faithfulness |   Answer Relevance |   nDCG@5 |   Hallucination Rate |
|:---------|----:|---------------:|-------------------:|---------:|---------------------:|
| semantic |   7 |          0.788 |               0.82 |     0.99 |                0.212 |

## Figuras

- `figures/rag_metrics_comparison.png`
- `figures/ndcg_per_query_heatmap.png`

## Reproducibilidad

```bash
python src/scripts/eval_rag_ragas.py --skip-ragas
python src/scripts/experiments.py --methods semantic
```
