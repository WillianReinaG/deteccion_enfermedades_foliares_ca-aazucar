# Informe de experimentación RAG — Recuperación documental

## Resumen ejecutivo

Se compararon tres métodos de recuperación sobre consultas agronómicas de caña de azúcar:
**TF-IDF**, **BM25** y **embeddings semánticos multilingües** (sentence-transformers + FAISS).

**Método recomendado para producción:** `bm25`

## Tabla comparativa

| Método   |   N consultas |   Recall@1 |   Recall@3 |   Recall@5 |   MRR |   nDCG@5 |
|:---------|--------------:|-----------:|-----------:|-----------:|------:|---------:|
| tfidf    |            10 |      0.517 |      0.672 |      0.741 |     1 |    0.96  |
| bm25     |            10 |      0.55  |      0.684 |      0.722 |     1 |    0.981 |
| semantic |            10 |      0.472 |      0.582 |      0.63  |     1 |    0.969 |

## Figuras

- Comparación por métrica: `figures/retrieval_methods_comparison.png`
- Heatmap nDCG@5 por consulta: `figures/ndcg_per_query_heatmap.png`

## Métricas evaluadas

| Métrica | Descripción |
|---------|-------------|
| Recall@k | Cobertura de términos relevantes en top-k fragmentos |
| MRR | Recíproco del rango del primer fragmento relevante |
| nDCG@5 | Ganancia acumulada descontada normalizada |

## Reproducibilidad

Ejecutar: `python -m src.scripts.experiments`
