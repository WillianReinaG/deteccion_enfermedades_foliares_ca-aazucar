# Informe de experimentación RAG — Recuperación documental

## Resumen ejecutivo

Se compararon tres métodos de recuperación sobre consultas agronómicas de caña de azúcar:
**TF-IDF**, **BM25** y **embeddings semánticos multilingües** (sentence-transformers + FAISS).

**Mejor método en experimento (score compuesto):** `bm25`

> El sistema de producción usa embeddings semánticos (`RAG_RETRIEVAL_METHOD=semantic`)
> por requisito del proyecto de grado y superior generalización semántica en consultas
> en lenguaje natural, aunque BM25/TF-IDF pueden superar en métricas léxicas con keywords.

## Tabla comparativa

| Método   |   N consultas |   Recall@1 |   Recall@3 |   Recall@5 |   MRR |   nDCG@5 |
|:---------|--------------:|-----------:|-----------:|-----------:|------:|---------:|
| tfidf    |            10 |      0.458 |      0.638 |      0.66  |     1 |    0.954 |
| bm25     |            10 |      0.523 |      0.701 |      0.729 |     1 |    0.953 |
| semantic |            10 |      0.428 |      0.569 |      0.608 |     1 |    0.922 |
| hybrid   |            10 |      0.515 |      0.587 |      0.666 |     1 |    0.947 |

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
