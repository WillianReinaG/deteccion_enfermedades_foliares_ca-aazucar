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
| tfidf    |            10 |      0.53  |      0.699 |      0.75  |     1 |    0.958 |
| bm25     |            10 |      0.552 |      0.69  |      0.757 |     1 |    0.956 |
| semantic |            10 |      0.498 |      0.609 |      0.674 |     1 |    0.956 |
| hybrid   |            10 |      0.591 |      0.665 |      0.73  |     1 |    0.982 |

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
