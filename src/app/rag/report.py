"""Generación de artefactos de investigación para evaluación RAG.

Métricas reportadas: Faithfulness, Answer Relevance, Hallucination Rate, nDCG@5.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from app.rag.compare import MethodRagSummary, QueryRagResult

METRIC_COLUMNS = [
    ("faithfulness_avg", "Faithfulness"),
    ("answer_relevance_avg", "Answer Relevance"),
    ("ndcg_at_5_avg", "nDCG@5"),
    ("hallucination_rate_avg", "Hallucination Rate"),
]


def _summary_dataframe(summaries: List[MethodRagSummary]) -> pd.DataFrame:
    return pd.DataFrame([s.to_dict() for s in summaries])


def plot_method_comparison(
    summaries: List[MethodRagSummary],
    output_path: Path,
    title: str = "Evaluación RAG — Faithfulness, Relevance, nDCG, Hallucination",
) -> Path:
    df = _summary_dataframe(summaries)
    plot_df = df.melt(
        id_vars=["method"],
        value_vars=[c[0] for c in METRIC_COLUMNS],
        var_name="metric",
        value_name="score",
    )
    label_map = {c[0]: c[1] for c in METRIC_COLUMNS}
    plot_df["metric"] = plot_df["metric"].map(label_map)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=plot_df, x="metric", y="score", hue="method", ax=ax, palette="Set2")
    ax.set_title(title)
    ax.set_xlabel("Métrica RAG")
    ax.set_ylabel("Puntuación")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Método")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_per_query_heatmap(
    results: List[QueryRagResult],
    output_path: Path,
    metric: str = "ndcg_at_5",
) -> Path:
    df = pd.DataFrame([r.to_dict() for r in results])
    pivot = df.pivot_table(index="query_id", columns="method", values=metric, aggfunc="mean")

    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.45)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
    ax.set_title(f"{metric} por consulta y método")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_markdown_table(summaries: List[MethodRagSummary]) -> str:
    df = _summary_dataframe(summaries)
    display = df.rename(
        columns={
            "method": "Método",
            "n_queries": "N",
            "faithfulness_avg": "Faithfulness",
            "answer_relevance_avg": "Answer Relevance",
            "ndcg_at_5_avg": "nDCG@5",
            "hallucination_rate_avg": "Hallucination Rate",
        }
    )
    for col in ["Faithfulness", "Answer Relevance", "nDCG@5", "Hallucination Rate"]:
        display[col] = display[col].map(lambda x: f"{x:.3f}")
    return display.to_markdown(index=False)


def generate_latex_table(summaries: List[MethodRagSummary]) -> str:
    df = _summary_dataframe(summaries)
    display = df.rename(
        columns={
            "method": "Método",
            "n_queries": "N",
            "faithfulness_avg": "Faithfulness",
            "answer_relevance_avg": "Ans. Rel.",
            "ndcg_at_5_avg": "nDCG@5",
            "hallucination_rate_avg": "Halluc.",
        }
    )
    for col in ["Faithfulness", "Ans. Rel.", "nDCG@5", "Halluc."]:
        display[col] = display[col].map(lambda x: f"{x:.3f}")

    latex = display.to_latex(index=False, escape=False, column_format="l" + "c" * (len(display.columns) - 1))
    return (
        "% Tabla generada automáticamente — métricas RAG oficiales\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Evaluación RAG con embeddings semánticos (Faithfulness, Answer Relevance, nDCG@5, Hallucination Rate)}\n"
        f"{latex}\n"
        "\\label{tab:rag-eval-metrics}\n"
        "\\end{table}\n"
    )


def write_research_artifacts(
    results: List[QueryRagResult],
    summaries: List[MethodRagSummary],
    output_dir: Path,
    best_method: str,
) -> dict:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    bar_path = plot_method_comparison(summaries, figures_dir / "rag_metrics_comparison.png")
    heatmap_path = plot_per_query_heatmap(results, figures_dir / "ndcg_per_query_heatmap.png")

    md_table = generate_markdown_table(summaries)
    latex_table = generate_latex_table(summaries)

    md_path = output_dir / "comparison_table.md"
    latex_path = output_dir / "comparison_table.tex"
    report_path = output_dir / "RESEARCH_REPORT.md"

    md_path.write_text(md_table + "\n", encoding="utf-8")
    latex_path.write_text(latex_table, encoding="utf-8")

    report = f"""# Informe de evaluación RAG — Métricas oficiales

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
- **Mejor método en comparación:** `{best_method}`

## Tabla comparativa

{md_table}

## Figuras

- `figures/rag_metrics_comparison.png`
- `figures/ndcg_per_query_heatmap.png`

## Reproducibilidad

```bash
python src/scripts/eval_rag_ragas.py --skip-ragas
python src/scripts/experiments.py --methods semantic
```
"""
    report_path.write_text(report, encoding="utf-8")

    return {
        "bar_chart": bar_path,
        "heatmap": heatmap_path,
        "markdown_table": md_path,
        "latex_table": latex_path,
        "report": report_path,
    }
