"""Evaluación RAG para tesis de maestría.

Métricas oficiales:
    - Faithfulness
    - Answer Relevance
    - Hallucination Rate
    - nDCG@5 (relevancia semántica por embeddings)

Retrieval: embeddings semánticos (sentence-transformers + FAISS).

Uso:
  python src/scripts/eval_rag_ragas.py
  python src/scripts/eval_rag_ragas.py --skip-ragas
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.agent.agent import SugarCaneAgent
from app.config.settings import OPENAI_API_KEY
from app.rag.eval_metrics import bootstrap_summary, check_thresholds, evaluate_agent_cases

BASE_DIR = SRC_DIR.parent
DEFAULT_DATASET = BASE_DIR / "data" / "eval" / "rag_eval_dataset.json"
DEFAULT_OUTPUT = BASE_DIR / "data" / "eval" / "rag_eval_results.json"
DEFAULT_CSV = BASE_DIR / "data" / "eval" / "rag_eval_results.csv"
DEFAULT_BOOTSTRAP_CSV = BASE_DIR / "data" / "eval" / "rag_eval_bootstrap.csv"
DEFAULT_SUMMARY = BASE_DIR / "data" / "eval" / "rag_eval_summary.txt"
DEFAULT_ABLATION_CSV = BASE_DIR / "data" / "eval" / "rag_eval_ablation.csv"
DEFAULT_RETRIEVAL_CSV = BASE_DIR / "data" / "eval" / "rag_eval_retrieved_chunks.csv"
DEFAULT_CORPUS_JSON = BASE_DIR / "data" / "eval" / "rag_corpus_report.json"
DEFAULT_CORPUS_TXT = BASE_DIR / "data" / "eval" / "rag_corpus_report.txt"


def _try_ragas_eval(agent: SugarCaneAgent, cases: list[dict], k: int = 5) -> dict | None:
    if not OPENAI_API_KEY:
        return None
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError:
        print("[ragas] Instale: pip install ragas datasets")
        return None

    rows = []
    for case in cases:
        if not case.get("in_domain", True):
            continue
        question = case["question"]
        prediction = case.get("prediction")
        chunks = agent.retriever.search(question, k=k, prediction=prediction)
        contexts = [c["text"] for c in chunks]
        answer = agent.generator.generate(question, chunks, prediction)
        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": case.get("ground_truth", ""),
            }
        )
    if not rows:
        return None

    try:
        ds = Dataset.from_list(rows)
        result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            return {
                "faithfulness": float(df["faithfulness"].mean()) if "faithfulness" in df else None,
                "answer_relevancy": float(df["answer_relevancy"].mean()) if "answer_relevancy" in df else None,
                "context_precision": float(df["context_precision"].mean()) if "context_precision" in df else None,
            }
        return {"raw": str(result)}
    except Exception as exc:
        print(f"[ragas] Error en evaluación oficial: {exc}")
        return {"error": str(exc)}


def _save_csv(path: Path, summary) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                "question",
                "in_domain",
                "retrieval_method",
                "faithfulness",
                "answer_relevance",
                "ndcg_at_5",
                "hallucination_rate",
                "sources",
            ]
        )
        for r in summary.results:
            writer.writerow(
                [
                    r.case_id,
                    r.question,
                    r.in_domain,
                    r.retrieval_method,
                    f"{r.faithfulness:.4f}",
                    f"{r.answer_relevance:.4f}",
                    f"{r.ndcg_at_5:.4f}",
                    f"{r.hallucination_rate:.4f}",
                    "; ".join(r.sources),
                ]
            )
        writer.writerow([])
        writer.writerow(["PROMEDIO (casos en dominio)"])
        writer.writerow(["faithfulness_avg", f"{summary.faithfulness_avg:.4f}"])
        writer.writerow(["answer_relevance_avg", f"{summary.answer_relevance_avg:.4f}"])
        writer.writerow(["ndcg_at_5_avg", f"{summary.ndcg_at_5_avg:.4f}"])
        writer.writerow(["hallucination_rate_avg", f"{summary.hallucination_rate_avg:.4f}"])


def _save_retrieved_chunks_csv(path: Path, summary) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                "question",
                "rank",
                "source",
                "chunk_id",
                "title",
                "diseases",
                "score_tfidf",
                "keyword_score",
                "disease_boost",
                "final_score",
                "mmr_score",
                "char_len",
                "text_preview",
            ]
        )
        for r in summary.results:
            for c in r.retrieved_chunks:
                writer.writerow(
                    [
                        r.case_id,
                        r.question,
                        c.get("rank"),
                        c.get("source"),
                        c.get("chunk_id"),
                        c.get("title"),
                        "; ".join(c.get("diseases", [])),
                        f"{float(c.get('score', 0)):.4f}",
                        f"{float(c.get('keyword_score', 0)):.4f}",
                        f"{float(c.get('disease_boost', 0)):.4f}",
                        f"{float(c.get('final_score', 0)):.4f}",
                        f"{float(c.get('mmr_score', 0)):.4f}",
                        c.get("char_len"),
                        str(c.get("text", ""))[:300].replace("\n", " "),
                    ]
                )


def _save_ablation_csv(path: Path, summary) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                "question",
                "in_domain",
                "faithfulness_with_rag",
                "faithfulness_without_rag",
                "answer_relevance_with_rag",
                "answer_relevance_without_rag",
                "ndcg_at_5_with_rag",
                "ndcg_at_5_without_rag",
                "hallucination_rate_with_rag",
                "hallucination_rate_without_rag",
                "delta_faithfulness",
                "delta_answer_relevance",
                "delta_ndcg_at_5",
                "delta_hallucination_rate",
                "answer_with_rag_preview",
                "answer_without_rag_preview",
            ]
        )
        for r in summary.results:
            if r.no_rag_answer is None:
                continue
            writer.writerow(
                [
                    r.case_id,
                    r.question,
                    r.in_domain,
                    f"{r.faithfulness:.4f}",
                    f"{float(r.no_rag_faithfulness):.4f}",
                    f"{r.answer_relevance:.4f}",
                    f"{float(r.no_rag_answer_relevance):.4f}",
                    f"{r.ndcg_at_5:.4f}",
                    f"{float(r.no_rag_ndcg_at_5 or 0.0):.4f}",
                    f"{r.hallucination_rate:.4f}",
                    f"{float(r.no_rag_hallucination_rate):.4f}",
                    f"{r.faithfulness - float(r.no_rag_faithfulness):.4f}",
                    f"{r.answer_relevance - float(r.no_rag_answer_relevance):.4f}",
                    f"{r.ndcg_at_5 - float(r.no_rag_ndcg_at_5 or 0.0):.4f}",
                    f"{r.hallucination_rate - float(r.no_rag_hallucination_rate):.4f}",
                    r.answer[:300].replace("\n", " "),
                    r.no_rag_answer[:300].replace("\n", " "),
                ]
            )
        if summary.ablation:
            writer.writerow([])
            writer.writerow(["PROMEDIO"])
            writer.writerow(["metric", "without_rag", "with_rag", "delta"])
            for metric in ["faithfulness", "answer_relevance", "ndcg_at_5", "hallucination_rate"]:
                writer.writerow(
                    [
                        metric,
                        f"{summary.ablation['without_rag'][metric]:.4f}",
                        f"{summary.ablation['with_rag'][metric]:.4f}",
                        f"{summary.ablation['delta'][metric]:.4f}",
                    ]
                )


def _save_bootstrap_csv(path: Path, summary) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "metric",
                "mean",
                "std",
                "ci_lower_95",
                "ci_upper_95",
                "n_samples",
                "n_bootstrap",
                "ci_level",
            ]
        )
        for stat in summary.bootstrap:
            writer.writerow(
                [
                    stat.metric,
                    f"{stat.mean:.4f}",
                    f"{stat.std:.4f}",
                    f"{stat.ci_lower:.4f}",
                    f"{stat.ci_upper:.4f}",
                    stat.n_samples,
                    stat.n_bootstrap,
                    stat.ci_level,
                ]
            )


def _save_corpus_report(agent: SugarCaneAgent, json_path: Path, txt_path: Path) -> Dict[str, Any]:
    report = agent.retriever.corpus_report() if hasattr(agent.retriever, "corpus_report") else {}
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "SugarCane — Reporte técnico de corpus y recuperación RAG",
        f"Fecha UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Directorio base documental: {report.get('knowledge_dir')}",
        f"Número de documentos: {report.get('document_count')}",
        f"Número de chunks: {report.get('chunk_count')}",
        "",
        "=== Chunking ===",
    ]
    chunking = report.get("chunking", {})
    lines.extend(
        [
            f"Estrategia: {chunking.get('strategy')}",
            f"Chunk size: {chunking.get('chunk_size_chars')} caracteres",
            f"Overlap: {chunking.get('chunk_overlap_chars')} caracteres",
            f"Stride: {chunking.get('chunk_stride_chars')} caracteres",
            "",
            "=== Vectorización / embeddings ===",
        ]
    )
    retrieval = report.get("retrieval", {})
    backend = retrieval.get("backend", {})
    lines.extend(
        [
            f"Método: {retrieval.get('method')}",
            f"Modelo embeddings: {backend.get('model', backend.get('semantic_backend', {}).get('model', 'N/A'))}",
            f"Índice: {backend.get('index', 'FAISS')}",
            f"Top-K por defecto: {retrieval.get('default_top_k')}",
            f"MMR habilitado: {retrieval.get('mmr', {}).get('enabled')}",
            f"Pesos score: {retrieval.get('scoring_weights')}",
            "",
            "=== Distribución de chunks por enfermedad ===",
        ]
    )
    for disease, count in sorted(report.get("disease_chunk_distribution", {}).items()):
        lines.append(f"{disease}: {count}")
    lines.append("")
    lines.append("=== Estadísticas de longitud de chunks ===")
    lines.append(str(report.get("chunk_length_stats_chars", {})))
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return report


def _save_summary_txt(path: Path, summary, ragas_metrics: dict | None, failures: list[str], corpus_report: Dict[str, Any]) -> None:
    lines = [
        "SugarCane — Evaluación RAG conversacional ampliada",
        f"Fecha UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "=== Configuración documental y recuperación ===",
        f"Documentos: {corpus_report.get('document_count')}",
        f"Chunks: {corpus_report.get('chunk_count')}",
        f"Embeddings / retrieval: {corpus_report.get('retrieval', {}).get('method')}",
        "",
        "=== Métricas RAG oficiales ===",
        f"faithfulness_avg:       {summary.faithfulness_avg:.4f}",
        f"answer_relevance_avg:   {summary.answer_relevance_avg:.4f}",
        f"ndcg_at_5_avg:          {summary.ndcg_at_5_avg:.4f}  (semántica por embeddings)",
        f"hallucination_rate_avg: {summary.hallucination_rate_avg:.4f}",
        "",
    ]
    if summary.ablation:
        lines.append("=== Ablación: sin RAG vs con RAG ===")
        lines.append(f"{'Métrica':<24} {'Sin RAG':>10} {'Con RAG':>10} {'Delta':>10}")
        for metric in ["faithfulness", "answer_relevance", "hallucination_rate"]:
            lines.append(
                f"{metric:<24} "
                f"{summary.ablation['without_rag'][metric]:>10.4f} "
                f"{summary.ablation['with_rag'][metric]:>10.4f} "
                f"{summary.ablation['delta'][metric]:>10.4f}"
            )
        lines.append("")
    if summary.bootstrap:
        lines.append("=== Bootstrap (IC 95%, casos en dominio) ===")
        lines.append(f"{'Métrica':<22} {'Media':>8} {'DE':>8} {'IC inf':>8} {'IC sup':>8} {'n':>4}")
        for stat in summary.bootstrap:
            lines.append(
                f"{stat.metric:<22} {stat.mean:>8.4f} {stat.std:>8.4f} "
                f"{stat.ci_lower:>8.4f} {stat.ci_upper:>8.4f} {stat.n_samples:>4}"
            )
        lines.append("")
    if ragas_metrics:
        lines.append("=== RAGAS (LLM juez — Es et al., 2024) ===")
        for key, value in ragas_metrics.items():
            lines.append(f"{key}: {value}")
        lines.append("")
    else:
        lines.append("RAGAS oficial: no ejecutado (requiere ragas + OPENAI_API_KEY).")
        lines.append("")
    if failures:
        lines.append("Umbrales no cumplidos:")
        lines.extend(f"  - {f}" for f in failures)
    else:
        lines.append("Todos los umbrales mínimos fueron cumplidos.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación RAG SugarCane ampliada")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--bootstrap-csv", type=Path, default=DEFAULT_BOOTSTRAP_CSV)
    parser.add_argument("--ablation-csv", type=Path, default=DEFAULT_ABLATION_CSV)
    parser.add_argument("--retrieval-csv", type=Path, default=DEFAULT_RETRIEVAL_CSV)
    parser.add_argument("--corpus-json", type=Path, default=DEFAULT_CORPUS_JSON)
    parser.add_argument("--corpus-txt", type=Path, default=DEFAULT_CORPUS_TXT)
    parser.add_argument("--n-bootstrap", type=int, default=5000, help="Réplicas bootstrap")
    parser.add_argument("--k", type=int, default=5, help="Número de chunks recuperados por consulta")
    parser.add_argument("--skip-ragas", action="store_true", help="Omitir RAGAS oficial")
    parser.add_argument("--no-ablation", action="store_true", help="No ejecutar comparación sin RAG vs con RAG")
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    agent = SugarCaneAgent()

    summary = evaluate_agent_cases(
        agent,
        dataset["cases"],
        k=args.k,
        include_ablation=not args.no_ablation,
    )
    summary.bootstrap = bootstrap_summary(summary, n_bootstrap=args.n_bootstrap)
    failures = check_thresholds(summary, dataset.get("thresholds", {}))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    corpus_report = _save_corpus_report(agent, args.corpus_json, args.corpus_txt)

    ragas_metrics = None if args.skip_ragas else _try_ragas_eval(agent, dataset["cases"], k=args.k)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": ["faithfulness", "answer_relevance", "hallucination_rate", "ndcg_at_5"],
        "retrieval_method": summary.retrieval_method,
        "proxy_metrics": summary.to_dict(),
        "thresholds": dataset.get("thresholds", {}),
        "threshold_failures": failures,
        "corpus_report": corpus_report,
        "ragas_metrics": ragas_metrics,
        "methodological_note": (
            "Métricas RAG oficiales del proyecto. nDCG@5 usa relevancia semántica "
            "(similitud coseno entre embeddings de consulta y fragmento). "
            "Retrieval: sentence-transformers multilingüe + FAISS."
        ),
    }

    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_csv(args.csv, summary)
    _save_retrieved_chunks_csv(args.retrieval_csv, summary)
    _save_ablation_csv(args.ablation_csv, summary)
    _save_bootstrap_csv(args.bootstrap_csv, summary)
    _save_summary_txt(args.summary, summary, ragas_metrics, failures, corpus_report)

    print("=== Evaluación RAG (métricas oficiales) ===")
    print(f"retrieval_method:       {summary.retrieval_method}")
    print(f"faithfulness_avg:       {summary.faithfulness_avg:.3f}")
    print(f"answer_relevance_avg:   {summary.answer_relevance_avg:.3f}")
    print(f"ndcg_at_5_avg:          {summary.ndcg_at_5_avg:.3f}")
    print(f"hallucination_rate_avg: {summary.hallucination_rate_avg:.3f}")
    if summary.ablation:
        print("\n=== Ablación sin RAG vs con RAG ===")
        for metric in ["faithfulness", "answer_relevance", "hallucination_rate"]:
            print(
                f"{metric}: sin_RAG={summary.ablation['without_rag'][metric]:.4f}, "
                f"con_RAG={summary.ablation['with_rag'][metric]:.4f}, "
                f"delta={summary.ablation['delta'][metric]:.4f}"
            )
    print("\nArchivos generados:")
    print(f"JSON:              {args.output}")
    print(f"CSV métricas:      {args.csv}")
    print(f"CSV chunks:        {args.retrieval_csv}")
    print(f"CSV ablación:      {args.ablation_csv}")
    print(f"CSV bootstrap:     {args.bootstrap_csv}")
    print(f"Corpus JSON:       {args.corpus_json}")
    print(f"Corpus TXT:        {args.corpus_txt}")
    print(f"Resumen:           {args.summary}")

    if ragas_metrics:
        print("\n=== RAGAS (LLM juez) ===")
        for key, value in ragas_metrics.items():
            print(f"{key}: {value}")
    else:
        print("\n[ragas] Omitido (sin OPENAI_API_KEY o paquete ragas no instalado).")

    if failures:
        print("\nUmbrales no cumplidos:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
