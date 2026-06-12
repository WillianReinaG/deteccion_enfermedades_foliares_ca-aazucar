"""Experimentos reproducibles de evaluación RAG con métricas oficiales.

Métricas: Faithfulness, Answer Relevance, Hallucination Rate, nDCG@5 (semántica).
Retrieval por defecto: embeddings semánticos.

Uso:
    python src/scripts/experiments.py
    python src/scripts/experiments.py --methods semantic
    python src/scripts/experiments.py --methods semantic bm25 --output results/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.config.settings import KNOWLEDGE_DIR, RAG_RETRIEVAL_METHOD, RESULTS_DIR
from app.rag.compare import compare_retrieval_methods, save_comparison_results, select_best_method
from app.rag.ingestion import load_corpus
from app.rag.report import write_research_artifacts
from app.rag.retriever import ModularRetriever

BASE_DIR = SRC_DIR.parent
DEFAULT_BENCHMARK = BASE_DIR / "data" / "eval" / "rag_eval_dataset.json"


def load_queries(benchmark_path: Path) -> list[dict]:
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    return payload.get("queries") or payload.get("cases", [])


def write_corpus_snapshot(output_dir: Path) -> None:
    chunks, source_files = load_corpus(KNOWLEDGE_DIR)
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "knowledge_dir": str(KNOWLEDGE_DIR),
        "document_count": len(source_files),
        "chunk_count": len(chunks),
        "retrieval_method": RAG_RETRIEVAL_METHOD,
        "source_files": source_files,
    }
    (output_dir / "corpus_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_production_index(method: str, output_dir: Path) -> None:
    retriever = ModularRetriever(method=method)
    report = retriever.corpus_report()
    report["selected_method"] = method
    report["production_ready"] = True
    report["rag_metrics"] = ["faithfulness", "answer_relevance", "hallucination_rate", "ndcg_at_5"]
    (output_dir / "production_index_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Índice de producción ({method}): {report['chunk_count']} chunks.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación RAG — Faithfulness, Relevance, nDCG, Hallucination")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument("--methods", nargs="+", default=["semantic"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--skip-production", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(args.benchmark)
    print(f"Consultas: {len(queries)} | Métodos: {args.methods} | Métricas RAG oficiales")
    print(f"Salida: {output_dir}")

    write_corpus_snapshot(output_dir)
    results, summaries = compare_retrieval_methods(
        queries=queries, methods=args.methods, knowledge_dir=KNOWLEDGE_DIR, k=args.k
    )

    paths = save_comparison_results(results, summaries, output_dir)
    best_method = select_best_method(summaries)
    artifacts = write_research_artifacts(results, summaries, output_dir, best_method)

    production_method = RAG_RETRIEVAL_METHOD
    if not args.skip_production:
        run_production_index(production_method, output_dir)

    manifest = {
        "timestamp": timestamp,
        "metrics": ["faithfulness", "answer_relevance", "hallucination_rate", "ndcg_at_5"],
        "ndcg_mode": "semantic_embedding_cosine",
        "production_method": production_method,
        "best_method": best_method,
        "summary": [s.to_dict() for s in summaries],
        "outputs": {k: str(v) for k, v in {**paths, **artifacts}.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- Resumen RAG por método ---")
    for s in summaries:
        print(
            f"  {s.method:10s} | Faith={s.faithfulness_avg:.3f} | "
            f"AnsRel={s.answer_relevance_avg:.3f} | nDCG@5={s.ndcg_at_5_avg:.3f} | "
            f"Halluc={s.hallucination_rate_avg:.3f}"
        )
    print(f"\nProducción: {production_method} | Mejor en comparación: {best_method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
