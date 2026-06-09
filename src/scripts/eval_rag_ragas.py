"""Evaluación RAG con métricas proxy (CI) y RAGAS opcional (requiere OPENAI_API_KEY).

Uso:
  python src/scripts/eval_rag_ragas.py
  python src/scripts/eval_rag_ragas.py --output data/eval/rag_eval_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datetime import datetime, timezone

from app.agent.agent import SugarCaneAgent
from app.config.settings import OPENAI_API_KEY
from app.rag.eval_metrics import check_thresholds, evaluate_agent_cases

BASE_DIR = SRC_DIR.parent
DEFAULT_DATASET = BASE_DIR / "data" / "eval" / "rag_eval_dataset.json"
DEFAULT_OUTPUT = BASE_DIR / "data" / "eval" / "rag_eval_results.json"
DEFAULT_CSV = BASE_DIR / "data" / "eval" / "rag_eval_results.csv"
DEFAULT_SUMMARY = BASE_DIR / "data" / "eval" / "rag_eval_summary.txt"


def _try_ragas_eval(agent: SugarCaneAgent, cases: list[dict]) -> dict | None:
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
        chunks = agent.retriever.search(question, k=5, prediction=prediction)
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
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
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
    import csv

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                "question",
                "in_domain",
                "faithfulness",
                "answer_relevance",
                "context_precision",
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
                    f"{r.faithfulness:.4f}",
                    f"{r.answer_relevance:.4f}",
                    f"{r.context_precision:.4f}",
                    f"{r.hallucination_rate:.4f}",
                    "; ".join(r.sources),
                ]
            )
        writer.writerow([])
        writer.writerow(["PROMEDIO (casos en dominio)", "", ""])
        writer.writerow(["faithfulness_avg", f"{summary.faithfulness_avg:.4f}"])
        writer.writerow(["answer_relevance_avg", f"{summary.answer_relevance_avg:.4f}"])
        writer.writerow(["context_precision_avg", f"{summary.context_precision_avg:.4f}"])
        writer.writerow(["hallucination_rate_avg", f"{summary.hallucination_rate_avg:.4f}"])


def _save_summary_txt(path: Path, summary, ragas_metrics: dict | None, failures: list[str]) -> None:
    lines = [
        "SugarCane — Evaluación RAG conversacional",
        f"Fecha UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "=== Métricas proxy (CI / léxicas) ===",
        f"faithfulness_avg:       {summary.faithfulness_avg:.4f}",
        f"answer_relevance_avg:   {summary.answer_relevance_avg:.4f}",
        f"context_precision_avg:  {summary.context_precision_avg:.4f}",
        f"hallucination_rate_avg: {summary.hallucination_rate_avg:.4f}",
        "",
    ]
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
    parser = argparse.ArgumentParser(description="Evaluación RAG SugarCane")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--skip-ragas", action="store_true", help="Omitir RAGAS oficial (solo métricas proxy)")
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    agent = SugarCaneAgent()
    summary = evaluate_agent_cases(agent, dataset["cases"])
    failures = check_thresholds(summary, dataset["thresholds"])

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "proxy_metrics": summary.to_dict(),
        "thresholds": dataset["thresholds"],
        "threshold_failures": failures,
        "ragas_metrics": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_csv(args.csv, summary)
    _save_summary_txt(args.summary, summary, None, failures)

    ragas_metrics = None if args.skip_ragas else _try_ragas_eval(agent, dataset["cases"])
    payload["ragas_metrics"] = ragas_metrics
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_summary_txt(args.summary, summary, ragas_metrics, failures)

    print("=== Evaluación RAG (métricas proxy — alineadas con RAGAS) ===")
    print(f"faithfulness_avg:       {summary.faithfulness_avg:.3f}")
    print(f"answer_relevance_avg:   {summary.answer_relevance_avg:.3f}")
    print(f"context_precision_avg:  {summary.context_precision_avg:.3f}")
    print(f"hallucination_rate_avg: {summary.hallucination_rate_avg:.3f}")
    print(f"JSON:    {args.output}")
    print(f"CSV:     {args.csv}")
    print(f"Resumen: {args.summary}")

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
