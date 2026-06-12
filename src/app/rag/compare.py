"""Comparación de métodos de recuperación con métricas RAG oficiales.

Evalúa Faithfulness, Answer Relevance, Hallucination Rate y nDCG@5 (semántica)
sobre el pipeline completo retrieve → generate por método de recuperación.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.rag.eval_metrics import RAG_METRICS, evaluate_with_retriever
from app.rag.generator import AnswerGenerator
from app.rag.retriever import ModularRetriever


@dataclass
class MethodRagSummary:
    method: str
    n_queries: int
    faithfulness_avg: float
    answer_relevance_avg: float
    ndcg_at_5_avg: float
    hallucination_rate_avg: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "n_queries": self.n_queries,
            "faithfulness_avg": self.faithfulness_avg,
            "answer_relevance_avg": self.answer_relevance_avg,
            "ndcg_at_5_avg": self.ndcg_at_5_avg,
            "hallucination_rate_avg": self.hallucination_rate_avg,
        }


@dataclass
class QueryRagResult:
    query_id: str
    question: str
    method: str
    faithfulness: float
    answer_relevance: float
    ndcg_at_5: float
    hallucination_rate: float
    top_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "method": self.method,
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "ndcg_at_5": self.ndcg_at_5,
            "hallucination_rate": self.hallucination_rate,
            "top_sources": self.top_sources,
        }


def compare_retrieval_methods(
    queries: List[Dict[str, Any]],
    methods: Optional[List[str]] = None,
    knowledge_dir: Optional[Path] = None,
    k: int = 5,
) -> tuple[List[QueryRagResult], List[MethodRagSummary]]:
    methods = methods or ["semantic"]
    in_domain = [q for q in queries if q.get("in_domain", True)]
    pool = in_domain or queries
    generator = AnswerGenerator()

    all_results: List[QueryRagResult] = []
    summaries: List[MethodRagSummary] = []

    for method in methods:
        retriever = (
            ModularRetriever(knowledge_dir=knowledge_dir, method=method)
            if knowledge_dir
            else ModularRetriever(method=method)
        )
        summary = evaluate_with_retriever(retriever, generator, pool, k=k, include_ablation=False)
        for case_result in summary.results:
            all_results.append(
                QueryRagResult(
                    query_id=case_result.case_id,
                    question=case_result.question,
                    method=method,
                    faithfulness=case_result.faithfulness,
                    answer_relevance=case_result.answer_relevance,
                    ndcg_at_5=case_result.ndcg_at_5,
                    hallucination_rate=case_result.hallucination_rate,
                    top_sources=case_result.sources[:3],
                )
            )
        summaries.append(
            MethodRagSummary(
                method=method,
                n_queries=len(summary.results),
                faithfulness_avg=summary.faithfulness_avg,
                answer_relevance_avg=summary.answer_relevance_avg,
                ndcg_at_5_avg=summary.ndcg_at_5_avg,
                hallucination_rate_avg=summary.hallucination_rate_avg,
            )
        )

    return all_results, summaries


def select_best_method(summaries: List[MethodRagSummary]) -> str:
    if not summaries:
        return "semantic"
    ranked = sorted(
        summaries,
        key=lambda s: s.faithfulness_avg + s.answer_relevance_avg + s.ndcg_at_5_avg - s.hallucination_rate_avg,
        reverse=True,
    )
    return ranked[0].method


def save_comparison_results(
    results: List[QueryRagResult],
    summaries: List[MethodRagSummary],
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / "rag_eval_comparison_detail.csv"
    summary_path = output_dir / "rag_eval_comparison_summary.csv"
    json_path = output_dir / "rag_eval_comparison.json"

    pd.DataFrame([r.to_dict() for r in results]).to_csv(detail_path, index=False, encoding="utf-8")
    pd.DataFrame([s.to_dict() for s in summaries]).to_csv(summary_path, index=False, encoding="utf-8")

    payload = {
        "metrics": list(RAG_METRICS),
        "ndcg_mode": "semantic_embedding_cosine",
        "detail": [r.to_dict() for r in results],
        "summary": [s.to_dict() for s in summaries],
        "best_method": select_best_method(summaries),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"detail": detail_path, "summary": summary_path, "json": json_path}
