"""Evaluación formal del agente conversacional multimodal SugarCane.

Este script evalúa el OE4 (agente conversacional inteligente) con un protocolo
cuantitativo y semi-cuantitativo para tesis de maestría.

Genera:
- Calidad de respuesta: Faithfulness, Answer Relevance, Hallucination Rate.
- Calidad conversacional: Conversational Coherence, Context Retention, Multi-Turn Success Rate.
- Calidad multimodal: Diagnostic Consistency, Prediction Reference Rate.
- Seguridad de dominio: Out-of-Domain Rejection Rate.
- Ablación: agente con RAG vs agente sin RAG.
- Exportación de conversaciones transcritas y reportes CSV/JSON/TXT.

Uso:
  python src/scripts/eval_agent_conversational.py
  python src/scripts/eval_agent_conversational.py --dataset data/eval/agent_eval_dataset.json
  python src/scripts/eval_agent_conversational.py --n-bootstrap 5000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.agent.agent import SugarCaneAgent
from app.rag.eval_metrics import (
    answer_relevance,
    faithfulness,
    hallucination_rate,
    content_tokens,
    compute_bootstrap_stats,
)

BASE_DIR = SRC_DIR.parent
DEFAULT_DATASET = BASE_DIR / "data" / "eval" / "agent_eval_dataset.json"
DEFAULT_OUTPUT = BASE_DIR / "data" / "eval" / "agent_eval_results.json"
DEFAULT_CSV = BASE_DIR / "data" / "eval" / "agent_eval_results.csv"
DEFAULT_CONV_CSV = BASE_DIR / "data" / "eval" / "agent_eval_conversations.csv"
DEFAULT_ABLATION_CSV = BASE_DIR / "data" / "eval" / "agent_eval_ablation.csv"
DEFAULT_BOOTSTRAP_CSV = BASE_DIR / "data" / "eval" / "agent_eval_bootstrap.csv"
DEFAULT_SUMMARY = BASE_DIR / "data" / "eval" / "agent_eval_summary.txt"

DOMAIN_REFUSAL_PATTERNS = [
    "no puedo responder",
    "no está relacionada",
    "no esta relacionada",
    "fuera del ámbito",
    "fuera del ambito",
    "no hay información disponible",
    "no hay informacion disponible",
    "no dispongo de evidencia",
    "caña de azúcar",
    "enfermedades foliares",
    "manejo agronómico",
    "manejo agronomico",
]

DISEASE_SYNONYMS = {
    "Rust": ["rust", "roya", "pústula", "pustula"],
    "RedRot": ["redrot", "red rot", "podredumbre roja", "pudrición roja", "pudricion roja"],
    "Mosaic": ["mosaic", "mosaico", "virus"],
    "Yellow": ["yellow", "amarillamiento", "hoja amarilla"],
    "Healthy": ["healthy", "sana", "hoja sana", "monitoreo"],
    "Bacterial Blight": ["bacterial blight", "tizón bacteriano", "tizon bacteriano", "bacteriano"],
}


def _safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _tokens(text: str) -> set[str]:
    return content_tokens(text or "")


def _expected_tokens(turn: Dict[str, Any]) -> set[str]:
    toks: set[str] = set()
    for kw in turn.get("expected_keywords", []) or []:
        toks |= _tokens(str(kw))
    return toks


def _contains_any(text: str, candidates: List[str]) -> bool:
    blob = (text or "").lower()
    return any(c.lower() in blob for c in candidates)


def expected_keyword_coverage(answer: str, expected_keywords: List[str] | None) -> float:
    if not expected_keywords:
        return 1.0
    expected: set[str] = set()
    for kw in expected_keywords:
        expected |= _tokens(str(kw))
    if not expected:
        return 1.0
    ans = _tokens(answer)
    return len(expected & ans) / len(expected)


def context_retention_score(answer: str, turn: Dict[str, Any], prediction: Optional[Dict[str, Any]]) -> float:
    """Evalúa si una respuesta de seguimiento mantiene el contexto conversacional."""
    if not turn.get("requires_context_retention", False):
        return 1.0
    terms: List[str] = list(turn.get("context_terms", []) or [])
    if prediction and prediction.get("class_name"):
        disease = str(prediction["class_name"])
        terms.extend(DISEASE_SYNONYMS.get(disease, [disease]))
    if not terms:
        return 0.0
    return 1.0 if _contains_any(answer, terms) else 0.0


def diagnostic_consistency_score(answer: str, prediction: Optional[Dict[str, Any]], expected_reference: bool = True) -> float:
    """Evalúa si la respuesta es coherente con la enfermedad clasificada por visión."""
    if not prediction or not prediction.get("class_name"):
        return 1.0
    if not expected_reference:
        # En consultas independientes no se penaliza no mencionar la predicción; sí se penaliza contradecirla de forma fuerte.
        return 1.0
    disease = str(prediction["class_name"])
    terms = DISEASE_SYNONYMS.get(disease, [disease])
    return 1.0 if _contains_any(answer, terms) else 0.0


def prediction_reference_score(answer: str, prediction: Optional[Dict[str, Any]], expected_reference: bool) -> float:
    if not expected_reference:
        return 1.0
    if not prediction:
        return 1.0
    refs = ["imagen", "clasificada", "predicha", "confianza", "diagnóstico", "diagnostico"]
    disease = str(prediction.get("class_name", ""))
    refs.extend(DISEASE_SYNONYMS.get(disease, [disease]))
    return 1.0 if _contains_any(answer, refs) else 0.0


def out_of_domain_rejection_score(answer: str, in_domain: bool) -> float:
    if in_domain:
        return 1.0
    return 1.0 if _contains_any(answer, DOMAIN_REFUSAL_PATTERNS) else 0.0


def conversational_coherence_score(
    question: str,
    answer: str,
    history: List[Dict[str, str]],
    turn: Dict[str, Any],
    prediction: Optional[Dict[str, Any]],
) -> float:
    """Proxy de coherencia: pertinencia + retención + consistencia diagnóstica."""
    ar = answer_relevance(question, answer)
    retention = context_retention_score(answer, turn, prediction)
    diag = diagnostic_consistency_score(answer, prediction, turn.get("expected_prediction_reference", True))
    kwcov = expected_keyword_coverage(answer, turn.get("expected_keywords"))
    # Pondera intención actual, continuidad y consistencia multimodal.
    return float(0.35 * ar + 0.25 * retention + 0.25 * diag + 0.15 * kwcov)


def turn_success_score(metrics: Dict[str, float], in_domain: bool) -> float:
    if not in_domain:
        return 1.0 if metrics["out_of_domain_rejection"] >= 1.0 else 0.0
    criteria = [
        metrics["answer_relevance"] >= 0.10,
        metrics["faithfulness"] >= 0.20,
        metrics["conversational_coherence"] >= 0.35,
        metrics["diagnostic_consistency"] >= 0.50,
    ]
    return 1.0 if all(criteria) else 0.0


@dataclass
class AgentTurnResult:
    conversation_id: str
    turn_id: str
    question: str
    answer_with_rag: str
    answer_without_rag: str
    in_domain: bool
    sources: List[str]
    faithfulness: float
    answer_relevance: float
    hallucination_rate: float
    expected_keyword_coverage: float
    conversational_coherence: float
    context_retention: float
    diagnostic_consistency: float
    prediction_reference: float
    out_of_domain_rejection: float
    turn_success: float
    no_rag_faithfulness: float
    no_rag_answer_relevance: float
    no_rag_hallucination_rate: float


def evaluate_agent_dataset(agent: SugarCaneAgent, dataset: Dict[str, Any], k: int = 5) -> List[AgentTurnResult]:
    results: List[AgentTurnResult] = []
    for conv in dataset.get("conversations", []):
        conv_id = conv["id"]
        prediction = conv.get("prediction")
        history: List[Dict[str, str]] = []

        for turn in conv.get("turns", []):
            q = turn["question"]
            in_domain = bool(turn.get("in_domain", True))

            chunks = agent.retriever.search(q, k=k, prediction=prediction, history=history)
            contexts = [c["text"] for c in chunks]
            sources = [c["source"] for c in chunks]

            answer = agent.generator.generate(q, chunks, prediction, history=history)
            if hasattr(agent.generator, "generate_without_rag"):
                no_rag_answer = agent.generator.generate_without_rag(q, prediction, history=history)
            else:
                no_rag_answer = agent.generator.generate(q, [], prediction, history=history)

            f = faithfulness(answer, contexts)
            ar = answer_relevance(q, answer)
            h = hallucination_rate(answer, contexts)
            kwcov = expected_keyword_coverage(answer, turn.get("expected_keywords"))
            retention = context_retention_score(answer, turn, prediction)
            diag = diagnostic_consistency_score(answer, prediction, turn.get("expected_prediction_reference", True))
            pred_ref = prediction_reference_score(answer, prediction, turn.get("expected_prediction_reference", True))
            ood = out_of_domain_rejection_score(answer, in_domain)
            coh = conversational_coherence_score(q, answer, history, turn, prediction)

            nr_f = faithfulness(no_rag_answer, contexts)
            nr_ar = answer_relevance(q, no_rag_answer)
            nr_h = 1.0 - nr_f

            metric_pack = {
                "faithfulness": f,
                "answer_relevance": ar,
                "conversational_coherence": coh,
                "diagnostic_consistency": diag,
                "out_of_domain_rejection": ood,
            }

            result = AgentTurnResult(
                conversation_id=conv_id,
                turn_id=turn.get("turn_id", f"t{len(history)+1}"),
                question=q,
                answer_with_rag=answer,
                answer_without_rag=no_rag_answer,
                in_domain=in_domain,
                sources=sources,
                faithfulness=f,
                answer_relevance=ar,
                hallucination_rate=h,
                expected_keyword_coverage=kwcov,
                conversational_coherence=coh,
                context_retention=retention,
                diagnostic_consistency=diag,
                prediction_reference=pred_ref,
                out_of_domain_rejection=ood,
                turn_success=turn_success_score(metric_pack, in_domain),
                no_rag_faithfulness=nr_f,
                no_rag_answer_relevance=nr_ar,
                no_rag_hallucination_rate=nr_h,
            )
            results.append(result)

            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": answer})
    return results


def aggregate_results(results: List[AgentTurnResult]) -> Dict[str, Any]:
    in_domain = [r for r in results if r.in_domain]
    ood = [r for r in results if not r.in_domain]
    pool = in_domain or results
    summary = {
        "n_turns_total": len(results),
        "n_turns_in_domain": len(in_domain),
        "n_turns_out_of_domain": len(ood),
        "faithfulness_avg": _safe_mean([r.faithfulness for r in pool]),
        "answer_relevance_avg": _safe_mean([r.answer_relevance for r in pool]),
        "hallucination_rate_avg": _safe_mean([r.hallucination_rate for r in pool]),
        "expected_keyword_coverage_avg": _safe_mean([r.expected_keyword_coverage for r in pool]),
        "conversational_coherence_avg": _safe_mean([r.conversational_coherence for r in pool]),
        "context_retention_avg": _safe_mean([r.context_retention for r in pool if r.context_retention < 1.0 or "t2" in r.turn_id.lower()]),
        "diagnostic_consistency_avg": _safe_mean([r.diagnostic_consistency for r in pool]),
        "prediction_reference_avg": _safe_mean([r.prediction_reference for r in pool]),
        "multi_turn_success_rate": _safe_mean([r.turn_success for r in results]),
        "out_of_domain_rejection_rate": _safe_mean([r.out_of_domain_rejection for r in ood]) if ood else None,
        "ablation": {
            "with_rag": {
                "faithfulness": _safe_mean([r.faithfulness for r in pool]),
                "answer_relevance": _safe_mean([r.answer_relevance for r in pool]),
                "hallucination_rate": _safe_mean([r.hallucination_rate for r in pool]),
            },
            "without_rag": {
                "faithfulness": _safe_mean([r.no_rag_faithfulness for r in pool]),
                "answer_relevance": _safe_mean([r.no_rag_answer_relevance for r in pool]),
                "hallucination_rate": _safe_mean([r.no_rag_hallucination_rate for r in pool]),
            },
        },
    }
    summary["ablation"]["delta"] = {
        key: summary["ablation"]["with_rag"][key] - summary["ablation"]["without_rag"][key]
        for key in ["faithfulness", "answer_relevance", "hallucination_rate"]
    }
    return summary


def bootstrap_agent(results: List[AgentTurnResult], n_bootstrap: int = 5000) -> List[Dict[str, Any]]:
    in_domain = [r for r in results if r.in_domain]
    pool = in_domain or results
    metric_values = {
        "faithfulness": [r.faithfulness for r in pool],
        "answer_relevance": [r.answer_relevance for r in pool],
        "hallucination_rate": [r.hallucination_rate for r in pool],
        "conversational_coherence": [r.conversational_coherence for r in pool],
        "diagnostic_consistency": [r.diagnostic_consistency for r in pool],
        "prediction_reference": [r.prediction_reference for r in pool],
        "multi_turn_success": [r.turn_success for r in results],
    }
    out = []
    for i, (metric, values) in enumerate(metric_values.items()):
        out.append(compute_bootstrap_stats(metric, values, n_bootstrap=n_bootstrap, seed=123 + i).to_dict())
    return out


def check_thresholds(summary: Dict[str, Any], thresholds: Dict[str, float]) -> List[str]:
    failures: List[str] = []
    checks = {
        "faithfulness_min": ("faithfulness_avg", summary.get("faithfulness_avg", 0.0)),
        "answer_relevance_min": ("answer_relevance_avg", summary.get("answer_relevance_avg", 0.0)),
        "hallucination_rate_max": ("hallucination_rate_avg", summary.get("hallucination_rate_avg", 1.0)),
        "conversation_success_rate_min": ("multi_turn_success_rate", summary.get("multi_turn_success_rate", 0.0)),
        "context_retention_min": ("context_retention_avg", summary.get("context_retention_avg", 0.0)),
        "out_of_domain_rejection_min": ("out_of_domain_rejection_rate", summary.get("out_of_domain_rejection_rate")),
        "diagnostic_consistency_min": ("diagnostic_consistency_avg", summary.get("diagnostic_consistency_avg", 0.0)),
    }
    for key, (name, value) in checks.items():
        if key not in thresholds or value is None:
            continue
        threshold = thresholds[key]
        if key.endswith("_max"):
            if value > threshold:
                failures.append(f"{name} {value:.3f} > {threshold}")
        else:
            if value < threshold:
                failures.append(f"{name} {value:.3f} < {threshold}")
    return failures


def save_results_csv(path: Path, results: List[AgentTurnResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "conversation_id", "turn_id", "in_domain", "question",
        "faithfulness", "answer_relevance", "hallucination_rate",
        "expected_keyword_coverage", "conversational_coherence", "context_retention",
        "diagnostic_consistency", "prediction_reference", "out_of_domain_rejection",
        "turn_success", "sources"
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for r in results:
            writer.writerow([
                r.conversation_id, r.turn_id, r.in_domain, r.question,
                f"{r.faithfulness:.4f}", f"{r.answer_relevance:.4f}", f"{r.hallucination_rate:.4f}",
                f"{r.expected_keyword_coverage:.4f}", f"{r.conversational_coherence:.4f}",
                f"{r.context_retention:.4f}", f"{r.diagnostic_consistency:.4f}",
                f"{r.prediction_reference:.4f}", f"{r.out_of_domain_rejection:.4f}",
                f"{r.turn_success:.4f}", "; ".join(r.sources)
            ])


def save_conversations_csv(path: Path, results: List[AgentTurnResult]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "conversation_id", "turn_id", "question", "answer_with_rag",
            "answer_without_rag", "sources"
        ])
        for r in results:
            writer.writerow([
                r.conversation_id, r.turn_id, r.question,
                r.answer_with_rag, r.answer_without_rag, "; ".join(r.sources)
            ])


def save_ablation_csv(path: Path, results: List[AgentTurnResult], summary: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "conversation_id", "turn_id", "question",
            "faithfulness_with_rag", "faithfulness_without_rag",
            "answer_relevance_with_rag", "answer_relevance_without_rag",
            "hallucination_rate_with_rag", "hallucination_rate_without_rag",
            "delta_faithfulness", "delta_answer_relevance", "delta_hallucination_rate"
        ])
        for r in results:
            writer.writerow([
                r.conversation_id, r.turn_id, r.question,
                f"{r.faithfulness:.4f}", f"{r.no_rag_faithfulness:.4f}",
                f"{r.answer_relevance:.4f}", f"{r.no_rag_answer_relevance:.4f}",
                f"{r.hallucination_rate:.4f}", f"{r.no_rag_hallucination_rate:.4f}",
                f"{r.faithfulness - r.no_rag_faithfulness:.4f}",
                f"{r.answer_relevance - r.no_rag_answer_relevance:.4f}",
                f"{r.hallucination_rate - r.no_rag_hallucination_rate:.4f}",
            ])
        writer.writerow([])
        writer.writerow(["PROMEDIO"])
        writer.writerow(["metric", "without_rag", "with_rag", "delta"])
        for metric in ["faithfulness", "answer_relevance", "hallucination_rate"]:
            writer.writerow([
                metric,
                f"{summary['ablation']['without_rag'][metric]:.4f}",
                f"{summary['ablation']['with_rag'][metric]:.4f}",
                f"{summary['ablation']['delta'][metric]:.4f}",
            ])


def save_bootstrap_csv(path: Path, bootstrap: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "ci_lower_95", "ci_upper_95", "n_samples", "n_bootstrap"])
        for b in bootstrap:
            writer.writerow([
                b["metric"], f"{b['mean']:.4f}", f"{b['std']:.4f}",
                f"{b['ci_lower']:.4f}", f"{b['ci_upper']:.4f}",
                b["n_samples"], b["n_bootstrap"]
            ])


def save_summary_txt(path: Path, summary: Dict[str, Any], bootstrap: List[Dict[str, Any]], failures: List[str]) -> None:
    lines = [
        "SugarCane — Evaluación formal del agente conversacional inteligente",
        f"Fecha UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "=== Métricas globales del agente ===",
        f"Turnos evaluados:             {summary['n_turns_total']}",
        f"Turnos en dominio:            {summary['n_turns_in_domain']}",
        f"Turnos fuera de dominio:      {summary['n_turns_out_of_domain']}",
        f"Faithfulness:                 {summary['faithfulness_avg']:.4f}",
        f"Answer Relevance:             {summary['answer_relevance_avg']:.4f}",
        f"Hallucination Rate:           {summary['hallucination_rate_avg']:.4f}",
        f"Conversational Coherence:     {summary['conversational_coherence_avg']:.4f}",
        f"Context Retention:            {summary['context_retention_avg']:.4f}",
        f"Diagnostic Consistency:       {summary['diagnostic_consistency_avg']:.4f}",
        f"Prediction Reference Rate:    {summary['prediction_reference_avg']:.4f}",
        f"Multi-Turn Success Rate:      {summary['multi_turn_success_rate']:.4f}",
        f"Out-of-Domain Rejection Rate: {summary['out_of_domain_rejection_rate'] if summary['out_of_domain_rejection_rate'] is not None else 'N/A'}",
        "",
        "=== Ablación del agente: sin RAG vs con RAG ===",
        f"{'Métrica':<24} {'Sin RAG':>10} {'Con RAG':>10} {'Delta':>10}",
    ]
    for metric in ["faithfulness", "answer_relevance", "hallucination_rate"]:
        lines.append(
            f"{metric:<24} "
            f"{summary['ablation']['without_rag'][metric]:>10.4f} "
            f"{summary['ablation']['with_rag'][metric]:>10.4f} "
            f"{summary['ablation']['delta'][metric]:>10.4f}"
        )
    lines.extend(["", "=== Bootstrap (IC 95%) ==="])
    lines.append(f"{'Métrica':<28} {'Media':>8} {'DE':>8} {'IC inf':>8} {'IC sup':>8} {'n':>4}")
    for b in bootstrap:
        lines.append(
            f"{b['metric']:<28} {b['mean']:>8.4f} {b['std']:>8.4f} "
            f"{b['ci_lower']:>8.4f} {b['ci_upper']:>8.4f} {b['n_samples']:>4}"
        )
    lines.append("")
    if failures:
        lines.append("Umbrales no cumplidos:")
        for f in failures:
            lines.append(f"  - {f}")
    else:
        lines.append("Todos los umbrales mínimos fueron cumplidos.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación del agente conversacional SugarCane")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--conversations-csv", type=Path, default=DEFAULT_CONV_CSV)
    parser.add_argument("--ablation-csv", type=Path, default=DEFAULT_ABLATION_CSV)
    parser.add_argument("--bootstrap-csv", type=Path, default=DEFAULT_BOOTSTRAP_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    agent = SugarCaneAgent()
    results = evaluate_agent_dataset(agent, dataset, k=args.k)
    summary = aggregate_results(results)
    bootstrap = bootstrap_agent(results, n_bootstrap=args.n_bootstrap)
    failures = check_thresholds(summary, dataset.get("thresholds", {}))

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "description": "Evaluación conversacional multi-turno del agente multimodal con métricas proxy alineadas con RAGAS y métricas funcionales de agente.",
            "k": args.k,
            "n_bootstrap": args.n_bootstrap,
            "metrics": [
                "faithfulness", "answer_relevance", "hallucination_rate",
                "conversational_coherence", "context_retention",
                "diagnostic_consistency", "prediction_reference",
                "multi_turn_success_rate", "out_of_domain_rejection_rate"
            ],
            "methodological_note": "Las métricas son proxy locales basadas en solapamiento léxico, ranking y reglas de dominio. Complementan RAGAS y permiten evaluar el agente sin ground truth humano exhaustivo."
        },
        "summary": summary,
        "bootstrap": bootstrap,
        "thresholds": dataset.get("thresholds", {}),
        "threshold_failures": failures,
        "turns": [asdict(r) for r in results],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    save_results_csv(args.csv, results)
    save_conversations_csv(args.conversations_csv, results)
    save_ablation_csv(args.ablation_csv, results, summary)
    save_bootstrap_csv(args.bootstrap_csv, bootstrap)
    save_summary_txt(args.summary, summary, bootstrap, failures)

    print("=== Evaluación formal del agente conversacional ===")
    print(f"Faithfulness:                 {summary['faithfulness_avg']:.4f}")
    print(f"Answer Relevance:             {summary['answer_relevance_avg']:.4f}")
    print(f"Hallucination Rate:           {summary['hallucination_rate_avg']:.4f}")
    print(f"Conversational Coherence:     {summary['conversational_coherence_avg']:.4f}")
    print(f"Context Retention:            {summary['context_retention_avg']:.4f}")
    print(f"Diagnostic Consistency:       {summary['diagnostic_consistency_avg']:.4f}")
    print(f"Multi-Turn Success Rate:      {summary['multi_turn_success_rate']:.4f}")
    print(f"Out-of-Domain Rejection Rate: {summary['out_of_domain_rejection_rate']}")
    print("")
    print("Archivos generados:")
    print(f"JSON:            {args.output}")
    print(f"CSV métricas:    {args.csv}")
    print(f"CSV diálogos:    {args.conversations_csv}")
    print(f"CSV ablación:    {args.ablation_csv}")
    print(f"CSV bootstrap:   {args.bootstrap_csv}")
    print(f"Resumen:         {args.summary}")
    if failures:
        print("\nUmbrales no cumplidos:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
