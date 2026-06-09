"""Métricas de evaluación RAG (proxy alineadas con RAGAS, Es et al., 2024).

En CI se usan heurísticas locales (sin LLM juez). Para evaluación completa con RAGAS
y OpenAI, ejecutar: python src/scripts/eval_rag_ragas.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

SPANISH_STOP = frozenset(
    "a al algo algunas algunos ante antes como con contra cual cuales cuando de del desde donde dos el ella ellas ellos en entre era eran es esa esas ese eso esos esta estaba estaban estamos estan estas este esto estos fue fueron ha habia habian han hasta hay la las le mas me mi mis mucho muy nos o para pero por porque que quien se sin sobre su sus tambien te tiene todo tu tus un una uno usted ustedes y ya".split()
)


def content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-záéíóúñü0-9]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in SPANISH_STOP}


def faithfulness(answer: str, contexts: List[str]) -> float:
    """Proporción de tokens de la respuesta sustentados por el contexto recuperado."""
    ctx_tokens: set[str] = set()
    for ctx in contexts:
        ctx_tokens |= content_tokens(ctx)
    ans_tokens = content_tokens(answer)
    if not ans_tokens:
        return 1.0
    return len(ans_tokens & ctx_tokens) / len(ans_tokens)


def answer_relevance(question: str, answer: str) -> float:
    """Cobertura de términos de la pregunta presentes en la respuesta."""
    q_tokens = content_tokens(question)
    a_tokens = content_tokens(answer)
    if not q_tokens or not a_tokens:
        return 0.0
    return len(q_tokens & a_tokens) / len(q_tokens)


def context_precision(
    question: str,
    contexts: List[str],
    expected_keywords: Optional[List[str]] = None,
    k: int = 5,
) -> float:
    """Fracción de contextos top-k relevantes para la consulta."""
    if not contexts:
        return 0.0
    q_tokens = content_tokens(question)
    if expected_keywords:
        q_tokens |= {kw.lower() for kw in expected_keywords if len(kw) > 2}
    if not q_tokens:
        return 0.0
    hits = 0
    for ctx in contexts[:k]:
        if q_tokens & content_tokens(ctx):
            hits += 1
    return hits / min(k, len(contexts))


def hallucination_rate(answer: str, contexts: List[str]) -> float:
    """Tasa de alucinación proxy: 1 - faithfulness."""
    return 1.0 - faithfulness(answer, contexts)


@dataclass
class RagEvalCaseResult:
    case_id: str
    question: str
    answer: str
    contexts: List[str]
    faithfulness: float
    answer_relevance: float
    context_precision: float
    hallucination_rate: float
    in_domain: bool = True
    sources: List[str] = field(default_factory=list)


@dataclass
class BootstrapMetricStats:
    metric: str
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    n_bootstrap: int
    ci_level: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "mean": self.mean,
            "std": self.std,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_samples": self.n_samples,
            "n_bootstrap": self.n_bootstrap,
            "ci_level": self.ci_level,
        }


@dataclass
class RagEvalSummary:
    results: List[RagEvalCaseResult]
    faithfulness_avg: float
    answer_relevance_avg: float
    context_precision_avg: float
    hallucination_rate_avg: float
    bootstrap: List[BootstrapMetricStats] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "faithfulness_avg": self.faithfulness_avg,
            "answer_relevance_avg": self.answer_relevance_avg,
            "context_precision_avg": self.context_precision_avg,
            "hallucination_rate_avg": self.hallucination_rate_avg,
            "cases": [
                {
                    "id": r.case_id,
                    "faithfulness": r.faithfulness,
                    "answer_relevance": r.answer_relevance,
                    "context_precision": r.context_precision,
                    "hallucination_rate": r.hallucination_rate,
                    "in_domain": r.in_domain,
                }
                for r in self.results
            ],
        }
        if self.bootstrap:
            payload["bootstrap"] = [b.to_dict() for b in self.bootstrap]
        return payload


def compute_bootstrap_stats(
    metric: str,
    values: List[float],
    n_bootstrap: int = 5000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> BootstrapMetricStats:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return BootstrapMetricStats(metric, 0.0, 0.0, 0.0, 0.0, 0, n_bootstrap, ci_level)
    if arr.size == 1:
        v = float(arr[0])
        return BootstrapMetricStats(metric, v, 0.0, v, v, 1, n_bootstrap, ci_level)

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boot_means[i] = sample.mean()

    alpha = (1.0 - ci_level) / 2.0
    ci_lower, ci_upper = np.percentile(boot_means, [100 * alpha, 100 * (1 - alpha)])
    return BootstrapMetricStats(
        metric=metric,
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n_samples=int(arr.size),
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
    )


def bootstrap_summary(
    summary: RagEvalSummary,
    n_bootstrap: int = 5000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> List[BootstrapMetricStats]:
    in_domain = [r for r in summary.results if r.in_domain]
    pool = in_domain or summary.results
    metrics = {
        "faithfulness": [r.faithfulness for r in pool],
        "answer_relevance": [r.answer_relevance for r in pool],
        "context_precision": [r.context_precision for r in pool],
        "hallucination_rate": [r.hallucination_rate for r in pool],
    }
    return [
        compute_bootstrap_stats(name, values, n_bootstrap=n_bootstrap, ci_level=ci_level, seed=seed + i)
        for i, (name, values) in enumerate(metrics.items())
    ]


def evaluate_case(
    case_id: str,
    question: str,
    answer: str,
    contexts: List[str],
    expected_keywords: Optional[List[str]] = None,
    in_domain: bool = True,
    sources: Optional[List[str]] = None,
) -> RagEvalCaseResult:
    return RagEvalCaseResult(
        case_id=case_id,
        question=question,
        answer=answer,
        contexts=contexts,
        faithfulness=faithfulness(answer, contexts),
        answer_relevance=answer_relevance(question, answer),
        context_precision=context_precision(question, contexts, expected_keywords),
        hallucination_rate=hallucination_rate(answer, contexts),
        in_domain=in_domain,
        sources=sources or [],
    )


def evaluate_agent_cases(agent: Any, cases: List[Dict[str, Any]]) -> RagEvalSummary:
    results: List[RagEvalCaseResult] = []
    for case in cases:
        question = case["question"]
        prediction = case.get("prediction")
        chunks = agent.retriever.search(question, k=5, prediction=prediction)
        contexts = [c["text"] for c in chunks]
        sources = [c["source"] for c in chunks]
        answer = agent.generator.generate(question, chunks, prediction)
        results.append(
            evaluate_case(
                case_id=case["id"],
                question=question,
                answer=answer,
                contexts=contexts,
                expected_keywords=case.get("expected_keywords"),
                in_domain=case.get("in_domain", True),
                sources=sources,
            )
        )
    in_domain = [r for r in results if r.in_domain]
    pool = in_domain or results
    n = len(pool)
    rag_summary = RagEvalSummary(
        results=results,
        faithfulness_avg=sum(r.faithfulness for r in pool) / n,
        answer_relevance_avg=sum(r.answer_relevance for r in pool) / n,
        context_precision_avg=sum(r.context_precision for r in pool) / n,
        hallucination_rate_avg=sum(r.hallucination_rate for r in pool) / n,
    )
    rag_summary.bootstrap = bootstrap_summary(rag_summary)
    return rag_summary


def check_thresholds(summary: RagEvalSummary, thresholds: Dict[str, float]) -> List[str]:
    failures: List[str] = []
    if summary.faithfulness_avg < thresholds.get("faithfulness_min", 0.0):
        failures.append(
            f"faithfulness_avg {summary.faithfulness_avg:.3f} < {thresholds['faithfulness_min']}"
        )
    if summary.answer_relevance_avg < thresholds.get("answer_relevance_min", 0.0):
        failures.append(
            f"answer_relevance_avg {summary.answer_relevance_avg:.3f} < {thresholds['answer_relevance_min']}"
        )
    if summary.context_precision_avg < thresholds.get("context_precision_min", 0.0):
        failures.append(
            f"context_precision_avg {summary.context_precision_avg:.3f} < {thresholds['context_precision_min']}"
        )
    max_hall = thresholds.get("hallucination_rate_max")
    if max_hall is not None and summary.hallucination_rate_avg > max_hall:
        failures.append(
            f"hallucination_rate_avg {summary.hallucination_rate_avg:.3f} > {max_hall}"
        )
    return failures
