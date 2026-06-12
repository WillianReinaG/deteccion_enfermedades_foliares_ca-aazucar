"""Métricas RAG para evaluación del pipeline con embeddings semánticos.

Métricas oficiales del proyecto de grado:
    - Faithfulness: sustento de la respuesta en el contexto recuperado.
    - Answer Relevance: alineación pregunta-respuesta (léxica + semántica).
    - Hallucination Rate: 1 - faithfulness.
    - nDCG@5: calidad del ranking de recuperación con relevancia semántica (coseno).

La nDCG utiliza similitud coseno entre embeddings de la consulta y cada fragmento,
alineada con el retriever semántico (sentence-transformers + FAISS).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np

from app.config.settings import FINETUNED_EMBEDDING_PATH, SEMANTIC_MODEL_NAME

SPANISH_STOP = frozenset(
    "a al algo algunas algunos ante antes como con contra cual cuales cuando de del desde donde dos el ella ellas ellos "
    "en entre era eran es esa esas ese eso esos esta estaba estaban estamos estan estas este esto estos fue fueron ha "
    "habia habian han hasta hay la las le les lo los mas me mi mis mucho muy nos o para pero por porque que quien se "
    "sin sobre su sus tambien te tiene todo tu tus un una uno usted ustedes y ya".split()
)

RAG_METRICS = ("faithfulness", "answer_relevance", "hallucination_rate", "ndcg_at_5")


def content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-záéíóúñü0-9]+", str(text).lower())
    return {t for t in tokens if len(t) > 2 and t not in SPANISH_STOP}


def _context_tokens(contexts: List[str]) -> set[str]:
    toks: set[str] = set()
    for ctx in contexts:
        toks |= content_tokens(ctx)
    return toks


@lru_cache(maxsize=1)
def _get_semantic_encoder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("sentence-transformers requerido para nDCG semántica") from exc
    model_path = FINETUNED_EMBEDDING_PATH if FINETUNED_EMBEDDING_PATH else SEMANTIC_MODEL_NAME
    return SentenceTransformer(model_path)


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Similitud coseno entre embeddings normalizados [0, 1]."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    encoder = _get_semantic_encoder()
    vectors = encoder.encode([text_a, text_b], normalize_embeddings=True)
    cosine = float(np.dot(vectors[0], vectors[1]))
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def semantic_relevance_score(question: str, context: str) -> float:
    """Relevancia semántica consulta-fragmento para nDCG."""
    return semantic_similarity(question, context)


def faithfulness(answer: str, contexts: List[str]) -> float:
    """Proporción de tokens de la respuesta sustentados por el contexto recuperado."""
    ctx_tokens = _context_tokens(contexts)
    ans_tokens = content_tokens(answer)
    if not ans_tokens:
        return 1.0
    if not ctx_tokens:
        return 0.0
    lexical = len(ans_tokens & ctx_tokens) / len(ans_tokens)
    if not contexts:
        return lexical
    semantic = max(semantic_similarity(answer, ctx) for ctx in contexts)
    return max(lexical, 0.4 * lexical + 0.6 * semantic)


def answer_relevance(question: str, answer: str) -> float:
    """Relevancia pregunta-respuesta: combina solapamiento léxico y similitud semántica."""
    q_tokens = content_tokens(question)
    a_tokens = content_tokens(answer)
    lexical = len(q_tokens & a_tokens) / len(q_tokens) if q_tokens and a_tokens else 0.0
    semantic = semantic_similarity(question, answer)
    return 0.35 * lexical + 0.65 * semantic


def hallucination_rate(answer: str, contexts: List[str]) -> float:
    return 1.0 - faithfulness(answer, contexts)


def ndcg_at_k(
    question: str,
    contexts: List[str],
    k: int = 5,
    expected_keywords: Optional[List[str]] = None,
) -> float:
    """nDCG@K con ganancias por relevancia semántica (embeddings).

    ``expected_keywords`` se conserva por compatibilidad de firma pero no se usa;
    la relevancia se mide por similitud embedding consulta-fragmento.
    """
    del expected_keywords
    if not contexts:
        return 0.0
    gains = [semantic_relevance_score(question, ctx) for ctx in contexts[:k]]

    def dcg(vals: List[float]) -> float:
        return sum((2**gain - 1) / math.log2(i + 2) for i, gain in enumerate(vals))

    actual = dcg(gains)
    ideal = dcg(sorted(gains, reverse=True))
    if ideal <= 0:
        return 0.0
    return actual / ideal


@dataclass
class RagEvalCaseResult:
    case_id: str
    question: str
    answer: str
    contexts: List[str]
    faithfulness: float
    answer_relevance: float
    ndcg_at_5: float
    hallucination_rate: float
    in_domain: bool = True
    sources: List[str] = field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_method: str = "semantic"
    no_rag_answer: Optional[str] = None
    no_rag_faithfulness: Optional[float] = None
    no_rag_answer_relevance: Optional[float] = None
    no_rag_hallucination_rate: Optional[float] = None
    no_rag_ndcg_at_5: Optional[float] = None

    def to_case_dict(self) -> Dict[str, Any]:
        payload = {
            "id": self.case_id,
            "question": self.question,
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "ndcg_at_5": self.ndcg_at_5,
            "hallucination_rate": self.hallucination_rate,
            "retrieval_method": self.retrieval_method,
            "in_domain": self.in_domain,
            "sources": self.sources,
        }
        if self.no_rag_answer is not None:
            payload["ablation_no_rag"] = {
                "faithfulness": self.no_rag_faithfulness,
                "answer_relevance": self.no_rag_answer_relevance,
                "ndcg_at_5": self.no_rag_ndcg_at_5,
                "hallucination_rate": self.no_rag_hallucination_rate,
            }
        return payload


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
    ndcg_at_5_avg: float
    hallucination_rate_avg: float
    retrieval_method: str = "semantic"
    bootstrap: List[BootstrapMetricStats] = field(default_factory=list)
    ablation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "metrics": list(RAG_METRICS),
            "retrieval_method": self.retrieval_method,
            "faithfulness_avg": self.faithfulness_avg,
            "answer_relevance_avg": self.answer_relevance_avg,
            "ndcg_at_5_avg": self.ndcg_at_5_avg,
            "hallucination_rate_avg": self.hallucination_rate_avg,
            "cases": [r.to_case_dict() for r in self.results],
        }
        if self.bootstrap:
            payload["bootstrap"] = [b.to_dict() for b in self.bootstrap]
        if self.ablation:
            payload["ablation"] = self.ablation
        return payload


def _safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


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
        "ndcg_at_5": [r.ndcg_at_5 for r in pool],
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
    in_domain: bool = True,
    sources: Optional[List[str]] = None,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    retrieval_method: str = "semantic",
    no_rag_answer: Optional[str] = None,
    expected_keywords: Optional[List[str]] = None,
) -> RagEvalCaseResult:
    del expected_keywords
    no_rag_f = no_rag_ar = no_rag_h = no_rag_ndcg = None
    if no_rag_answer is not None:
        no_rag_f = faithfulness(no_rag_answer, contexts)
        no_rag_ar = answer_relevance(question, no_rag_answer)
        no_rag_h = 1.0 - no_rag_f
        no_rag_ndcg = 0.0

    return RagEvalCaseResult(
        case_id=case_id,
        question=question,
        answer=answer,
        contexts=contexts,
        faithfulness=faithfulness(answer, contexts),
        answer_relevance=answer_relevance(question, answer),
        ndcg_at_5=ndcg_at_k(question, contexts, k=5),
        hallucination_rate=hallucination_rate(answer, contexts),
        in_domain=in_domain,
        sources=sources or [],
        retrieved_chunks=retrieved_chunks or [],
        retrieval_method=retrieval_method,
        no_rag_answer=no_rag_answer,
        no_rag_faithfulness=no_rag_f,
        no_rag_answer_relevance=no_rag_ar,
        no_rag_hallucination_rate=no_rag_h,
        no_rag_ndcg_at_5=no_rag_ndcg,
    )


def _build_ablation_summary(results: List[RagEvalCaseResult]) -> Optional[Dict[str, Any]]:
    pool = [r for r in results if r.in_domain and r.no_rag_answer is not None]
    if not pool:
        return None
    with_rag = {
        "faithfulness": _safe_mean([r.faithfulness for r in pool]),
        "answer_relevance": _safe_mean([r.answer_relevance for r in pool]),
        "ndcg_at_5": _safe_mean([r.ndcg_at_5 for r in pool]),
        "hallucination_rate": _safe_mean([r.hallucination_rate for r in pool]),
    }
    without_rag = {
        "faithfulness": _safe_mean([float(r.no_rag_faithfulness) for r in pool]),
        "answer_relevance": _safe_mean([float(r.no_rag_answer_relevance) for r in pool]),
        "ndcg_at_5": _safe_mean([float(r.no_rag_ndcg_at_5 or 0.0) for r in pool]),
        "hallucination_rate": _safe_mean([float(r.no_rag_hallucination_rate) for r in pool]),
    }
    return {
        "with_rag": with_rag,
        "without_rag": without_rag,
        "delta": {k: with_rag[k] - without_rag[k] for k in with_rag},
        "n_cases": len(pool),
    }


def evaluate_agent_cases(
    agent: Any,
    cases: List[Dict[str, Any]],
    k: int = 5,
    include_ablation: bool = True,
) -> RagEvalSummary:
    results: List[RagEvalCaseResult] = []
    method = getattr(getattr(agent, "retriever", None), "method", "semantic")
    for case in cases:
        question = case["question"]
        prediction = case.get("prediction")
        history = case.get("history")
        chunks = agent.retriever.search(question, k=k, prediction=prediction, history=history)
        contexts = [c["text"] for c in chunks]
        sources = [c["source"] for c in chunks]
        answer = agent.generator.generate(question, chunks, prediction, history=history)

        no_rag_answer = None
        if include_ablation:
            if hasattr(agent.generator, "generate_without_rag"):
                no_rag_answer = agent.generator.generate_without_rag(question, prediction, history=history)
            else:
                no_rag_answer = agent.generator.generate(question, [], prediction, history=history)

        results.append(
            evaluate_case(
                case_id=case["id"],
                question=question,
                answer=answer,
                contexts=contexts,
                in_domain=case.get("in_domain", True),
                sources=sources,
                retrieved_chunks=chunks,
                retrieval_method=method,
                no_rag_answer=no_rag_answer,
            )
        )

    in_domain = [r for r in results if r.in_domain]
    pool = in_domain or results
    summary = RagEvalSummary(
        results=results,
        faithfulness_avg=_safe_mean([r.faithfulness for r in pool]),
        answer_relevance_avg=_safe_mean([r.answer_relevance for r in pool]),
        ndcg_at_5_avg=_safe_mean([r.ndcg_at_5 for r in pool]),
        hallucination_rate_avg=_safe_mean([r.hallucination_rate for r in pool]),
        retrieval_method=method,
    )
    summary.ablation = _build_ablation_summary(results)
    summary.bootstrap = bootstrap_summary(summary)
    return summary


def evaluate_with_retriever(
    retriever: Any,
    generator: Any,
    cases: List[Dict[str, Any]],
    k: int = 5,
    include_ablation: bool = False,
) -> RagEvalSummary:
    """Evalúa pipeline RAG con retriever y generador explícitos (comparación por método)."""
    class _MiniAgent:
        def __init__(self, r, g):
            self.retriever = r
            self.generator = g

    return evaluate_agent_cases(_MiniAgent(retriever, generator), cases, k=k, include_ablation=include_ablation)


def check_thresholds(summary: RagEvalSummary, thresholds: Dict[str, float]) -> List[str]:
    failures: List[str] = []
    checks = {
        "faithfulness_min": ("faithfulness_avg", summary.faithfulness_avg),
        "answer_relevance_min": ("answer_relevance_avg", summary.answer_relevance_avg),
        "ndcg_at_5_min": ("ndcg_at_5_avg", summary.ndcg_at_5_avg),
    }
    for threshold_key, (metric_name, metric_value) in checks.items():
        if threshold_key in thresholds and metric_value < thresholds[threshold_key]:
            failures.append(f"{metric_name} {metric_value:.3f} < {thresholds[threshold_key]}")
    max_hall = thresholds.get("hallucination_rate_max")
    if max_hall is not None and summary.hallucination_rate_avg > max_hall:
        failures.append(f"hallucination_rate_avg {summary.hallucination_rate_avg:.3f} > {max_hall}")
    return failures
