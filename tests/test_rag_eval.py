import json
from pathlib import Path

import pytest

from app.agent.agent import SugarCaneAgent
from app.rag.eval_metrics import (
    RAG_METRICS,
    answer_relevance,
    bootstrap_summary,
    check_thresholds,
    compute_bootstrap_stats,
    evaluate_agent_cases,
    faithfulness,
    hallucination_rate,
    ndcg_at_k,
    RagEvalCaseResult,
    RagEvalSummary,
)

pytestmark = pytest.mark.rag_eval

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "eval" / "rag_eval_dataset.json"


def _load_dataset() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def agent():
    return SugarCaneAgent()


def test_rag_metrics_official_set():
    assert RAG_METRICS == ("faithfulness", "answer_relevance", "hallucination_rate", "ndcg_at_5")


def test_bootstrap_stats_computes_ci():
    values = [0.4, 0.5, 0.6, 0.55, 0.45, 0.52, 0.58]
    stat = compute_bootstrap_stats("faithfulness", values, n_bootstrap=2000, seed=7)
    assert stat.mean == sum(values) / len(values)
    assert stat.std > 0
    assert stat.ci_lower <= stat.mean <= stat.ci_upper

    dummy = RagEvalSummary(
        results=[
            RagEvalCaseResult(
                case_id="a",
                question="q",
                answer="a",
                contexts=["ctx"],
                faithfulness=0.6,
                answer_relevance=0.7,
                ndcg_at_5=0.8,
                hallucination_rate=0.4,
                in_domain=True,
            ),
            RagEvalCaseResult(
                case_id="b",
                question="q2",
                answer="b",
                contexts=["ctx2"],
                faithfulness=0.5,
                answer_relevance=0.6,
                ndcg_at_5=0.75,
                hallucination_rate=0.5,
                in_domain=True,
            ),
        ],
        faithfulness_avg=0.55,
        answer_relevance_avg=0.65,
        ndcg_at_5_avg=0.775,
        hallucination_rate_avg=0.45,
    )
    boot = bootstrap_summary(dummy, n_bootstrap=1000, seed=1)
    assert len(boot) == 4
    assert all(b.ci_lower <= b.mean <= b.ci_upper for b in boot)


def test_rag_metric_functions_basic():
    ctx = ["La roya en caña de azúcar produce manchas alargadas en las hojas."]
    ans = "La roya produce manchas alargadas en las hojas de caña."
    assert faithfulness(ans, ctx) > 0.3
    assert answer_relevance("síntomas de roya en caña", ans) > 0.0
    assert ndcg_at_k("síntomas de roya en caña", ctx, k=1) > 0.0
    assert hallucination_rate(ans, ctx) < 0.8


def test_retriever_uses_semantic_embeddings(agent):
    assert agent.retriever.method == "semantic"


def test_retriever_indexes_knowledge_base(agent):
    chunks = agent.retriever.search("mancha roja caña de azúcar síntomas", k=5)
    assert chunks, "El retriever debe indexar documentos en knowledge_base/"


def test_rag_pipeline_meets_thresholds(agent):
    dataset = _load_dataset()
    summary = evaluate_agent_cases(agent, dataset["cases"])
    failures = check_thresholds(summary, dataset["thresholds"])
    if failures:
        detail = json.dumps(summary.to_dict(), ensure_ascii=False, indent=2)
        pytest.fail("Métricas RAG por debajo del umbral:\n" + "\n".join(failures) + f"\n{detail}")


def test_in_domain_cases_retrieve_context(agent):
    dataset = _load_dataset()
    for case in dataset["cases"]:
        if not case.get("in_domain", True):
            continue
        chunks = agent.retriever.search(case["question"], k=5, prediction=case.get("prediction"))
        assert chunks, f"Sin contexto recuperado para caso {case['id']}"
