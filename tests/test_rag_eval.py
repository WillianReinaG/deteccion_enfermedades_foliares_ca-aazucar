import json
from pathlib import Path

import pytest

from app.agent.agent import SugarCaneAgent
from app.rag.eval_metrics import (
    answer_relevance,
    check_thresholds,
    context_precision,
    evaluate_agent_cases,
    faithfulness,
    hallucination_rate,
)

pytestmark = pytest.mark.rag_eval

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "eval" / "rag_eval_dataset.json"


def _load_dataset() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def agent():
    return SugarCaneAgent()


def test_rag_metric_functions_basic():
    ctx = ["La roya en caña de azúcar produce manchas alargadas en las hojas."]
    ans = "La roya produce manchas alargadas en las hojas de caña."
    assert faithfulness(ans, ctx) > 0.3
    assert answer_relevance("síntomas de roya en caña", ans) > 0.0
    assert context_precision("síntomas roya caña", [ctx[0]], ["roya"]) >= 0.5
    assert hallucination_rate(ans, ctx) < 0.8


def test_retriever_indexes_expanded_knowledge_base(agent):
    chunks = agent.retriever.search("mancha roja caña de azúcar síntomas", k=5)
    assert chunks, "El retriever debe indexar documentos en knowledge_base/"
    sources = {c["source"].lower() for c in chunks}
    doc_hints = ("mancha roja", "cartilla", "fitosanitario", "plagas-de-la-cana")
    assert any(any(h in s for h in doc_hints) for s in sources)


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
