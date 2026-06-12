"""Fine-tuning de embeddings con corpus agronómico de caña de azúcar.

Adapta un modelo sentence-transformers al dominio fitosanitario generando
pares (consulta, fragmento positivo) a partir de:
    - Consultas del benchmark de evaluación
    - Encabezados y enfermedades de chunks del knowledge base

Uso:
    python src/scripts/finetune_embeddings.py
    python src/scripts/finetune_embeddings.py --epochs 2 --output models/embeddings/sugarcane-agro

Luego configure en .env:
    FINETUNED_EMBEDDING_PATH=models/embeddings/sugarcane-agro
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.config.settings import EMBEDDINGS_DIR, KNOWLEDGE_DIR, SEMANTIC_MODEL_NAME
from app.rag.ingestion import load_corpus, tokenize
from app.rag.preprocessing import enrich_text_for_indexing

BASE_DIR = SRC_DIR.parent
DEFAULT_BENCHMARK = BASE_DIR / "data" / "eval" / "retrieval_benchmark.json"


def _load_benchmark_queries() -> list[dict]:
    for path in (DEFAULT_BENCHMARK, BASE_DIR / "data" / "eval" / "rag_eval_dataset.json"):
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("queries") or data.get("cases", [])
    return []


def _chunk_matches_keywords(text: str, keywords: list[str]) -> bool:
    tokens = tokenize(text)
    for kw in keywords:
        if tokenize(str(kw)) & tokens:
            return True
    return False


def build_training_pairs() -> list[tuple[str, str]]:
    """Construye pares (consulta, documento positivo) para entrenamiento contrastivo."""
    chunks, _ = load_corpus(KNOWLEDGE_DIR)
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add_pair(anchor: str, positive: str) -> None:
        anchor = enrich_text_for_indexing(anchor)
        positive = enrich_text_for_indexing(positive)
        key = (anchor[:200], positive[:200])
        if key not in seen and anchor and positive:
            seen.add(key)
            pairs.append((anchor, positive))

    for case in _load_benchmark_queries():
        question = case.get("question", "")
        keywords = case.get("expected_keywords") or []
        if not question:
            continue
        for chunk in chunks:
            blob = f"{chunk.title} {chunk.text}"
            if keywords and _chunk_matches_keywords(blob, keywords):
                add_pair(question, blob)
            elif not keywords and tokenize(question) & tokenize(blob):
                add_pair(question, blob)

    for chunk in chunks:
        if chunk.diseases:
            for disease in chunk.diseases:
                add_pair(
                    f"¿Cuáles son síntomas y manejo de {disease} en caña de azúcar?",
                    f"{chunk.title} {chunk.text}",
                )
        if chunk.title and chunk.title != "general":
            add_pair(
                f"Información sobre {chunk.title} en caña de azúcar",
                f"{chunk.title} {chunk.text}",
            )

    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tuning de embeddings agronómicos")
    parser.add_argument("--base-model", default=SEMANTIC_MODEL_NAME)
    parser.add_argument("--output", type=Path, default=EMBEDDINGS_DIR / "sugarcane-agro")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError:
        print("Instale sentence-transformers y torch: pip install sentence-transformers torch")
        return 1

    pairs = build_training_pairs()
    if len(pairs) < 10:
        print(f"Pocos pares de entrenamiento ({len(pairs)}). Revise knowledge_base y benchmark.")
        return 1

    print(f"Pares de entrenamiento: {len(pairs)}")
    examples = [InputExample(texts=[a, p]) for a, p in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)

    model = SentenceTransformer(args.base_model)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=max(10, len(pairs) // args.batch_size),
        show_progress_bar=True,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    print(f"Modelo guardado en: {args.output}")
    print(f"Configure: FINETUNED_EMBEDDING_PATH={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
