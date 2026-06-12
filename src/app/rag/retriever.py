"""Recuperador documental modular para el sistema RAG agronómico.

Integra ingesta (``ingestion``), representación vectorial (``embeddings``) y
búsqueda con diversificación MMR. Soporta tres métodos comparables:

    - ``tfidf``: recuperación léxica TF-IDF + coseno.
    - ``bm25``: Okapi BM25 probabilístico.
    - ``semantic``: embeddings multilingües preentrenados + FAISS.
    - ``hybrid``: fusión RRF semantic + BM25 (producción por defecto).

Configurable vía ``RAG_RETRIEVAL_METHOD`` en ``.env``.
"""
from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.config.settings import KNOWLEDGE_DIR, RAG_RETRIEVAL_METHOD, SEMANTIC_MODEL_NAME
from app.rag.embeddings import (
    DEFAULT_SEMANTIC_MODEL,
    EmbeddingBackend,
    TfidfBackend,
    build_corpus_texts,
    create_backend,
)
from app.rag.ingestion import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    CHUNK_STRIDE_CHARS,
    DocumentChunk,
    load_corpus,
    tokenize,
)
from app.rag.preprocessing import enrich_query

DISEASE_SYNONYMS = {
    "Rust": "roya rust puccinia melanocephala manchas naranja rojizas pustulas uredinios esporas hoja caña azúcar control fungicida manejo integrado",
    "RedRot": "red rot podredumbre roja pudricion roja colletotrichum falcatum tallo entrenudos lesiones rojas caña azúcar manejo variedades sanidad",
    "Mosaic": "mosaico mosaic sugarcane mosaic virus scmv virus estrias cloroticas amarillas verdes pulgones material semilla sano",
    "Yellow": "amarillamiento yellow leaf syndrome scylv hoja amarilla nervadura media pulgones nutricion nitrogeno potasio estrés hídrico",
    "Healthy": "hoja sana healthy cultivo sano monitoreo preventivo buenas prácticas agronómicas",
    "Bacterial Blight": "tizón bacteriano bacterial blight xanthomonas manchas acuosas hoja caña",
}

DISEASE_FILE_HINTS: Dict[str, List[str]] = {
    "Rust": ["roya", "rust"],
    "RedRot": ["redrot", "podredumbre", "red_rot"],
    "Mosaic": ["mosaico", "mosaic"],
    "Yellow": ["amarillamiento", "yellow"],
    "Healthy": ["healthy", "hoja_sana", "sana"],
    "Bacterial Blight": ["bacterial", "tizon", "blight"],
}

QUERY_INTENT_KEYWORDS = {
    "sintoma": ["síntoma", "sintoma", "signo", "manifest", "presenta", "aparece", "veo"],
    "manejo": ["manejo", "control", "tratar", "mitigar", "aplicar", "fungicida", "prevención", "prevenir"],
    "causa": ["causa", "patógeno", "agente", "virus", "hongo", "bacteria", "por qué", "origen"],
    "diagnostico": ["diagnóstico", "diagnostico", "confirmar", "diferenciar", "identificar", "detect"],
}

DEFAULT_TOP_K = 5
MMR_LAMBDA = 0.72
MIN_SCORE = 0.08


class ModularRetriever:
    """Recuperador con backend intercambiable y diversificación MMR."""

    def __init__(
        self,
        knowledge_dir: Path = KNOWLEDGE_DIR,
        method: str = RAG_RETRIEVAL_METHOD,
        semantic_model: str = SEMANTIC_MODEL_NAME,
    ):
        self.knowledge_dir = Path(knowledge_dir)
        self.method = method.lower().strip()
        self.semantic_model = semantic_model
        self.chunks: List[DocumentChunk] = []
        self.docs: List[Dict] = []
        self.source_files: List[Dict[str, Any]] = []
        self.corpus_texts: List[str] = []
        self.backend: EmbeddingBackend = create_backend(method, semantic_model=semantic_model)
        self._tfidf_for_mmr: Optional[TfidfBackend] = None
        self._build()

    def _build(self) -> None:
        self.chunks, self.source_files = load_corpus(self.knowledge_dir)
        self.docs = [c.to_dict() for c in self.chunks]
        self.corpus_texts = build_corpus_texts(self.chunks)
        if self.corpus_texts:
            self.backend.fit(self.corpus_texts)
            if self.method != "tfidf":
                self._tfidf_for_mmr = TfidfBackend()
                self._tfidf_for_mmr.fit(self.corpus_texts)

    def expand_query(
        self,
        query: str,
        prediction: Dict | None = None,
        history: List[Dict] | None = None,
    ) -> str:
        parts = [query]
        q_lower = query.lower()
        for intent, kws in QUERY_INTENT_KEYWORDS.items():
            if any(k in q_lower for k in kws):
                parts.append(intent)
        if prediction:
            disease = prediction.get("class_name", "")
            parts.append(f"enfermedad {disease} caña de azúcar síntomas manejo control prevención diagnóstico")
            parts.append(DISEASE_SYNONYMS.get(disease, ""))
            for hint in DISEASE_FILE_HINTS.get(disease, []):
                parts.append(hint)
        if history:
            recent = " ".join(
                m.get("content", "")
                for m in history[-4:]
                if m.get("role") in {"user", "assistant"}
            )
            parts.append(recent[:800])
        return enrich_query(" ".join(parts))

    def _keyword_score(self, query_tokens: Set[str], doc: Dict) -> float:
        doc_tokens = tokenize(f"{doc['title']} {doc['text']} {' '.join(doc['diseases'])}")
        if not query_tokens or not doc_tokens:
            return 0.0
        overlap = len(query_tokens & doc_tokens)
        return overlap / max(len(query_tokens), 1)

    def _disease_boost(self, doc: Dict, prediction: Dict | None) -> float:
        if not prediction:
            return 0.0
        disease = prediction.get("class_name", "")
        if not disease:
            return 0.0
        boost = 0.0
        if disease in doc.get("diseases", []):
            boost += 0.35
        source_lower = doc["source"].lower()
        for hint in DISEASE_FILE_HINTS.get(disease, []):
            if hint in source_lower:
                boost += 0.25
        if disease.lower() in doc.get("text", "").lower():
            boost += 0.10
        return min(boost, 0.55)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_s, max_s = float(scores.min()), float(scores.max())
        if max_s - min_s < 1e-9:
            return np.ones_like(scores, dtype=float)
        return (scores - min_s) / (max_s - min_s)

    def _mmr_select(self, candidates: List[Dict], k: int, lambda_param: float = MMR_LAMBDA) -> List[Dict]:
        if not candidates:
            return []
        selected: List[Dict] = []
        remaining = candidates.copy()
        mmr_backend = self._tfidf_for_mmr if self._tfidf_for_mmr else (
            self.backend if isinstance(self.backend, TfidfBackend) else None
        )
        doc_matrix = mmr_backend.matrix if mmr_backend and mmr_backend.matrix is not None else None

        while remaining and len(selected) < k:
            best_idx, best_score = -1, -1.0
            for i, cand in enumerate(remaining):
                rel = cand["final_score"]
                div_penalty = 0.0
                if selected and doc_matrix is not None:
                    cand_idx = cand.get("global_chunk_id")
                    if cand_idx is not None:
                        sims = []
                        for sel in selected:
                            sel_idx = sel.get("global_chunk_id")
                            if sel_idx is not None:
                                sims.append(float(cosine_similarity(doc_matrix[cand_idx], doc_matrix[sel_idx])[0, 0]))
                        if sims:
                            div_penalty = max(sims)
                mmr_score = lambda_param * rel - (1 - lambda_param) * div_penalty
                if mmr_score > best_score:
                    best_score, best_idx = mmr_score, i
            if best_idx < 0:
                break
            chosen = remaining.pop(best_idx)
            chosen["mmr_score"] = best_score
            selected.append(chosen)
        return selected

    def search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        prediction: Dict | None = None,
        history: List[Dict] | None = None,
        min_score: float = MIN_SCORE,
    ) -> List[Dict]:
        if not self.docs:
            return []

        expanded = self.expand_query(query, prediction, history)
        query_tokens = tokenize(expanded)
        pool_size = max(k * 5, 20) if self.method in {"semantic", "hybrid"} else max(k * 3, 12)
        idx, raw_scores = self.backend.score(expanded, top_k=pool_size)
        norm_scores = self._normalize_scores(raw_scores)

        candidates: List[Dict] = []
        for rank, (doc_idx, retrieval_score, norm_score) in enumerate(
            zip(idx, raw_scores, norm_scores), start=1
        ):
            doc = self.docs[int(doc_idx)]
            kw = self._keyword_score(query_tokens, doc)
            disease_b = self._disease_boost(doc, prediction)
            final = 0.55 * float(norm_score) + 0.30 * kw + disease_b
            if final < min_score and not (prediction and disease_b > 0.2):
                continue
            candidates.append(
                {
                    **doc,
                    "rank_pre_mmr": rank,
                    "score": float(retrieval_score),
                    "normalized_score": float(norm_score),
                    "keyword_score": kw,
                    "disease_boost": disease_b,
                    "final_score": final,
                    "expanded_query": expanded,
                    "retrieval_method": self.method,
                }
            )

        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        for rank, cand in enumerate(candidates, start=1):
            cand["rank_pre_mmr"] = rank

        selected = self._mmr_select(candidates[:pool_size], k)
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank
        return selected

    def corpus_report(self) -> Dict[str, Any]:
        lengths = [int(d.get("char_len", len(d["text"]))) for d in self.docs]
        token_lengths = [int(d.get("token_len_proxy", 0)) for d in self.docs]
        disease_counts = Counter()
        for d in self.docs:
            if d.get("diseases"):
                disease_counts.update(d["diseases"])
            else:
                disease_counts.update(["general"])

        suffix_counts = Counter(f["suffix"] for f in self.source_files)
        backend_meta = self.backend.metadata
        report: Dict[str, Any] = {
            "knowledge_dir": str(self.knowledge_dir),
            "document_count": len(self.source_files),
            "chunk_count": len(self.docs),
            "file_types": dict(suffix_counts),
            "chunking": {
                "strategy": "Markdown headings when available; sliding character window otherwise",
                "chunk_size_chars": CHUNK_SIZE_CHARS,
                "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
                "chunk_stride_chars": CHUNK_STRIDE_CHARS,
            },
            "retrieval": {
                "method": self.method,
                "backend": backend_meta,
                "scoring_weights": {
                    "retrieval_score": 0.55,
                    "keyword_overlap": 0.25,
                    "disease_boost_max": 0.55,
                },
                "mmr": {"enabled": True, "lambda_param": MMR_LAMBDA},
                "default_top_k": DEFAULT_TOP_K,
                "min_score": MIN_SCORE,
            },
            "disease_chunk_distribution": dict(disease_counts),
            "source_files": self.source_files,
        }
        if lengths:
            report["chunk_length_stats_chars"] = {
                "min": min(lengths),
                "max": max(lengths),
                "mean": statistics.mean(lengths),
                "median": statistics.median(lengths),
            }
        if token_lengths:
            report["chunk_length_stats_token_proxy"] = {
                "min": min(token_lengths),
                "max": max(token_lengths),
                "mean": statistics.mean(token_lengths),
                "median": statistics.median(token_lengths),
            }
        return report


class LocalRetriever(ModularRetriever):
    """Alias retrocompatible usado por el agente conversacional."""

    def __init__(self, knowledge_dir: Path = KNOWLEDGE_DIR):
        super().__init__(
            knowledge_dir=knowledge_dir,
            method=RAG_RETRIEVAL_METHOD,
            semantic_model=SEMANTIC_MODEL_NAME,
        )
