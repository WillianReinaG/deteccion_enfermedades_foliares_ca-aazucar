"""Backends de representación vectorial para recuperación documental.

Métodos soportados:
    - tfidf: léxico-estadístico
    - bm25: probabilístico léxico
    - semantic: embeddings densos (sentence-transformers / OpenAI) + FAISS
    - hybrid: fusión RRF de semantic + BM25 (recomendado en producción)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config.settings import (
    FINETUNED_EMBEDDING_PATH,
    HYBRID_BM25_WEIGHT,
    HYBRID_RRF_K,
    HYBRID_SEMANTIC_WEIGHT,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    USE_OPENAI_EMBEDDINGS,
)
from app.rag.ingestion import SPANISH_STOP, corpus_to_index_text, tokenize
from app.rag.preprocessing import enrich_text_for_indexing

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

DEFAULT_SEMANTIC_MODEL = "paraphrase-multilingual-mpnet-base-v2"
RETRIEVAL_METHODS = ("tfidf", "bm25", "semantic", "hybrid")


def _resolve_semantic_model_name(semantic_model: str) -> str:
    if FINETUNED_EMBEDDING_PATH and Path(FINETUNED_EMBEDDING_PATH).exists():
        return FINETUNED_EMBEDDING_PATH
    return semantic_model


def reciprocal_rank_fusion(
    ranked_lists: List[List[int]],
    weights: Optional[List[float]] = None,
    rrf_k: int = 60,
) -> Dict[int, float]:
    """Fusión por RRF: combina rankings de múltiples retrievers."""
    scores: Dict[int, float] = {}
    for i, ranking in enumerate(ranked_lists):
        w = weights[i] if weights and i < len(weights) else 1.0
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (rrf_k + rank + 1)
    return scores


class EmbeddingBackend(ABC):
    name: str

    @abstractmethod
    def fit(self, corpus_texts: List[str]) -> None:
        pass

    @abstractmethod
    def score(self, query: str, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        pass

    @property
    @abstractmethod
    def metadata(self) -> dict:
        pass


class TfidfBackend(EmbeddingBackend):
    name = "tfidf"

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words=list(SPANISH_STOP),
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            max_df=0.95,
        )
        self.matrix = None

    def fit(self, corpus_texts: List[str]) -> None:
        enriched = [enrich_text_for_indexing(t) for t in corpus_texts]
        self.matrix = self.vectorizer.fit_transform(enriched)

    def score(self, query: str, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.matrix is None:
            return np.array([], dtype=int), np.array([], dtype=float)
        q_vec = self.vectorizer.transform([enrich_text_for_indexing(query)])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        k = min(top_k, len(sims))
        if k == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return idx, sims[idx]

    @property
    def metadata(self) -> dict:
        return {
            "method": self.name,
            "vectorizer": "TfidfVectorizer",
            "ngram_range": [1, 2],
            "similarity": "cosine",
            "preprocessing": "agronomy_term_expansion",
            "vocabulary_size": len(getattr(self.vectorizer, "vocabulary_", {}) or {}),
        }


class BM25Backend(EmbeddingBackend):
    name = "bm25"

    def __init__(self) -> None:
        self._bm25 = None
        self._tokenized_corpus: List[List[str]] = []

    def _tokenize_list(self, text: str) -> List[str]:
        return list(tokenize(enrich_text_for_indexing(text)))

    def fit(self, corpus_texts: List[str]) -> None:
        if BM25Okapi is None:
            raise ImportError("Instale rank-bm25: pip install rank-bm25")
        self._tokenized_corpus = [self._tokenize_list(t) for t in corpus_texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def score(self, query: str, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._bm25 is None:
            return np.array([], dtype=int), np.array([], dtype=float)
        q_tokens = self._tokenize_list(query)
        scores = self._bm25.get_scores(q_tokens)
        k = min(top_k, len(scores))
        if k == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return idx, scores[idx]

    def score_all(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            return np.array([], dtype=float)
        return self._bm25.get_scores(self._tokenize_list(query))

    @property
    def metadata(self) -> dict:
        return {
            "method": self.name,
            "algorithm": "BM25Okapi",
            "preprocessing": "agronomy_term_expansion",
        }


class SemanticBackend(EmbeddingBackend):
    name = "semantic"

    def __init__(
        self,
        model_name: str = DEFAULT_SEMANTIC_MODEL,
        cache_dir: Optional[Path] = None,
        use_openai: bool = False,
        openai_model: str = OPENAI_EMBEDDING_MODEL,
    ) -> None:
        self.model_name = _resolve_semantic_model_name(model_name)
        self.cache_dir = cache_dir
        self.use_openai = use_openai and bool(OPENAI_API_KEY)
        self.openai_model = openai_model
        self.model: Optional[SentenceTransformer] = None
        self.index = None
        self.dim = 0
        self._corpus_size = 0
        self._openai_client = None

    def _load_st_model(self) -> SentenceTransformer:
        if SentenceTransformer is None:
            raise ImportError("Instale sentence-transformers")
        if self.model is None:
            kwargs = {}
            if self.cache_dir:
                kwargs["cache_folder"] = str(self.cache_dir)
            self.model = SentenceTransformer(self.model_name, **kwargs)
        return self.model

    def _load_openai(self):
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=OPENAI_API_KEY)
        return self._openai_client

    def _encode_openai(self, texts: List[str]) -> np.ndarray:
        client = self._load_openai()
        enriched = [enrich_text_for_indexing(t) for t in texts]
        response = client.embeddings.create(model=self.openai_model, input=enriched)
        vectors = [item.embedding for item in response.data]
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        return arr / norms

    def _encode_local(self, texts: List[str]) -> np.ndarray:
        model = self._load_st_model()
        enriched = [enrich_text_for_indexing(t) for t in texts]
        return np.asarray(
            model.encode(enriched, batch_size=32, show_progress_bar=False, normalize_embeddings=True),
            dtype=np.float32,
        )

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.use_openai:
            return self._encode_openai(texts)
        return self._encode_local(texts)

    def fit(self, corpus_texts: List[str]) -> None:
        if faiss is None:
            raise ImportError("Instale faiss-cpu")
        embeddings = self.encode(corpus_texts)
        self.dim = embeddings.shape[1]
        self._corpus_size = len(corpus_texts)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

    def score(self, query: str, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None or self.index.ntotal == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        q_emb = self.encode([query])
        k = min(top_k, self.index.ntotal)
        scores, idx = self.index.search(q_emb, k)
        return idx[0], scores[0]

    @property
    def metadata(self) -> dict:
        return {
            "method": self.name,
            "model": self.model_name,
            "provider": "openai" if self.use_openai else "sentence-transformers",
            "openai_model": self.openai_model if self.use_openai else None,
            "finetuned": bool(FINETUNED_EMBEDDING_PATH),
            "index": "FAISS IndexFlatIP",
            "embedding_dim": self.dim,
            "corpus_size": self._corpus_size,
            "preprocessing": "agronomy_term_expansion",
        }


class HybridBackend(EmbeddingBackend):
    """Fusión híbrida: embeddings semánticos + BM25 vía RRF."""

    name = "hybrid"

    def __init__(
        self,
        semantic_model: str = DEFAULT_SEMANTIC_MODEL,
        cache_dir: Optional[Path] = None,
        rrf_k: int = HYBRID_RRF_K,
        semantic_weight: float = HYBRID_SEMANTIC_WEIGHT,
        bm25_weight: float = HYBRID_BM25_WEIGHT,
        use_openai: bool = USE_OPENAI_EMBEDDINGS,
    ) -> None:
        self.semantic = SemanticBackend(
            model_name=semantic_model,
            cache_dir=cache_dir,
            use_openai=use_openai,
        )
        self.bm25 = BM25Backend()
        self.rrf_k = rrf_k
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self._corpus_size = 0

    def fit(self, corpus_texts: List[str]) -> None:
        self._corpus_size = len(corpus_texts)
        self.semantic.fit(corpus_texts)
        self.bm25.fit(corpus_texts)

    def score(self, query: str, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        pool = min(max(top_k * 4, 24), self._corpus_size)
        sem_idx, _ = self.semantic.score(query, top_k=pool)
        bm25_scores = self.bm25.score_all(query)
        bm25_ranking = np.argsort(-bm25_scores)[:pool].tolist()
        sem_ranking = sem_idx.tolist()

        fused = reciprocal_rank_fusion(
            [sem_ranking, bm25_ranking],
            weights=[self.semantic_weight, self.bm25_weight],
            rrf_k=self.rrf_k,
        )
        if not fused:
            return np.array([], dtype=int), np.array([], dtype=float)

        sorted_docs = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        idx = np.array([d for d, _ in sorted_docs], dtype=int)
        scores = np.array([s for _, s in sorted_docs], dtype=float)
        return idx, scores

    @property
    def metadata(self) -> dict:
        return {
            "method": self.name,
            "fusion": "RRF",
            "rrf_k": self.rrf_k,
            "semantic_weight": self.semantic_weight,
            "bm25_weight": self.bm25_weight,
            "semantic_backend": self.semantic.metadata,
            "bm25_backend": self.bm25.metadata,
        }


def create_backend(
    method: str,
    semantic_model: str = DEFAULT_SEMANTIC_MODEL,
    cache_dir: Optional[Path] = None,
) -> EmbeddingBackend:
    method = method.lower().strip()
    if method == "tfidf":
        return TfidfBackend()
    if method == "bm25":
        return BM25Backend()
    if method == "semantic":
        return SemanticBackend(
            model_name=semantic_model,
            cache_dir=cache_dir,
            use_openai=USE_OPENAI_EMBEDDINGS,
        )
    if method == "hybrid":
        return HybridBackend(
            semantic_model=semantic_model,
            cache_dir=cache_dir,
            use_openai=USE_OPENAI_EMBEDDINGS,
        )
    raise ValueError(f"Método desconocido: {method}. Opciones: {RETRIEVAL_METHODS}")


def build_corpus_texts(chunks) -> List[str]:
    return [corpus_to_index_text(c) for c in chunks]
