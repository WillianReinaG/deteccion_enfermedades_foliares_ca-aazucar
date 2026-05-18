from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Set
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.config.settings import KNOWLEDGE_DIR

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

SPANISH_STOP = frozenset(
    "a al algo algunas algunos ante antes como con contra cual cuales cuando de del desde donde dos el ella ellas ellos en entre era eran es esa esas ese eso esos esta estaba estaban estamos estan estas este esto estos fue fueron ha habia habian han hasta hay la las le les lo los me mi mis mucho muy nos o para pero por porque que quien se sin sobre su sus tambien te tiene todo tu tus un una uno usted ustedes y ya".split()
)

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


def _read_file(path: Path) -> str:
    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".pdf" and PdfReader is not None:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return ""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> Set[str]:
    tokens = re.findall(r"[a-záéíóúñü0-9]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in SPANISH_STOP}


def _detect_diseases(text: str, source: str) -> Set[str]:
    blob = f"{source} {text}".lower()
    found: Set[str] = set()
    for disease, hints in DISEASE_FILE_HINTS.items():
        if disease.lower() in blob:
            found.add(disease)
            continue
        if any(h in blob for h in hints):
            found.add(disease)
    return found


def _chunks_markdown(text: str, source: str, max_size: int = 900) -> List[Dict]:
    """Divide por encabezados Markdown para conservar contexto semántico por enfermedad."""
    text = text.strip()
    if not text:
        return []
    sections: List[Dict] = []
    current_title = "general"
    current_lines: List[str] = []

    def flush():
        if not current_lines:
            return
        body = _normalize("\n".join(current_lines))
        if not body:
            return
        parts = [body]
        if len(body) > max_size:
            parts = []
            start = 0
            while start < len(body):
                parts.append(body[start : start + max_size])
                start += max(1, max_size - 200)
        for i, part in enumerate(parts):
            diseases = _detect_diseases(f"{current_title} {part}", source)
            sections.append(
                {
                    "title": current_title,
                    "text": part,
                    "diseases": sorted(diseases),
                    "chunk_id": len(sections),
                }
            )

    for line in text.splitlines():
        if re.match(r"^#{1,4}\s+", line):
            flush()
            current_title = re.sub(r"^#{1,4}\s+", "", line).strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    if not sections:
        plain = _normalize(text)
        for i, start in enumerate(range(0, len(plain), max_size - 200)):
            part = plain[start : start + max_size]
            if part:
                sections.append(
                    {
                        "title": "general",
                        "text": part,
                        "diseases": sorted(_detect_diseases(part, source)),
                        "chunk_id": i,
                    }
                )
    return sections


class LocalRetriever:
    def __init__(self, knowledge_dir: Path = KNOWLEDGE_DIR):
        self.knowledge_dir = Path(knowledge_dir)
        self.docs: List[Dict] = []
        self.vectorizer = TfidfVectorizer(
            stop_words=list(SPANISH_STOP),
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            max_df=0.95,
        )
        self.matrix = None
        self._build()

    def _build(self):
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.docs = []
        for path in sorted(self.knowledge_dir.glob("**/*")):
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}:
                text = _read_file(path)
                for chunk in _chunks_markdown(text, path.name):
                    self.docs.append(
                        {
                            "source": path.name,
                            "chunk_id": chunk["chunk_id"],
                            "title": chunk["title"],
                            "text": chunk["text"],
                            "diseases": chunk["diseases"],
                        }
                    )
        if self.docs:
            corpus = [f"{d['title']} {d['text']} {' '.join(d['diseases'])}" for d in self.docs]
            self.matrix = self.vectorizer.fit_transform(corpus)

    def expand_query(self, query: str, prediction: Dict | None = None, history: List[Dict] | None = None) -> str:
        parts = [query]
        q_lower = query.lower()
        for intent, kws in QUERY_INTENT_KEYWORDS.items():
            if any(k in q_lower for k in kws):
                parts.append(intent)
        if prediction:
            disease = prediction.get("class_name", "")
            parts.append(
                f"enfermedad {disease} caña de azúcar síntomas manejo control prevención diagnóstico"
            )
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
        return " ".join(parts)

    def _keyword_score(self, query_tokens: Set[str], doc: Dict) -> float:
        doc_tokens = _tokenize(f"{doc['title']} {doc['text']} {' '.join(doc['diseases'])}")
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

    def _mmr_select(
        self,
        candidates: List[Dict],
        query_vec,
        k: int,
        lambda_param: float = 0.72,
    ) -> List[Dict]:
        """Maximal Marginal Relevance: diversifica fuentes sin perder relevancia."""
        if not candidates:
            return []
        selected: List[Dict] = []
        remaining = candidates.copy()
        doc_texts = [f"{d['title']} {d['text']}" for d in self.docs]
        doc_matrix = self.vectorizer.transform(doc_texts) if self.matrix is not None else None

        while remaining and len(selected) < k:
            best_idx, best_score = -1, -1.0
            for i, cand in enumerate(remaining):
                rel = cand["final_score"]
                div_penalty = 0.0
                if selected and doc_matrix is not None:
                    cand_idx = next(
                        (j for j, d in enumerate(self.docs) if d["source"] == cand["source"] and d["chunk_id"] == cand["chunk_id"]),
                        None,
                    )
                    if cand_idx is not None:
                        sims = []
                        for sel in selected:
                            sel_idx = next(
                                (j for j, d in enumerate(self.docs) if d["source"] == sel["source"] and d["chunk_id"] == sel["chunk_id"]),
                                None,
                            )
                            if sel_idx is not None:
                                sims.append(float(cosine_similarity(doc_matrix[cand_idx], doc_matrix[sel_idx])[0, 0]))
                        if sims:
                            div_penalty = max(sims)
                mmr = lambda_param * rel - (1 - lambda_param) * div_penalty
                if mmr > best_score:
                    best_score, best_idx = mmr, i
            if best_idx < 0:
                break
            selected.append(remaining.pop(best_idx))
        return selected

    def search(
        self,
        query: str,
        k: int = 5,
        prediction: Dict | None = None,
        history: List[Dict] | None = None,
        min_score: float = 0.08,
    ) -> List[Dict]:
        if not self.docs or self.matrix is None:
            return []
        expanded = self.expand_query(query, prediction, history)
        q_vec = self.vectorizer.transform([expanded])
        tfidf_sims = cosine_similarity(q_vec, self.matrix)[0]
        query_tokens = _tokenize(expanded)

        candidates: List[Dict] = []
        for i, doc in enumerate(self.docs):
            tfidf = float(tfidf_sims[i])
            kw = self._keyword_score(query_tokens, doc)
            disease_b = self._disease_boost(doc, prediction)
            final = 0.55 * tfidf + 0.25 * kw + disease_b
            if final < min_score and not (prediction and disease_b > 0.2):
                continue
            candidates.append({**doc, "score": tfidf, "final_score": final})

        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        top_pool = candidates[: max(k * 3, 12)]
        return self._mmr_select(top_pool, q_vec, k)
