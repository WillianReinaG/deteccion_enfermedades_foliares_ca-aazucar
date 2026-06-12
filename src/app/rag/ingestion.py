"""Ingesta y preprocesamiento de documentos agronómicos para el pipeline RAG.

Chunking optimizado para documentos técnicos (~400 tokens ≈ 1800 caracteres)
con división en límites de oración/párrafo para preservar coherencia semántica.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set

from app.config.settings import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS
from app.rag.preprocessing import AGRONOMY_EXTRA_STOP, enrich_text_for_indexing

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

SPANISH_STOP = frozenset(
    "a al algo algunas algunos ante antes como con contra cual cuales cuando de del desde donde dos el ella ellas ellos "
    "en entre era eran es esa esas ese eso esos esta estaba estaban estamos estan estas este esto estos fue fueron ha "
    "habia habian han hasta hay la las le les lo los me mi mis mucho muy nos o para pero por porque que quien se sin "
    "sobre su sus tambien te tiene todo tu tus un una uno usted ustedes y ya".split()
) | AGRONOMY_EXTRA_STOP

DISEASE_FILE_HINTS: Dict[str, List[str]] = {
    "Rust": ["roya", "rust"],
    "RedRot": ["redrot", "podredumbre", "red_rot"],
    "Mosaic": ["mosaico", "mosaic"],
    "Yellow": ["amarillamiento", "yellow"],
    "Healthy": ["healthy", "hoja_sana", "sana"],
    "Bacterial Blight": ["bacterial", "tizon", "blight"],
}

CHUNK_STRIDE_CHARS = CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n{2,}")


@dataclass
class DocumentChunk:
    """Fragmento indexable con metadatos agronómicos."""

    source: str
    chunk_id: int
    global_chunk_id: int
    title: str
    text: str
    diseases: List[str] = field(default_factory=list)
    char_len: int = 0
    token_len_proxy: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "chunk_id": self.chunk_id,
            "global_chunk_id": self.global_chunk_id,
            "title": self.title,
            "text": self.text,
            "diseases": self.diseases,
            "char_len": self.char_len,
            "token_len_proxy": self.token_len_proxy,
        }


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> Set[str]:
    tokens = re.findall(r"[a-záéíóúñü0-9]+", str(text).lower())
    return {t for t in tokens if len(t) > 2 and t not in SPANISH_STOP}


def read_document(path: Path) -> str:
    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".pdf" and PdfReader is not None:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    return ""


def clean_text(text: str) -> str:
    """Limpieza básica sin enriquecimiento (se aplica al indexar cada chunk)."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    return normalize_text(text)


def detect_diseases(text: str, source: str) -> Set[str]:
    blob = f"{source} {text}".lower()
    found: Set[str] = set()
    for disease, hints in DISEASE_FILE_HINTS.items():
        if disease.lower() in blob:
            found.add(disease)
            continue
        if any(h in blob for h in hints):
            found.add(disease)
    return found


def _split_long_text(body: str, max_size: int, overlap: int) -> List[str]:
    """Divide texto largo respetando oraciones; fallback a ventana de caracteres."""
    if len(body) <= max_size:
        return [body]

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(body) if s.strip()]
    if not sentences:
        sentences = [body]

    parts: List[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_size:
            if current:
                parts.append(current.strip())
                current = ""
            stride = max(1, max_size - overlap)
            for start in range(0, len(sentence), stride):
                segment = sentence[start : start + max_size].strip()
                if segment:
                    parts.append(segment)
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                parts.append(current.strip())
            current = sentence

    if current:
        parts.append(current.strip())

    if not parts:
        stride = max(1, max_size - overlap)
        for start in range(0, len(body), stride):
            segment = body[start : start + max_size].strip()
            if segment:
                parts.append(segment)
    return parts


def chunk_text(
    text: str,
    source: str,
    max_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> List[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return []

    sections: List[Dict[str, Any]] = []
    current_title = "general"
    current_lines: List[str] = []

    def add_parts(body: str, title: str) -> None:
        for part in _split_long_text(body, max_size, overlap):
            part = part.strip()
            if not part:
                continue
            diseases = detect_diseases(f"{title} {part}", source)
            sections.append(
                {
                    "title": title,
                    "text": part,
                    "diseases": sorted(diseases),
                    "chunk_id": len(sections),
                    "char_len": len(part),
                    "token_len_proxy": len(tokenize(part)),
                }
            )

    def flush() -> None:
        if not current_lines:
            return
        body = normalize_text("\n".join(current_lines))
        if body:
            add_parts(body, current_title)

    for line in text.splitlines():
        if re.match(r"^#{1,4}\s+", line):
            flush()
            current_title = re.sub(r"^#{1,4}\s+", "", line).strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()

    if not sections:
        plain = normalize_text(text)
        for part in _split_long_text(plain, max_size, overlap):
            if part:
                sections.append(
                    {
                        "title": "general",
                        "text": part,
                        "diseases": sorted(detect_diseases(part, source)),
                        "chunk_id": len(sections),
                        "char_len": len(part),
                        "token_len_proxy": len(tokenize(part)),
                    }
                )
    return sections


def load_corpus(knowledge_dir: Path) -> tuple[List[DocumentChunk], List[Dict[str, Any]]]:
    knowledge_dir = Path(knowledge_dir)
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    chunks: List[DocumentChunk] = []
    source_files: List[Dict[str, Any]] = []

    for path in sorted(knowledge_dir.glob("**/*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        raw = read_document(path)
        cleaned = clean_text(raw)
        raw_chunks = chunk_text(cleaned, path.name)
        source_files.append(
            {
                "source": path.name,
                "suffix": path.suffix.lower(),
                "text_chars": len(cleaned),
                "chunks": len(raw_chunks),
            }
        )
        for chunk in raw_chunks:
            doc = DocumentChunk(
                source=path.name,
                chunk_id=chunk["chunk_id"],
                global_chunk_id=len(chunks),
                title=chunk["title"],
                text=chunk["text"],
                diseases=chunk["diseases"],
                char_len=chunk.get("char_len", len(chunk["text"])),
                token_len_proxy=chunk.get("token_len_proxy", len(tokenize(chunk["text"]))),
            )
            chunks.append(doc)

    return chunks, source_files


def corpus_to_index_text(chunk: DocumentChunk) -> str:
    base = f"{chunk.title} {chunk.text} {' '.join(chunk.diseases)}"
    return enrich_text_for_indexing(base)
