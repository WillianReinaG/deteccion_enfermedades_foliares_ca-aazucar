"""Preprocesamiento especializado para dominio agronómico.

Normaliza siglas y términos técnicos, expande sinónimos de fitopatología
y enriquece texto antes de indexación/embedding. Mejora la representación
semántica de vocabulario de campo (roya, fertirriego, MIP, NPK, etc.).
"""
from __future__ import annotations

import re
from typing import Dict, List

# Expansión de siglas y términos técnicos → formas descriptivas para embeddings
AGRONOMY_TERM_EXPANSIONS: Dict[str, str] = {
    r"\bnpk\b": "nitrógeno fósforo potasio fertilización",
    r"\bmip\b": "manejo integrado de plagas",
    r"\bmid\b": "manejo integrado de enfermedades",
    r"\bfertirriego\b": "fertirriego fertilización riego nutrición",
    r"\bfitosanitario\b": "fitosanitario control sanitario agrícola",
    r"\bbroca\b": "broca insecto plaga barrenador",
    r"\bantracnosis\b": "antracnosis hongo colletotrichum enfermedad",
    r"\broya\b": "roya rust puccinia enfermedad fúngica hoja",
    r"\bmosaico\b": "mosaico virus scmv enfermedad viral",
    r"\bpodredumbre\s+roja\b": "podredumbre roja red rot colletotrichum",
    r"\bamarillamiento\b": "amarillamiento yellow leaf síndrome hoja amarilla",
    r"\btizón\s+bacteriano\b": "tizón bacteriano bacterial blight xanthomonas",
    r"\bscmv\b": "sugarcane mosaic virus mosaico caña",
    r"\bscylv\b": "sugarcane yellow leaf virus amarillamiento",
    r"\bipm\b": "manejo integrado plagas enfermedades",
    r"\buca\b": "unidad cosechadora de azúcar rendimiento",
    r"\btha\b": "toneladas por hectárea rendimiento",
    r"\budm\b": "unidades de daño manejo integrado",
}

# Stopwords adicionales de documentos técnicos (ruido frecuente en PDFs)
AGRONOMY_EXTRA_STOP = frozenset(
    "fig figura tabla cuadro página pag capítulo sección ver véase ref referencia "
    "ibid op cit ed eds vol pp inc ltda sa sas".split()
)

_LEMMA_MAP: Dict[str, str] = {
    "enfermedades": "enfermedad",
    "síntomas": "síntoma",
    "sintomas": "síntoma",
    "manejos": "manejo",
    "controles": "control",
    "plagas": "plaga",
    "hojas": "hoja",
    "cultivos": "cultivo",
    "fertilizantes": "fertilizante",
    "diagnósticos": "diagnóstico",
    "diagnosticos": "diagnóstico",
    "lesiones": "lesión",
    "lesiones": "lesion",
    "manchas": "mancha",
    "pustulas": "pustula",
    "pústulas": "pustula",
}


def simple_lemmatize(text: str) -> str:
    """Lematización léxica ligera sin dependencias NLP externas."""
    tokens = re.findall(r"[a-záéíóúñü0-9]+|[^\w\s]", text.lower(), flags=re.UNICODE)
    out: List[str] = []
    for tok in tokens:
        if re.match(r"[^\w\s]", tok):
            out.append(tok)
        else:
            out.append(_LEMMA_MAP.get(tok, tok))
    return " ".join(out)


def expand_agronomy_terms(text: str) -> str:
    """Expande siglas y términos técnicos agronómicos en el texto."""
    result = text
    for pattern, expansion in AGRONOMY_TERM_EXPANSIONS.items():
        result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)
    return result


def enrich_text_for_indexing(text: str, apply_lemmatization: bool = False) -> str:
    """Pipeline de enriquecimiento previo a TF-IDF, BM25 o embeddings."""
    text = expand_agronomy_terms(text)
    if apply_lemmatization:
        text = simple_lemmatize(text)
    return re.sub(r"\s+", " ", text).strip()


def enrich_query(query: str) -> str:
    """Enriquecimiento de consultas: expansión de términos sin lematizar."""
    return enrich_text_for_indexing(query, apply_lemmatization=False)
