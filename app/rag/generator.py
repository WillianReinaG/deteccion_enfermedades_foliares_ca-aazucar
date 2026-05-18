from __future__ import annotations
from typing import List, Dict, Optional
import re
import requests
from app.config.settings import OPENAI_API_KEY, OPENAI_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL

SYSTEM_PROMPT = """
Eres un Agente IA agronómico especializado en caña de azúcar, visión por computador y manejo integrado de enfermedades foliares.
Responde en español, con tono profesional, claro y útil para productores, estudiantes y técnicos.
Usa ÚNICAMENTE la base de conocimiento recuperada por RAG y el histórico de conversación. Si la evidencia no cubre la pregunta, dilo explícitamente.
No presentes la predicción de imagen como diagnóstico definitivo; es una inferencia de apoyo que requiere verificación en campo.
Prioriza fragmentos con mayor score y alineados con la enfermedad predicha. No mezcles síntomas de otras enfermedades salvo para descartar diferencias.
Cuando falte información, pide el dato mínimo necesario.
""".strip()


def build_context(chunks: List[Dict]) -> str:
    if not chunks:
        return "No se recuperaron documentos de soporte. Responde con conocimiento general y prudencia."
    blocks = []
    for c in chunks:
        title = c.get("title", "")
        diseases = ", ".join(c.get("diseases", [])) or "general"
        blocks.append(
            f"[Fuente: {c['source']} | sección: {title} | enfermedades: {diseases} | relevancia: {c.get('final_score', c.get('score', 0)):.3f}]\n{c['text']}"
        )
    return "\n\n".join(blocks)


def build_history(history: List[Dict] | None, max_turns: int = 8) -> str:
    if not history:
        return "Sin histórico previo."
    recent = history[-max_turns:]
    return "\n".join(f"{m.get('role','user')}: {m.get('content','')[:600]}" for m in recent)


def _extract_relevant_sentences(question: str, chunks: List[Dict], max_sentences: int = 6) -> List[str]:
    q_tokens = set(re.findall(r"[a-záéíóúñü0-9]+", question.lower()))
    q_tokens -= {"que", "como", "cual", "para", "con", "del", "las", "los", "una", "por", "son", "esta", "este"}
    scored: List[tuple[float, str, str]] = []
    for c in chunks:
        for sent in re.split(r"(?<=[.!?])\s+", c["text"]):
            sent = sent.strip()
            if len(sent) < 40:
                continue
            stokens = set(re.findall(r"[a-záéíóúñü0-9]+", sent.lower()))
            overlap = len(q_tokens & stokens) if q_tokens else 0
            score = overlap + c.get("final_score", c.get("score", 0)) * 2
            if score > 0 or len(sent) > 80:
                scored.append((score, c["source"], sent))
    scored.sort(key=lambda x: x[0], reverse=True)
    seen, out = set(), []
    for _, src, sent in scored:
        key = sent[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(f"({src}) {sent}")
        if len(out) >= max_sentences:
            break
    return out


class AnswerGenerator:
    def generate(self, question: str, chunks: List[Dict], prediction: Optional[Dict] = None, history: List[Dict] | None = None) -> str:
        context = build_context(chunks)
        hist = build_history(history)
        pred_txt = "No hay imagen clasificada en esta sesión."
        if prediction:
            gradcam_note = ""
            if prediction.get("gradcam_image") is not None:
                gradcam_note = (
                    f"\nExplicabilidad Grad-CAM disponible para clase {prediction.get('gradcam_class')}: "
                    "el modelo concentró atención en las regiones resaltadas en rojo/amarillo de la imagen."
                )
            pred_txt = (
                f"Clase predicha por imagen: {prediction.get('class_name')}\n"
                f"Confianza: {prediction.get('confidence', 0):.2%}\n"
                f"Modelo: {prediction.get('model_id')} | Framework: {prediction.get('framework')}\n"
                f"Top predicciones: {prediction.get('top_predictions')}"
                f"{gradcam_note}"
            )
        prompt = f"""
{SYSTEM_PROMPT}

HISTÓRICO RECIENTE:
{hist}

CLASIFICACIÓN ACTUAL (visión por computador):
{pred_txt}

EVIDENCIA RAG (usa solo esto como base factual):
{context}

PREGUNTA DEL USUARIO:
{question}

INSTRUCCIONES DE RESPUESTA:
1. Responde directamente a la pregunta, sin plantillas genéricas.
2. Cita la enfermedad predicha solo si es coherente con la evidencia recuperada.
3. Estructura: respuesta directa → relación con la imagen → explicación técnica breve → manejo en campo → límites del diagnóstico.
4. Si la evidencia es insuficiente, indícalo y sugiere qué revisar en campo.
5. No inventes fuentes, dosis exactas ni tratamientos no respaldados por el contexto.
""".strip()
        openai = self._try_openai(prompt)
        if openai:
            return openai
        ollama = self._try_ollama(prompt)
        if ollama:
            return ollama
        return self._fallback(question, chunks, prediction)

    def _try_openai(self, prompt: str) -> Optional[str]:
        if not OPENAI_API_KEY:
            return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            r = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                temperature=0.15,
            )
            return r.choices[0].message.content
        except Exception:
            return None

    def _try_ollama(self, prompt: str) -> Optional[str]:
        try:
            r = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=90,
            )
            if r.ok:
                return r.json().get("response")
        except Exception:
            return None
        return None

    def _fallback(self, question: str, chunks: List[Dict], prediction: Optional[Dict]) -> str:
        disease = prediction.get("class_name") if prediction else None
        conf = prediction.get("confidence", 0) if prediction else 0
        sentences = _extract_relevant_sentences(question, chunks)
        if not sentences and chunks:
            sentences = [f"({c['source']}) {c['text'][:400]}" for c in chunks[:3]]

        intro = "**Respuesta basada en la base de conocimiento local**\n\n"
        if disease:
            intro += (
                f"La imagen fue clasificada como **{disease}** ({conf:.1%} de confianza). "
                "Esto es apoyo de IA, no un diagnóstico de campo.\n\n"
            )

        if sentences:
            body = "**Información recuperada relevante a su consulta:**\n"
            for s in sentences:
                body += f"- {s}\n"
        else:
            body = (
                "No encontré fragmentos suficientemente relevantes en la base local para esta pregunta. "
                "Agregue documentos en `app/knowledge_base/` o reformule la consulta con la enfermedad específica.\n"
            )

        footer = (
            "\n**Recomendación:** valide síntomas en varias hojas y sectores del lote. "
            "Si la afectación progresa, consulte a un agrónomo o laboratorio fitosanitario."
        )
        return (intro + body + footer).strip()
