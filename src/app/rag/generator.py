from __future__ import annotations

from typing import List, Dict, Optional
import re
import requests
from app.config.settings import OPENAI_API_KEY, OPENAI_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL

SYSTEM_PROMPT = """
Eres un agrónomo especializado en caña de azúcar (Saccharum spp.), con enfoque en fitopatología foliar,
manejo integrado de plagas y enfermedades (MIP/MID), y uso de visión por computador como herramienta de apoyo.

REGLAS OBLIGATORIAS:
1. Responde ÚNICAMENTE con base en la EVIDENCIA RAG proporcionada, el histórico de conversación y la
   clasificación de imagen cuando aplique.
2. Si la pregunta está fuera del ámbito agronómico de caña de azúcar o no hay evidencia suficiente en el
   contexto, responde de forma breve y profesional que no puedes responder con la información disponible e
   indica qué dato o documento haría falta.
3. No inventes fuentes, dosis, productos comerciales, normativas ni diagnósticos confirmados.
4. No presentes la predicción por imagen como diagnóstico definitivo; indícalo siempre como inferencia de
   apoyo sujeta a verificación en campo.
5. No mezcles síntomas o manejos de enfermedades distintas salvo para diferenciar diagnósticos, y solo si
   la evidencia lo respalda.

ESTILO DE RESPUESTA:
- Español técnico-profesional, claro y específico, como informe de extensión agrícola.
- Evita generalidades vacías y lenguaje coloquial.
- Prioriza hechos verificables del contexto: síntomas, condiciones favorables, dispersión y manejo según lo documentado.
- Si falta un dato crítico (variedad, fenología, clima, zona, severidad), solicita solo lo mínimo necesario.

Cuando la evidencia lo permita, estructura la respuesta así:
1. Respuesta directa a la pregunta.
2. Relación con la clasificación de imagen (si existe).
3. Síntomas y criterios técnicos relevantes.
4. Manejo recomendado según el contexto recuperado.
5. Limitaciones del análisis y recomendación de validación en campo por agrónomo.
""".strip()

NO_RAG_SYSTEM_PROMPT = """
Eres un agrónomo especializado en caña de azúcar. Este modo se usa únicamente para evaluación experimental
de ablación SIN recuperación documental. Responde con tu conocimiento general, sin citar evidencia RAG.
Si no estás seguro, advierte la limitación. No inventes dosis, productos comerciales ni normativas.
""".strip()

RESPONSE_INSTRUCTIONS = """
INSTRUCCIONES DE RESPUESTA:
1. Responde directamente a la pregunta, sin plantillas genéricas.
2. Cita la enfermedad predicha solo si es coherente con la evidencia recuperada.
3. Estructura: respuesta directa → relación con la imagen → explicación técnica breve → manejo en campo → límites del diagnóstico.
4. Si la evidencia es insuficiente, indícalo y sugiere qué revisar en campo.
5. No inventes fuentes, dosis exactas ni tratamientos no respaldados por el contexto.
6. Si la pregunta no es sobre caña de azúcar, enfermedades foliares, manejo agronómico o el diagnóstico asistido del sistema, recházala educadamente.
7. No respondas temas ajenos (política, medicina humana, otros cultivos sin evidencia en el contexto, etc.).
""".strip()

NO_RAG_RESPONSE_INSTRUCTIONS = """
INSTRUCCIONES DE RESPUESTA PARA MODO SIN RAG:
1. Responde usando conocimiento general, sin usar documentos recuperados.
2. No cites fuentes ni fragmentos documentales.
3. Mantén una advertencia breve indicando que la respuesta no está sustentada por recuperación documental.
4. No inventes dosis, productos comerciales o diagnósticos confirmados.
""".strip()


def get_active_llm_mode() -> str:
    """Indica el motor de respuesta configurado (sin verificar conectividad)."""
    if OPENAI_API_KEY:
        return f"openai ({OPENAI_MODEL})"
    return "local"


def get_active_llm_label() -> str:
    """Etiqueta legible para la UI."""
    mode = get_active_llm_mode()
    if mode.startswith("openai"):
        return f"OpenAI ({OPENAI_MODEL})"
    return "RAG extractivo local (configure OPENAI_API_KEY en .env)"


def build_context(chunks: List[Dict]) -> str:
    if not chunks:
        return (
            "SIN EVIDENCIA RAG DISPONIBLE. No se recuperaron fragmentos documentales para esta consulta. "
            "Debes indicar al usuario que no cuentas con información suficiente en la base de conocimiento "
            "para responder. No uses conocimiento general ni completes con suposiciones."
        )
    blocks = []
    for c in chunks:
        title = c.get("title", "")
        diseases = ", ".join(c.get("diseases", [])) or "general"
        blocks.append(
            f"[Fuente: {c['source']} | sección: {title} | enfermedades: {diseases} | "
            f"relevancia: {c.get('final_score', c.get('score', 0)):.3f}]\n{c['text']}"
        )
    return "\n\n".join(blocks)


def build_history(history: List[Dict] | None, max_turns: int = 8) -> str:
    if not history:
        return "Sin histórico previo."
    recent = history[-max_turns:]
    return "\n".join(f"{m.get('role','user')}: {m.get('content','')[:600]}" for m in recent)


def build_prediction_text(prediction: Optional[Dict]) -> str:
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
    return pred_txt


def build_user_prompt(
    question: str,
    context: str,
    hist: str,
    pred_txt: str,
) -> str:
    return f"""
HISTÓRICO RECIENTE:
{hist}

CLASIFICACIÓN ACTUAL (visión por computador):
{pred_txt}

EVIDENCIA RAG (usa solo esto como base factual):
{context}

PREGUNTA DEL USUARIO:
{question}

{RESPONSE_INSTRUCTIONS}
""".strip()


def build_no_rag_prompt(
    question: str,
    hist: str,
    pred_txt: str,
) -> str:
    return f"""
HISTÓRICO RECIENTE:
{hist}

CLASIFICACIÓN ACTUAL (visión por computador):
{pred_txt}

MODO EXPERIMENTAL SIN RAG:
No se proporcionan fragmentos documentales recuperados. Esta respuesta se usará únicamente para comparar
un modelo generativo sin recuperación contra el sistema RAG.

PREGUNTA DEL USUARIO:
{question}

{NO_RAG_RESPONSE_INSTRUCTIONS}
""".strip()


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
    def generate(
        self,
        question: str,
        chunks: List[Dict],
        prediction: Optional[Dict] = None,
        history: List[Dict] | None = None,
    ) -> str:
        """Genera respuesta CON RAG usando fragmentos recuperados."""
        context = build_context(chunks)
        hist = build_history(history)
        pred_txt = build_prediction_text(prediction)
        user_prompt = build_user_prompt(question, context, hist, pred_txt)

        openai = self._try_openai(user_prompt, system_prompt=SYSTEM_PROMPT)
        if openai:
            return openai
        ollama = self._try_ollama(user_prompt, system_prompt=SYSTEM_PROMPT)
        if ollama:
            return ollama
        return self._fallback(question, chunks, prediction)

    def generate_without_rag(
        self,
        question: str,
        prediction: Optional[Dict] = None,
        history: List[Dict] | None = None,
    ) -> str:
        """Genera respuesta SIN RAG para experimento de ablación.

        Se usa para comparar el aporte de la recuperación documental frente al LLM sin contexto.
        """
        hist = build_history(history)
        pred_txt = build_prediction_text(prediction)
        user_prompt = build_no_rag_prompt(question, hist, pred_txt)

        openai = self._try_openai(user_prompt, system_prompt=NO_RAG_SYSTEM_PROMPT)
        if openai:
            return openai
        ollama = self._try_ollama(user_prompt, system_prompt=NO_RAG_SYSTEM_PROMPT)
        if ollama:
            return ollama
        return self._fallback_no_rag(question, prediction)

    def _try_openai(self, user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> Optional[str]:
        if not OPENAI_API_KEY:
            return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            r = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.15,
            )
            return r.choices[0].message.content
        except Exception:
            return None

    def _try_ollama(self, user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> Optional[str]:
        if OPENAI_API_KEY:
            return None
        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            r = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False},
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

        intro = (
            "**Respuesta basada en la base de conocimiento local (sin LLM generativo)**\n\n"
            "Esta respuesta se limita estrictamente a los fragmentos recuperados de la base documental. "
            "No sustituye una inspección de campo.\n\n"
        )
        if disease:
            intro += (
                f"La imagen fue clasificada como **{disease}** ({conf:.1%} de confianza). "
                "Esto es una inferencia de apoyo por visión computacional, no un diagnóstico confirmado.\n\n"
            )

        if sentences:
            body = "**Información documental relevante a su consulta:**\n"
            for s in sentences:
                body += f"- {s}\n"
        else:
            body = (
                "No dispongo de evidencia documental suficiente en la base de conocimiento para responder "
                "de forma específica a esta consulta. Le sugiero reformular la pregunta indicando la enfermedad "
                "o síntoma de interés, o ampliar los documentos en `src/app/knowledge_base/`.\n"
            )

        footer = (
            "\n**Recomendación profesional:** valide síntomas en varias hojas y sectores del lote. "
            "Si la afectación progresa, consulte a un agrónomo o laboratorio fitosanitario."
        )
        return (intro + body + footer).strip()

    def _fallback_no_rag(self, question: str, prediction: Optional[Dict]) -> str:
        disease = prediction.get("class_name") if prediction else None
        conf = prediction.get("confidence", 0) if prediction else 0
        intro = (
            "**Respuesta experimental sin RAG**\n\n"
            "Esta respuesta no utiliza fragmentos documentales recuperados; se genera solo para comparar "
            "el comportamiento del sistema con y sin recuperación aumentada.\n\n"
        )
        if disease:
            intro += (
                f"La imagen fue clasificada como **{disease}** ({conf:.1%} de confianza). "
                "Esta inferencia debe validarse en campo.\n\n"
            )
        return (
            intro
            + "No se dispone de evidencia documental recuperada en este modo. Para una recomendación técnica "
            + "sustentada, use el modo RAG con base de conocimiento agronómica."
        ).strip()
