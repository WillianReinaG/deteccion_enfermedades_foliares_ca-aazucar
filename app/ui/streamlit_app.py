import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from PIL import Image
import pandas as pd

from app.classifier.predictor import SugarCanePredictor
from app.agent.agent import SugarCaneAgent
from app.memory.conversation_store import new_session_id, load_history, save_history, clear_history, list_sessions

st.set_page_config(page_title="SugarCane AI Agent", page_icon="🌱", layout="wide")

@st.cache_resource
def get_predictor():
    return SugarCanePredictor()

@st.cache_resource
def get_agent():
    return SugarCaneAgent()

predictor = get_predictor()
agent = get_agent()

if "session_id" not in st.session_state:
    st.session_state.session_id = new_session_id()
if "messages" not in st.session_state:
    st.session_state.messages = load_history(st.session_state.session_id)

st.title("🌱 Sistema inteligente para detección de enfermedades foliares en caña de azúcar")
st.caption("Clasificación por visión computacional + RAG avanzado + Agente IA conversacional con memoria local.")

with st.sidebar:
    st.header("Modelo cargado")
    st.write(f"**Modelo:** {predictor.model_id}")
    st.write(f"**Framework:** {predictor.framework}")
    st.write(f"**Checkpoint:** `{predictor.ckpt_path}`")
    if predictor.demo_mode:
        st.warning("Modo demo: no se encontró un peso entrenado. Copia el `best.pt` en `models/` o los artefactos en `artifacts/`.")

    st.divider()
    st.header("Memoria conversacional")
    st.caption(f"Sesión actual: `{st.session_state.session_id}`")
    if st.button("Nueva conversación"):
        st.session_state.session_id = new_session_id()
        st.session_state.messages = []
        st.rerun()
    if st.button("Guardar conversación"):
        save_history(st.session_state.session_id, st.session_state.messages)
        st.success("Histórico guardado en data/conversations/")
    if st.button("Borrar esta conversación"):
        clear_history(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()
    sessions = list_sessions()
    if sessions:
        selected_session = st.selectbox("Cargar conversación anterior", [""] + sessions)
        if selected_session and st.button("Cargar histórico seleccionado"):
            st.session_state.session_id = selected_session
            st.session_state.messages = load_history(selected_session)
            st.rerun()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Cargar imagen de hoja")
    uploaded = st.file_uploader("Sube una imagen JPG/PNG de hoja de caña", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Imagen cargada", use_container_width=True)
        if st.button("Clasificar hoja", type="primary"):
            pred = predictor.predict(image)
            st.session_state["prediction"] = pred
            agent.set_prediction(pred)
            resumen = f"Imagen clasificada como {pred['class_name']} con confianza {pred['confidence']:.2%}."
            st.session_state.messages.append({"role": "assistant", "content": resumen})
            save_history(st.session_state.session_id, st.session_state.messages)
            st.rerun()

with col2:
    st.subheader("2. Resultado de clasificación")
    pred = st.session_state.get("prediction")
    if pred:
        st.metric("Clase predicha", pred["class_name"], f"{pred['confidence']:.2%}")
        df = pd.DataFrame(pred["top_predictions"])
        df["confidence"] = df["confidence"].map(lambda x: f"{x:.2%}")
        st.dataframe(df, use_container_width=True, hide_index=True)
        if pred.get("demo_mode"):
            st.info("Este resultado es ilustrativo porque aún no hay checkpoint real configurado.")

        gradcam_img = pred.get("gradcam_image")
        if gradcam_img is not None:
            st.subheader("3. Explicabilidad visual (Grad-CAM)")
            st.caption(
                f"Mapa de calor sobre la región que **{pred.get('model_id', 'el modelo')}** "
                f"más influyó para predecir **{pred.get('gradcam_class', pred['class_name'])}**. "
                "Las zonas cálidas (rojo/amarillo) indican mayor peso en la decisión."
            )
            st.image(gradcam_img, caption=f"Grad-CAM — clase: {pred.get('gradcam_class')}", use_container_width=True)
        elif pred.get("gradcam_error"):
            st.caption(f"Grad-CAM no disponible: {pred['gradcam_error']}")
    else:
        st.info("Carga una imagen y presiona **Clasificar hoja**.")

st.divider()
st.subheader("4. Chat con Agente IA agronómico")
st.caption("El agente usa: resultado de imagen + Grad-CAM (si aplica) + RAG mejorado + histórico de la conversación.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

example = "Explícame la enfermedad detectada, sus síntomas, manejo y prevención."
question = st.chat_input(example)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Consultando base de conocimiento, histórico y resultado de imagen..."):
            answer = agent.answer(
                question,
                prediction=st.session_state.get("prediction"),
                history=st.session_state.messages,
                session_id=st.session_state.session_id,
            )
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_history(st.session_state.session_id, st.session_state.messages)
