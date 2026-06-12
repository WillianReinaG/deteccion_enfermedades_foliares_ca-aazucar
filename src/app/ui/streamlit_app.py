import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from datetime import datetime

from app.classifier.predictor import SugarCanePredictor
from app.agent.agent import SugarCaneAgent
from app.memory.conversation_store import new_session_id, load_history, save_history, clear_history, list_sessions
from app.rag.generator import get_active_llm_label
from app.services.alert_service import alerts_configured, is_disease, send_immediate_alert
from app.services.prediction_logger import log_prediction


def _safe_name(value: str) -> str:
    """Normaliza textos para usarlos en nombres de archivo."""
    value = str(value or "sin_clase").strip().replace(" ", "_")
    return "".join(ch for ch in value if ch.isalnum() or ch in ("_", "-"))


def _estimate_leaf_mask(image: Image.Image, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Estima una máscara binaria simple de hoja/fondo sin anotación manual.

    Nota metodológica:
    Esta máscara NO reemplaza una máscara experta. Se usa únicamente para estimar
    qué proporción de energía Grad-CAM cae sobre la región visible de la hoja.
    """
    h, w = target_shape
    img = image.convert("RGB").resize((w, h))
    arr = np.asarray(img).astype(np.float32) / 255.0

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    saturation = maxc - minc
    brightness = maxc

    # Heurística general para hojas: evita fondo blanco/negro y privilegia píxeles con color.
    green_dominance = (g >= r * 0.75) & (g >= b * 0.75)
    colored_region = saturation > 0.08
    valid_brightness = (brightness > 0.08) & (brightness < 0.98)

    mask = (colored_region & valid_brightness & green_dominance)

    # Respaldo: si la máscara queda demasiado pequeña, usar región coloreada general.
    if mask.mean() < 0.03:
        mask = colored_region & valid_brightness

    return mask.astype(bool)


def save_gradcam_report(
    image: Image.Image,
    pred: dict,
    session_id: str,
    uploaded_name: str | None = None,
    output_root: str = "artifacts/gradcam_eval",
    threshold: float = 0.50,
) -> dict:
    """
    Guarda evidencia Grad-CAM y métricas semi-cuantitativas por imagen clasificada.

    Archivos generados:
    - Imagen original.
    - Imagen Grad-CAM superpuesta.
    - Matriz CAM cruda (.npy).
    - Máscara estimada de hoja.
    - CSV acumulado con métricas por clasificación.
    """
    output_dir = Path(output_root)
    images_dir = output_dir / "images"
    cams_dir = output_dir / "cam_arrays"
    masks_dir = output_dir / "leaf_masks"

    images_dir.mkdir(parents=True, exist_ok=True)
    cams_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    class_name = str(pred.get("class_name", "sin_clase"))
    class_safe = _safe_name(class_name)
    session_safe = _safe_name(session_id)
    stem = f"{timestamp}_{session_safe}_{class_safe}"

    original_path = images_dir / f"{stem}_original.jpg"
    gradcam_path = images_dir / f"{stem}_gradcam.jpg"
    cam_path = cams_dir / f"{stem}_cam.npy"
    leaf_mask_path = masks_dir / f"{stem}_leaf_mask.png"

    image.save(original_path)

    gradcam_img = pred.get("gradcam_image")
    if gradcam_img is not None:
        gradcam_img.save(gradcam_path)
    else:
        gradcam_path = None

    cam = pred.get("gradcam_array")
    metrics = {
        "timestamp": timestamp,
        "session_id": session_id,
        "uploaded_name": uploaded_name or "",
        "class_name": class_name,
        "model_id": pred.get("model_id", ""),
        "framework": pred.get("framework", ""),
        "confidence": float(pred.get("confidence", 0.0)),
        "original_image_path": str(original_path),
        "gradcam_image_path": str(gradcam_path) if gradcam_path else "",
        "cam_array_path": "",
        "leaf_mask_path": "",
        "mean_cam": "",
        "max_cam": "",
        "std_cam": "",
        "active_pixels": "",
        "active_pixels_pct": "",
        "cam_energy_inside_leaf": "",
        "active_pixels_inside_leaf_pct": "",
        "iou_with_manual_mask": "",
        "notes": "",
    }

    if cam is None:
        metrics["notes"] = "Grad-CAM no disponible o no retornó matriz CAM."
    else:
        cam = np.asarray(cam, dtype=np.float32)
        cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
        cam = np.clip(cam, 0.0, 1.0)

        np.save(cam_path, cam)
        metrics["cam_array_path"] = str(cam_path)

        active = cam >= threshold
        total_pixels = int(cam.size)
        active_pixels = int(active.sum())

        leaf_mask = _estimate_leaf_mask(image, cam.shape)
        leaf_mask_img = Image.fromarray((leaf_mask.astype(np.uint8) * 255))
        leaf_mask_img.save(leaf_mask_path)
        metrics["leaf_mask_path"] = str(leaf_mask_path)

        cam_energy_total = float(cam.sum())
        if cam_energy_total > 0:
            cam_energy_inside_leaf = float((cam * leaf_mask).sum() / cam_energy_total)
        else:
            cam_energy_inside_leaf = 0.0

        if active_pixels > 0:
            active_pixels_inside_leaf_pct = float((active & leaf_mask).sum() / active_pixels)
        else:
            active_pixels_inside_leaf_pct = 0.0

        metrics.update({
            "mean_cam": float(cam.mean()),
            "max_cam": float(cam.max()),
            "std_cam": float(cam.std()),
            "active_pixels": active_pixels,
            "active_pixels_pct": float(active_pixels / total_pixels),
            "cam_energy_inside_leaf": cam_energy_inside_leaf,
            "active_pixels_inside_leaf_pct": active_pixels_inside_leaf_pct,
            "iou_with_manual_mask": "",  # Requiere máscara manual experta.
            "notes": "IoU no calculado: no se proporcionó máscara manual de lesión.",
        })

    metrics_file = output_dir / "gradcam_metrics_ui.csv"
    row = pd.DataFrame([metrics])
    if metrics_file.exists():
        row.to_csv(metrics_file, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        row.to_csv(metrics_file, index=False, encoding="utf-8-sig")

    # Resumen por clase actualizado.
    try:
        df_all = pd.read_csv(metrics_file)
        numeric_cols = [
            "confidence",
            "mean_cam",
            "max_cam",
            "std_cam",
            "active_pixels",
            "active_pixels_pct",
            "cam_energy_inside_leaf",
            "active_pixels_inside_leaf_pct",
        ]
        for col in numeric_cols:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

        by_class = (
            df_all.groupby("class_name", dropna=False)[numeric_cols]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        by_class.to_csv(output_dir / "gradcam_metrics_by_class_ui.csv", index=False, encoding="utf-8-sig")
    except Exception:
        pass

    return metrics


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
    st.header("Motor de respuesta")
    st.info(get_active_llm_label())

    st.divider()
    st.header("Alertas por correo")
    if alerts_configured():
        st.success("Correo SMTP configurado — alertas activas")
    else:
        st.warning("Configure SMTP_USER, SMTP_APP_PASSWORD y ALERT_EMAIL en .env para alertas.")

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

            # Guardar evidencia Grad-CAM y métricas semi-cuantitativas de la clasificación.
            gradcam_metrics = save_gradcam_report(
                image=image,
                pred=pred,
                session_id=st.session_state.session_id,
                uploaded_name=getattr(uploaded, "name", None),
            )
            pred["gradcam_metrics"] = gradcam_metrics

            log_prediction(pred, session_id=st.session_state.session_id)
            if is_disease(pred):
                sent = send_immediate_alert(pred, session_id=st.session_state.session_id)
                if sent:
                    st.toast("Alerta enviada por correo", icon="📧")
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

            metrics = pred.get("gradcam_metrics")
            if metrics:
                st.subheader("4. Métricas semi-cuantitativas Grad-CAM")
                c1, c2, c3 = st.columns(3)
                c1.metric("Intensidad media CAM", f"{float(metrics.get('mean_cam') or 0):.4f}")
                c2.metric("Píxeles activados", f"{int(metrics.get('active_pixels') or 0):,}")
                c3.metric("Energía CAM en hoja", f"{float(metrics.get('cam_energy_inside_leaf') or 0):.2%}")

                st.caption(
                    "El IoU requiere máscaras manuales de lesión. "
                    "Si no existen máscaras anotadas, se reporta como no calculado."
                )
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
