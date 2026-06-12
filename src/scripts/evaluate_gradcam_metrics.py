from __future__ import annotations

"""
Evaluación semi-cuantitativa de Grad-CAM para SugarCane AI Agent.

Genera métricas por imagen y reportes agregados por clase:
- Porcentaje de energía CAM dentro de la hoja.
- Intensidad CAM promedio.
- Número y porcentaje de píxeles activados.
- Confianza de la predicción.
- IoU con máscara, si se suministra carpeta de máscaras.
- Exportación de overlays Grad-CAM y mapas CAM, opcional.

Uso recomendado desde la raíz del proyecto:
python src/scripts/evaluate_gradcam_metrics.py \
  --images_dir data/test \
  --output_dir artifacts/gradcam_eval \
  --cam_threshold 0.50 \
  --save_overlays

Con máscaras manuales opcionales:
python src/scripts/evaluate_gradcam_metrics.py \
  --images_dir data/test \
  --masks_dir data/masks_test \
  --output_dir artifacts/gradcam_eval \
  --cam_threshold 0.50 \
  --save_overlays
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

try:
    import cv2
except ImportError as exc:
    raise RuntimeError("Falta opencv-python-headless. Instale: pip install opencv-python-headless") from exc

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError as exc:
    raise RuntimeError("Falta grad-cam. Instale: pip install grad-cam opencv-python-headless") from exc

# Permite ejecutar el script desde la raíz del proyecto o desde src/scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for p in [PROJECT_ROOT, SRC_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.classifier.predictor import SugarCanePredictor, EVAL_TF  # noqa: E402
from app.classifier.gradcam import gradcam_supported, _get_target_layers, _tensor_to_rgb_uint8  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(images_dir: Path) -> List[Path]:
    return sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def infer_true_class(image_path: Path, images_dir: Path) -> str:
    """Si el dataset está en formato ImageFolder, usa el nombre de la carpeta como clase real."""
    try:
        rel = image_path.relative_to(images_dir)
        return rel.parts[0] if len(rel.parts) > 1 else "unknown"
    except Exception:
        return "unknown"


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    cam_min, cam_max = float(cam.min()), float(cam.max())
    if cam_max - cam_min < 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    return (cam - cam_min) / (cam_max - cam_min)


def estimate_leaf_mask(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Máscara automática aproximada de hoja/fondo.
    No reemplaza máscaras manuales, pero permite estimar energía CAM sobre la hoja.
    Usa saturación, intensidad y componente verde en HSV/RGB.
    """
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    r, g, b = rgb_uint8[..., 0], rgb_uint8[..., 1], rgb_uint8[..., 2]

    # Detecta regiones vegetales o de hoja: verdes, amarillas, cafés/rojizas con saturación suficiente.
    greenish = (g.astype(np.int16) >= r.astype(np.int16) - 15) & (g.astype(np.int16) >= b.astype(np.int16) - 15)
    saturated = s > 35
    not_too_dark = v > 35
    not_white_bg = ~((r > 235) & (g > 235) & (b > 235))

    mask = (saturated & not_too_dark & not_white_bg) | (greenish & not_too_dark & not_white_bg)
    mask = mask.astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Conserva el componente conectado más grande para reducir ruido de fondo.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)

    return mask.astype(bool)


def load_manual_mask(image_path: Path, masks_dir: Optional[Path], target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    if masks_dir is None:
        return None

    candidates = []
    stem = image_path.stem
    for ext in IMAGE_EXTS:
        candidates.append(masks_dir / f"{stem}{ext}")
        candidates.append(masks_dir / image_path.parent.name / f"{stem}{ext}")

    mask_path = next((p for p in candidates if p.exists()), None)
    if mask_path is None:
        return None

    mask = Image.open(mask_path).convert("L").resize(target_size, Image.Resampling.NEAREST)
    mask_np = np.asarray(mask)
    return mask_np > 0


def compute_iou(binary_cam: np.ndarray, mask: Optional[np.ndarray]) -> Optional[float]:
    if mask is None:
        return None
    binary_cam = binary_cam.astype(bool)
    mask = mask.astype(bool)
    intersection = np.logical_and(binary_cam, mask).sum()
    union = np.logical_or(binary_cam, mask).sum()
    if union == 0:
        return None
    return float(intersection / union)


def generate_numeric_gradcam(
    predictor: SugarCanePredictor,
    image: Image.Image,
    class_index: int,
) -> Tuple[np.ndarray, np.ndarray, Image.Image]:
    """
    Retorna: cam normalizado [0,1], imagen RGB uint8, overlay PIL.
    """
    if predictor.model is None or predictor.demo_mode:
        raise RuntimeError("No hay modelo entrenado cargado. Revise checkpoint/modelos.")
    if not gradcam_supported(predictor.model_id, predictor.framework):
        raise RuntimeError(f"Grad-CAM no soportado para {predictor.model_id} / {predictor.framework}")

    target_layers = _get_target_layers(predictor.model, predictor.model_id)
    if not target_layers:
        raise RuntimeError(f"No se encontró capa objetivo Grad-CAM para {predictor.model_id}")

    input_tensor = EVAL_TF(image.convert("RGB")).unsqueeze(0).to(predictor.device)
    rgb_uint8 = _tensor_to_rgb_uint8(input_tensor)
    float_img = rgb_uint8.astype(np.float32) / 255.0

    cam_obj = None
    try:
        cam_obj = GradCAM(model=predictor.model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(class_index)]
        grayscale_cam = cam_obj(input_tensor=input_tensor, targets=targets)[0]
        grayscale_cam = normalize_cam(grayscale_cam)
        overlay = show_cam_on_image(float_img, grayscale_cam, use_rgb=True)
        return grayscale_cam, rgb_uint8, Image.fromarray(overlay)
    finally:
        if cam_obj is not None:
            cam_obj.__del__()


def evaluate_image(
    predictor: SugarCanePredictor,
    image_path: Path,
    images_dir: Path,
    masks_dir: Optional[Path],
    output_dir: Path,
    cam_threshold: float,
    save_overlays: bool,
) -> Dict:
    image = Image.open(image_path).convert("RGB")
    pred = predictor.predict(image, with_gradcam=False)
    pred_class = pred["class_name"]
    confidence = float(pred["confidence"])

    # Índice de la clase predicha según class_names del predictor.
    class_index = predictor.class_names.index(pred_class) if pred_class in predictor.class_names else 0

    cam, rgb_uint8, overlay = generate_numeric_gradcam(predictor, image, class_index)
    h, w = cam.shape
    leaf_mask = estimate_leaf_mask(rgb_uint8)
    manual_mask = load_manual_mask(image_path, masks_dir, target_size=(w, h))

    active = cam >= cam_threshold
    total_pixels = int(cam.size)
    active_pixels = int(active.sum())
    leaf_pixels = int(leaf_mask.sum())

    cam_sum = float(cam.sum())
    energy_inside_leaf = float((cam * leaf_mask).sum() / cam_sum) if cam_sum > 0 else 0.0
    active_inside_leaf = float(np.logical_and(active, leaf_mask).sum() / max(active_pixels, 1))
    iou_manual_mask = compute_iou(active, manual_mask)

    if save_overlays:
        overlay_dir = output_dir / "overlays" / pred_class
        cam_dir = output_dir / "cam_arrays" / pred_class
        mask_dir = output_dir / "leaf_masks" / pred_class
        overlay_dir.mkdir(parents=True, exist_ok=True)
        cam_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        safe_name = image_path.stem
        overlay.save(overlay_dir / f"{safe_name}_gradcam.png")
        np.save(cam_dir / f"{safe_name}_cam.npy", cam)
        Image.fromarray((leaf_mask.astype(np.uint8) * 255)).save(mask_dir / f"{safe_name}_leaf_mask.png")

    return {
        "image_path": str(image_path),
        "file_name": image_path.name,
        "true_class_folder": infer_true_class(image_path, images_dir),
        "predicted_class": pred_class,
        "confidence": confidence,
        "model_id": pred.get("model_id"),
        "framework": pred.get("framework"),
        "mean_cam": float(cam.mean()),
        "max_cam": float(cam.max()),
        "std_cam": float(cam.std()),
        "cam_threshold": cam_threshold,
        "active_pixels": active_pixels,
        "active_pixels_pct": float(active_pixels / total_pixels),
        "leaf_pixels": leaf_pixels,
        "leaf_pixels_pct": float(leaf_pixels / total_pixels),
        "cam_energy_inside_leaf": energy_inside_leaf,
        "active_pixels_inside_leaf_pct": active_inside_leaf,
        "iou_with_manual_mask": iou_manual_mask,
        "has_manual_mask": manual_mask is not None,
    }


def build_reports(df: pd.DataFrame, output_dir: Path) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "gradcam_metrics_per_image.csv", index=False, encoding="utf-8-sig")

    agg = (
        df.groupby("predicted_class")
        .agg(
            n_images=("file_name", "count"),
            confidence_mean=("confidence", "mean"),
            confidence_std=("confidence", "std"),
            mean_cam_mean=("mean_cam", "mean"),
            mean_cam_std=("mean_cam", "std"),
            active_pixels_mean=("active_pixels", "mean"),
            active_pixels_pct_mean=("active_pixels_pct", "mean"),
            cam_energy_inside_leaf_mean=("cam_energy_inside_leaf", "mean"),
            cam_energy_inside_leaf_std=("cam_energy_inside_leaf", "std"),
            active_pixels_inside_leaf_pct_mean=("active_pixels_inside_leaf_pct", "mean"),
            iou_with_manual_mask_mean=("iou_with_manual_mask", "mean"),
            manual_masks_available=("has_manual_mask", "sum"),
        )
        .reset_index()
    )
    agg.to_csv(output_dir / "gradcam_metrics_by_class.csv", index=False, encoding="utf-8-sig")

    summary = {
        "n_images": int(len(df)),
        "classes_detected": sorted(df["predicted_class"].dropna().unique().tolist()),
        "confidence_mean": float(df["confidence"].mean()),
        "mean_cam_global": float(df["mean_cam"].mean()),
        "active_pixels_mean": float(df["active_pixels"].mean()),
        "active_pixels_pct_mean": float(df["active_pixels_pct"].mean()),
        "cam_energy_inside_leaf_mean": float(df["cam_energy_inside_leaf"].mean()),
        "active_pixels_inside_leaf_pct_mean": float(df["active_pixels_inside_leaf_pct"].mean()),
        "manual_masks_available": int(df["has_manual_mask"].sum()),
        "iou_with_manual_mask_mean": None
        if df["iou_with_manual_mask"].dropna().empty
        else float(df["iou_with_manual_mask"].dropna().mean()),
    }

    with open(output_dir / "gradcam_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Reporte legible en TXT para pegar en tesis.
    with open(output_dir / "gradcam_metrics_summary.txt", "w", encoding="utf-8") as f:
        f.write("SugarCane — Evaluación semi-cuantitativa Grad-CAM\n")
        f.write("=================================================\n\n")
        f.write(f"Imágenes evaluadas: {summary['n_images']}\n")
        f.write(f"Clases detectadas: {', '.join(summary['classes_detected'])}\n")
        f.write(f"Confianza promedio: {summary['confidence_mean']:.4f}\n")
        f.write(f"Intensidad CAM promedio: {summary['mean_cam_global']:.4f}\n")
        f.write(f"Píxeles activados promedio: {summary['active_pixels_mean']:.2f}\n")
        f.write(f"Porcentaje promedio de píxeles activados: {summary['active_pixels_pct_mean']:.4f}\n")
        f.write(f"Energía CAM promedio dentro de la hoja: {summary['cam_energy_inside_leaf_mean']:.4f}\n")
        f.write(f"Píxeles activados dentro de la hoja: {summary['active_pixels_inside_leaf_pct_mean']:.4f}\n")
        f.write(f"Máscaras manuales disponibles: {summary['manual_masks_available']}\n")
        if summary["iou_with_manual_mask_mean"] is None:
            f.write("IoU promedio con máscaras manuales: No calculado (no hay máscaras disponibles).\n")
        else:
            f.write(f"IoU promedio con máscaras manuales: {summary['iou_with_manual_mask_mean']:.4f}\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación semi-cuantitativa de Grad-CAM")
    parser.add_argument("--images_dir", type=str, required=True, help="Carpeta de imágenes, idealmente formato ImageFolder")
    parser.add_argument("--masks_dir", type=str, default=None, help="Carpeta opcional de máscaras binarias manuales")
    parser.add_argument("--output_dir", type=str, default="artifacts/gradcam_eval", help="Carpeta de salida")
    parser.add_argument("--cam_threshold", type=float, default=0.50, help="Umbral para considerar píxel CAM activado")
    parser.add_argument("--save_overlays", action="store_true", help="Guardar overlays Grad-CAM, CAM .npy y máscaras de hoja estimadas")
    parser.add_argument("--limit", type=int, default=None, help="Limitar número de imágenes para prueba rápida")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir) if args.masks_dir else None
    output_dir = Path(args.output_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"No existe images_dir: {images_dir}")
    if masks_dir is not None and not masks_dir.exists():
        raise FileNotFoundError(f"No existe masks_dir: {masks_dir}")

    images = list_images(images_dir)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise RuntimeError(f"No se encontraron imágenes en {images_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = SugarCanePredictor()

    if predictor.demo_mode:
        raise RuntimeError("El predictor está en modo demo. Copie el checkpoint real antes de evaluar Grad-CAM.")
    if not gradcam_supported(predictor.model_id, predictor.framework):
        raise RuntimeError(f"Modelo no compatible con Grad-CAM: {predictor.model_id} / {predictor.framework}")

    rows: List[Dict] = []
    errors: List[Dict] = []

    print(f"Modelo: {predictor.model_id} | Framework: {predictor.framework}")
    print(f"Imágenes a evaluar: {len(images)}")
    print(f"Salida: {output_dir}")

    for idx, image_path in enumerate(images, start=1):
        try:
            row = evaluate_image(
                predictor=predictor,
                image_path=image_path,
                images_dir=images_dir,
                masks_dir=masks_dir,
                output_dir=output_dir,
                cam_threshold=args.cam_threshold,
                save_overlays=args.save_overlays,
            )
            rows.append(row)
            if idx % 25 == 0 or idx == len(images):
                print(f"Procesadas {idx}/{len(images)} imágenes")
        except Exception as exc:
            errors.append({"image_path": str(image_path), "error": str(exc)})
            print(f"ERROR en {image_path}: {exc}")

    if not rows:
        raise RuntimeError("No se pudo procesar ninguna imagen.")

    df = pd.DataFrame(rows)
    summary = build_reports(df, output_dir)

    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "gradcam_errors.csv", index=False, encoding="utf-8-sig")

    print("\nEvaluación finalizada.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nArchivos generados en: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
