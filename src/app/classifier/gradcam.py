from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GradCAM = None
    ClassifierOutputTarget = None
    show_cam_on_image = None
    GRADCAM_AVAILABLE = False


def gradcam_supported(model_id: str, framework: str) -> bool:
    """Indica si el modelo cargado soporta Grad-CAM."""
    if framework != "torch":
        return False
    key = model_id.lower()
    return any(x in key for x in ("resnet", "mobilenet", "regnet", "simple_cnn"))


def _get_target_layers(model: nn.Module, model_id: str) -> Optional[List[nn.Module]]:
    """Selecciona automáticamente la última capa convolucional relevante."""
    key = model_id.lower()

    if "resnet" in key and hasattr(model, "layer4"):
        return [model.layer4[-1]]

    if "mobilenet" in key and hasattr(model, "features"):
        return [model.features[-1]]

    if "regnet" in key:
        if hasattr(model, "trunk_output") and hasattr(model.trunk_output, "block4"):
            return [model.trunk_output.block4[-1]]
        if hasattr(model, "stem") and hasattr(model, "trunk"):
            return [model.trunk[-1]]

    if "simple_cnn" in key and hasattr(model, "features"):
        conv_layers = [m for m in model.features if isinstance(m, nn.Conv2d)]
        if conv_layers:
            return [conv_layers[-1]]

    return None


def _tensor_to_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Invierte la normalización ImageNet para visualización."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = tensor.squeeze(0).detach().permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * std + mean, 0, 1)
    return (img * 255).astype(np.uint8)


def generate_gradcam(
    model: nn.Module,
    model_id: str,
    input_tensor: torch.Tensor,
    class_index: int,
    device: str = "cpu",
) -> Tuple[Optional[Image.Image], Optional[str], Optional[np.ndarray]]:
    """
    Genera Grad-CAM para una imagen clasificada.

    Retorna:
        overlay: imagen PIL con mapa de calor superpuesto.
        error: mensaje de error si ocurre algún problema.
        grayscale_cam: matriz Grad-CAM normalizada en rango [0, 1].
    """
    if not GRADCAM_AVAILABLE:
        return None, "Instale grad-cam: pip install grad-cam opencv-python-headless", None

    target_layers = _get_target_layers(model, model_id)
    if not target_layers:
        return None, f"Grad-CAM no configurado para el modelo '{model_id}'.", None

    model.eval()
    rgb = _tensor_to_rgb_uint8(input_tensor)
    float_img = rgb.astype(np.float32) / 255.0

    cam = None
    try:
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(class_index)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        grayscale_cam = np.asarray(grayscale_cam, dtype=np.float32)
        grayscale_cam = np.clip(grayscale_cam, 0.0, 1.0)

        overlay = show_cam_on_image(float_img, grayscale_cam, use_rgb=True)
        return Image.fromarray(overlay), None, grayscale_cam

    except Exception as e:
        return None, f"Error al generar Grad-CAM: {e}", None

    finally:
        if cam is not None:
            try:
                cam.__del__()
            except Exception:
                pass


def generate_gradcam_array(
    model: nn.Module,
    model_id: str,
    input_tensor: torch.Tensor,
    class_index: int,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Devuelve únicamente la matriz Grad-CAM cruda para cálculo de métricas.

    Retorna:
        grayscale_cam: matriz Grad-CAM normalizada.
        error: mensaje de error si no se pudo generar.
    """
    overlay, err, grayscale_cam = generate_gradcam(
        model=model,
        model_id=model_id,
        input_tensor=input_tensor,
        class_index=class_index,
    )
    return grayscale_cam, err
