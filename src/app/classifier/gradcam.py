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
    if framework != "torch":
        return False
    key = model_id.lower()
    return any(x in key for x in ("resnet", "mobilenet", "regnet", "simple_cnn"))


def _get_target_layers(model: nn.Module, model_id: str) -> Optional[List[nn.Module]]:
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
    """Invierte normalización ImageNet para visualización."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * std + mean, 0, 1)
    return (img * 255).astype(np.uint8)


def generate_gradcam(
    model: nn.Module,
    model_id: str,
    input_tensor: torch.Tensor,
    class_index: int,
    device: str = "cpu",
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Genera mapa Grad-CAM superpuesto sobre la imagen de entrada.
    Retorna (PIL overlay, mensaje_error).
    """
    if not GRADCAM_AVAILABLE:
        return None, "Instale grad-cam: pip install grad-cam opencv-python-headless"

    target_layers = _get_target_layers(model, model_id)
    if not target_layers:
        return None, f"Grad-CAM no configurado para el modelo '{model_id}'."

    model.eval()
    rgb = _tensor_to_rgb_uint8(input_tensor)
    float_img = rgb.astype(np.float32) / 255.0

    try:
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(class_index)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        overlay = show_cam_on_image(float_img, grayscale_cam, use_rgb=True)
        return Image.fromarray(overlay), None
    except Exception as e:
        return None, f"Error al generar Grad-CAM: {e}"
    finally:
        if "cam" in locals():
            cam.__del__()
