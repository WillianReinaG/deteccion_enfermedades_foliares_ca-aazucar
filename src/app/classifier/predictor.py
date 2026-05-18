from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.config.settings import MODEL_DIR, ARTIFACTS_DIR, CLASS_NAMES_DEFAULT, IMG_SIZE
from app.classifier.models import build_torch_model
from app.classifier.gradcam import generate_gradcam, gradcam_supported

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

EVAL_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_class_names() -> List[str]:
    candidates = [ARTIFACTS_DIR / "class_names.json", MODEL_DIR / "class_names.json"]
    for p in candidates:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "class_names" in data:
                return list(data["class_names"])
            if isinstance(data, list):
                return list(data)
    return CLASS_NAMES_DEFAULT


def find_best_model() -> Dict[str, Any]:
    leaderboard = ARTIFACTS_DIR / "leaderboard_final.csv"
    if leaderboard.exists():
        df = pd.read_csv(leaderboard)
        metric = "macro_f1" if "macro_f1" in df.columns else "top1"
        row = df.sort_values(metric, ascending=False).iloc[0].to_dict()
        ckpt = Path(str(row.get("best_ckpt", "")))
        local = MODEL_DIR / ckpt.name
        if local.exists():
            row["best_ckpt"] = str(local)
        elif ckpt.exists():
            row["best_ckpt"] = str(ckpt)
        else:
            # Buscar best.pt por nombre de modelo
            matches = list(MODEL_DIR.rglob("best.pt")) + list(MODEL_DIR.rglob("*.pt"))
            if matches:
                row["best_ckpt"] = str(matches[0])
        return row

    metadata = MODEL_DIR / "model_metadata.json"
    if metadata.exists():
        return json.loads(metadata.read_text(encoding="utf-8"))

    matches = list(MODEL_DIR.rglob("best.pt")) + list(MODEL_DIR.rglob("*.pt"))
    if matches:
        return {"model_id": "resnet50", "framework": "torch", "best_ckpt": str(matches[0])}
    return {"model_id": "demo", "framework": "demo", "best_ckpt": ""}


class SugarCanePredictor:
    def __init__(self):
        self.class_names = load_class_names()
        self.info = find_best_model()
        self.framework = str(self.info.get("framework", "torch")).lower()
        self.model_id = str(self.info.get("model_id", "resnet50"))
        self.ckpt_path = Path(str(self.info.get("best_ckpt", ""))) if self.info.get("best_ckpt") else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.demo_mode = False
        self._load()

    def _load(self):
        if not self.ckpt_path or not self.ckpt_path.exists():
            self.demo_mode = True
            return
        if self.framework == "yolo" or "yolo" in self.model_id.lower():
            if YOLO is None:
                raise RuntimeError("Ultralytics no está instalado. Ejecuta: pip install ultralytics")
            self.model = YOLO(str(self.ckpt_path))
            self.framework = "yolo"
            return
        self.model = build_torch_model(self.model_id, len(self.class_names)).to(self.device).eval()
        ck = torch.load(str(self.ckpt_path), map_location=self.device)
        state = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
        self.model.load_state_dict(state, strict=False)
        self.framework = "torch"

    def predict(self, image: Image.Image, with_gradcam: bool = True) -> Dict[str, Any]:
        image = image.convert("RGB")
        if self.demo_mode:
            return self._demo_predict()
        if self.framework == "yolo":
            res = self.model.predict(image, imgsz=IMG_SIZE, verbose=False)[0]
            probs = res.probs.data.detach().cpu().numpy()
            names = [res.names[i] for i in range(len(probs))] if hasattr(res, "names") else self.class_names
            return self._format(probs, names)
        x = EVAL_TF(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        result = self._format(probs, self.class_names)
        if with_gradcam and gradcam_supported(self.model_id, self.framework):
            top_idx = int(np.argmax(probs))
            overlay, err = generate_gradcam(self.model, self.model_id, x, top_idx, self.device)
            result["gradcam_image"] = overlay
            result["gradcam_class"] = result["class_name"]
            result["gradcam_error"] = err
        return result

    def _format(self, probs: np.ndarray, names: List[str]) -> Dict[str, Any]:
        order = np.argsort(probs)[::-1]
        top = int(order[0])
        return {
            "class_name": str(names[top]),
            "confidence": float(probs[top]),
            "top_predictions": [
                {"class_name": str(names[i]), "confidence": float(probs[i])}
                for i in order[: min(5, len(order))]
            ],
            "model_id": self.model_id,
            "framework": self.framework,
            "checkpoint": str(self.ckpt_path),
            "demo_mode": False,
        }

    def _demo_predict(self) -> Dict[str, Any]:
        probs = np.array([0.12, 0.18, 0.22, 0.31, 0.17], dtype=float)
        return {**self._format(probs, self.class_names), "demo_mode": True, "checkpoint": "No configurado"}
