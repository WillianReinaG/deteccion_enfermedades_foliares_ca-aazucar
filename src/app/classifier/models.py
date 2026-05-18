import torch
import torch.nn as nn
from torchvision import models
import timm

class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_torch_model(model_key: str, n_classes: int):
    model_key = model_key.lower().strip()
    if model_key == "simple_cnn":
        return SimpleCNN(n_classes)
    if model_key == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        return m
    if model_key == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
        return m
    if model_key == "regnet_y_3_2gf":
        m = models.regnet_y_3_2gf(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        return m
    if model_key in {"vit2026", "vit", "vit_base_patch16_224"}:
        return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=n_classes)
    if model_key in {"hybrid_cnn_transformer", "mobilevit_s", "mobilevit"}:
        try:
            return timm.create_model("mobilevit_s", pretrained=False, num_classes=n_classes)
        except Exception:
            return timm.create_model("mobilevitv2_100", pretrained=False, num_classes=n_classes)
    raise ValueError(f"Modelo PyTorch no soportado: {model_key}")
