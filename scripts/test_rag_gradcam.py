"""Prueba rápida de RAG mejorado y Grad-CAM."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.retriever import LocalRetriever
from app.rag.generator import AnswerGenerator

r = LocalRetriever()
pred = {"class_name": "RedRot", "confidence": 0.95}
q = "cuales son los sintomas y como mitigar la podredumbre roja"
chunks = r.search(q, k=5, prediction=pred)
print("Chunks recuperados:")
for c in chunks:
    print(f"  [{c['final_score']:.3f}] {c['source']} - {c.get('title')} - {c.get('diseases')}")

gen = AnswerGenerator()
ans = gen._fallback(q, chunks, pred)
print("\n--- Fallback ---\n")
print(ans[:900])

try:
    import torch
    from torchvision import transforms
    from PIL import Image
    import numpy as np
    from app.classifier.models import build_torch_model
    from app.classifier.gradcam import generate_gradcam, gradcam_supported

    print("\nGrad-CAM soportado para resnet50:", gradcam_supported("resnet50", "torch"))
    model = build_torch_model("resnet50", 5).eval()
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(img).unsqueeze(0)
    overlay, err = generate_gradcam(model, "resnet50", x, 0)
    print("Grad-CAM overlay:", "OK" if overlay else f"Error: {err}")
except Exception as e:
    print("Grad-CAM test error:", e)
