"""
Copia los artefactos generados por el notebook de entrenamiento al proyecto Streamlit.
Uso ejemplo:
python scripts/prepare_artifacts.py --exp-root "C:/ruta/sugarcane_multimodel_2026"
"""
from pathlib import Path
import argparse, shutil, json
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--exp-root", required=True, help="Carpeta sugarcane_multimodel_2026 generada por el notebook")
args = parser.parse_args()

root = Path(args.exp_root)
project = Path(__file__).resolve().parents[1]
models_dir = project / "models"
artifacts_dir = project / "artifacts"
models_dir.mkdir(exist_ok=True)
artifacts_dir.mkdir(exist_ok=True)

for name in ["leaderboard_final.csv", "leaderboard_screen_all.csv", "computational_cost.csv", "full_experiment_summary.json"]:
    src = root / name
    if src.exists():
        shutil.copy2(src, artifacts_dir / name)

summary = root / "full_experiment_summary.json"
if summary.exists():
    data = json.loads(summary.read_text(encoding="utf-8"))
    if "class_names" in data:
        (artifacts_dir / "class_names.json").write_text(json.dumps(data["class_names"], ensure_ascii=False, indent=2), encoding="utf-8")

leader = artifacts_dir / "leaderboard_final.csv"
if leader.exists():
    df = pd.read_csv(leader)
    metric = "macro_f1" if "macro_f1" in df.columns else "top1"
    best = df.sort_values(metric, ascending=False).iloc[0]
    ckpt = Path(str(best.get("best_ckpt", "")))
    if ckpt.exists():
        dst = models_dir / "best.pt"
        shutil.copy2(ckpt, dst)
        meta = best.to_dict()
        meta["best_ckpt"] = str(dst)
        (models_dir / "model_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Mejor modelo copiado: {best.get('model_id')} -> {dst}")
    else:
        print("No se encontró el checkpoint indicado en leaderboard_final.csv. Copia manualmente el best.pt en models/.")
else:
    print("No se encontró leaderboard_final.csv. Copia manualmente los artefactos.")
