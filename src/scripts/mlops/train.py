"""
Etapa de entrenamiento (integración incremental).

Conecte aquí la exportación del notebook o un script de entrenamiento real.
Si EXP_ROOT apunta a la carpeta del experimento, copia artefactos al proyecto.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREPARE = PROJECT_ROOT / "src" / "scripts" / "prepare_artifacts.py"


def main() -> int:
    exp_root = os.getenv("EXP_ROOT", "").strip()
    if not exp_root:
        print(
            "[train] EXP_ROOT no definido. "
            "Ejecute el notebook de entrenamiento y luego prepare_artifacts, "
            "o defina EXP_ROOT en el workflow MLOps."
        )
        return 0

    if not PREPARE.exists():
        print(f"[train] No se encontró {PREPARE}", file=sys.stderr)
        return 1

    cmd = [sys.executable, str(PREPARE), "--exp-root", exp_root]
    print("[train] Ejecutando:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[train] Artefactos copiados a models/ y artifacts/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
