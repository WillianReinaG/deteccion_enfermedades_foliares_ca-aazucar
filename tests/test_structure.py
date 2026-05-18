from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_project_layout():
    for rel in ("src/app", "src/scripts", "tests", "docker", "data"):
        assert (ROOT / rel).exists(), f"Falta directorio: {rel}"
