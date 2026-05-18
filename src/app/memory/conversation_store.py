from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import json, uuid
from datetime import datetime

# Campos que no deben persistirse en JSON (objetos PIL, tensores, etc.)
_NON_SERIALIZABLE_KEYS = frozenset({"gradcam_image", "image", "overlay"})


def _json_safe(value: Any) -> Any:
    """Convierte estructuras a tipos serializables por JSON."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            k: _json_safe(v)
            for k, v in value.items()
            if k not in _NON_SERIALIZABLE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        from PIL import Image
        if isinstance(value, Image.Image):
            return None
    except ImportError:
        pass
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    if hasattr(value, "item"):  # escalares numpy/torch
        try:
            return value.item()
        except Exception:
            pass
    return str(value)

ROOT = Path(__file__).resolve().parents[3]
HISTORY_DIR = ROOT / "data" / "conversations"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]


def _path(session_id: str) -> Path:
    safe = "".join(c for c in session_id if c.isalnum() or c in "_-.")
    return HISTORY_DIR / f"{safe}.json"


def load_history(session_id: str) -> List[Dict]:
    p = _path(session_id)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_history(session_id: str, messages: List[Dict]) -> None:
    p = _path(session_id)
    safe_messages = _json_safe(messages)
    p.write_text(json.dumps(safe_messages, ensure_ascii=False, indent=2), encoding="utf-8")


def append_message(session_id: str, role: str, content: str, extra: Dict | None = None) -> None:
    history = load_history(session_id)
    item = {"role": role, "content": content, "timestamp": datetime.now().isoformat(timespec="seconds")}
    if extra:
        item.update(_json_safe(extra))
    history.append(item)
    save_history(session_id, history)


def list_sessions() -> List[str]:
    return sorted([p.stem for p in HISTORY_DIR.glob("*.json")], reverse=True)


def clear_history(session_id: str) -> None:
    p = _path(session_id)
    if p.exists():
        p.unlink()
