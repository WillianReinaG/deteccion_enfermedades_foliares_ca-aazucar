from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = BASE_DIR / os.getenv("MODEL_DIR", "models")
ARTIFACTS_DIR = BASE_DIR / os.getenv("ARTIFACTS_DIR", "artifacts")
KNOWLEDGE_DIR = BASE_DIR / os.getenv("KNOWLEDGE_DIR", "src/app/knowledge_base")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD", "").strip()
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "bebesowi@gmail.com").strip()
ALERT_FROM = os.getenv("ALERT_FROM", "").strip() or SMTP_USER
ALERT_CONFIDENCE_MIN = float(os.getenv("ALERT_CONFIDENCE_MIN", "0.5"))

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "").strip()
BQ_DATASET = os.getenv("BQ_DATASET", "sugarcane")
BQ_TABLE = os.getenv("BQ_TABLE", "predictions")

CLASS_NAMES_DEFAULT = ["Bacterial Blight","Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]
IMG_SIZE = 224
