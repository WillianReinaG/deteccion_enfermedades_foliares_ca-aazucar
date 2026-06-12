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

# RAG — recuperación documental
RAG_RETRIEVAL_METHOD = os.getenv("RAG_RETRIEVAL_METHOD", "semantic").strip().lower()
SEMANTIC_MODEL_NAME = os.getenv(
    "SEMANTIC_MODEL_NAME", "paraphrase-multilingual-mpnet-base-v2"
).strip()
FINETUNED_EMBEDDING_PATH = os.getenv("FINETUNED_EMBEDDING_PATH", "").strip()
USE_OPENAI_EMBEDDINGS = os.getenv("USE_OPENAI_EMBEDDINGS", "false").strip().lower() in {"1", "true", "yes"}
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "1400"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "300"))
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))
HYBRID_SEMANTIC_WEIGHT = float(os.getenv("HYBRID_SEMANTIC_WEIGHT", "0.5"))
HYBRID_BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", "0.5"))
RESULTS_DIR = BASE_DIR / os.getenv("RESULTS_DIR", "results")
EMBEDDINGS_DIR = MODEL_DIR / "embeddings"
