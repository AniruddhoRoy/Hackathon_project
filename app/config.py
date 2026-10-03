import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
VECTOR_DIR = DATA_DIR / "vectordb"
DB_PATH = DATA_DIR / "app.db"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # ollama | google

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemma-4-26b-a4b-it")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
COLLECTION_NAME = "nctb_chunks"

TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0"))

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "change-me")

PDF_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
