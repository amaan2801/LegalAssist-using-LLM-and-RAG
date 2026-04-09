from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
VECTOR_DIR = BASE_DIR / "data" / "vectorstore"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

TOP_K_RESULTS = 4

LLM_MODEL = "deepseek-r1:1.5b"
LLM_TEMPERATURE = 0.1