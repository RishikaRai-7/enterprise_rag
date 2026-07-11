import os
from dotenv import load_dotenv

load_dotenv()

# ==========================
# API Keys
# ==========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ==========================
# Database
# ==========================
CHROMA_DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    "data/chroma_db"
)

# ==========================
# Embeddings
# ==========================
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5"
)

# ==========================
# Chunking
# ==========================
CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", 500)
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", 100)
)

# ==========================
# Retrieval
# ==========================
TOP_K = int(
    os.getenv("TOP_K", 10)
)

RERANK_TOP_K = int(
    os.getenv("RERANK_TOP_K", 5)
)