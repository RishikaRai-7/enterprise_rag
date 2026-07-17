import os
from dotenv import load_dotenv

load_dotenv()


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "deepseek/deepseek-chat-v3"
)

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
# Cross Encoder
# ==========================
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
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