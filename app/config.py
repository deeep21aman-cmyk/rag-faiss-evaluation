import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


CHUNKS_FILE = os.path.join(BASE_DIR, "data", "chunks.json")
VECTOR_FILE = os.path.join(BASE_DIR, "data", "vectors.npy")
FILE_NAME = os.path.join(BASE_DIR, "data", "knowledge.txt")
INDEX_FILE_NAME = os.path.join(BASE_DIR, "data", "index.faiss")

CHUNK_SIZE=300

CHUNK_OVERLAP=50

TOP_K=3

EMBEDDING_MODEL_NAME="text-embedding-3-small"

OPEN_AI_MODEL="gpt-4.1-nano"