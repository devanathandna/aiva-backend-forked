import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Always load the backend .env from the project root, regardless of where scripts are run.
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

# ── Gemini API keys (rotation: KEY_1 → KEY_2 on rate-limit) ──────────
GEMINI_API_KEY_1 = os.getenv("GEMINI_API_KEY_1", "").strip()
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2", "").strip()

# Fallback: single GEMINI_API_KEY for backward compat
_fallback = os.getenv("GEMINI_API_KEY", "").strip()

GEMINI_API_KEYS: list = [k for k in [GEMINI_API_KEY_1, GEMINI_API_KEY_2] if k]
if not GEMINI_API_KEYS and _fallback:
    GEMINI_API_KEYS = [_fallback]

if not GEMINI_API_KEYS:
    raise RuntimeError(
        "No Gemini API key found. "
        "Set GEMINI_API_KEY_1 and GEMINI_API_KEY_2 (or GEMINI_API_KEY) in .env"
    )

# For backward compat — modules that import GEMINI_API_KEY directly
GEMINI_API_KEY = GEMINI_API_KEYS[0]

# ── Gemini embedding model ────────────────────────────────────────────
EMBEDDING_MODEL = "models/gemini-embedding-001"

# ── Source documents ──────────────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, "rag_faiss", "data")

KNOWLEDGE_FILES = {
    "Achievements": os.path.join(BASE_DIR, "Achievements.txt"),
    "Admissions":   os.path.join(BASE_DIR, "Admissions.txt"),
    "Cutoffs":      os.path.join(BASE_DIR, "Cutoffs.txt"),
    "Dataset":      os.path.join(BASE_DIR, "Dataset.txt"),
    "Departments":  os.path.join(BASE_DIR, "Departments.txt"),
    "Fees_Structure":   os.path.join(BASE_DIR, "Fees_Structure.txt"),
    "Higher_Education": os.path.join(BASE_DIR, "Higher_Education.txt"),
    "Hostel":       os.path.join(BASE_DIR, "Hostel.txt"),
    "Overview":     os.path.join(BASE_DIR, "Overview.txt"),
    "Placements":   os.path.join(BASE_DIR, "Placements.txt"),
    "Bus_Details":  os.path.join(BASE_DIR, "Bus_Details.txt"),
}

# ── Persistence paths ─────────────────────────────────────────────────
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "rag_faiss", "embeddings")
PICKLES_DIR    = os.path.join(BASE_DIR, "rag_faiss", "pickles")

FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss_index.bin")
INDEX_MAP_PATH   = os.path.join(EMBEDDINGS_DIR, "index_map.pkl")

# ── Chunking ──────────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100

# ── FAISS tuning ──────────────────────────────────────────────────────
HNSW_M              = 40
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH       = 64

# ── Retrieval defaults ────────────────────────────────────────────────
TOP_K = 5
