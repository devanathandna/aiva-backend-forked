"""
retriever.py – Load FAISS index once, then answer queries with
sub-50 ms retrieval by looking up pickle chunks via the index map.

Embedding: Google Gemini API (models/gemini-embedding-001)
  - Cloud-hosted, no local model download required
  - Dual API-key rotation on rate-limit (429) errors
  - LRU embedding cache – identical queries skip the API call entirely
  - Persistent FAISS index loaded once at startup

Usage (standalone test):
    python -m rag_faiss.retriever
"""

import os
import pickle
import time
import threading
import numpy as np
import faiss

from typing import Optional, Dict, Tuple, List
import logging

import google.generativeai as genai

from rag_faiss.config import (
    GEMINI_API_KEYS,
    FAISS_INDEX_PATH,
    INDEX_MAP_PATH,
    PICKLES_DIR,
    TOP_K,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


# ── Gemini key rotation manager ──────────────────────────────────────────────

class _GeminiKeyManager:
    """Thread-safe round-robin over GEMINI_API_KEY_1 / _2."""

    def __init__(self, keys: List[str]):
        self._keys  = keys
        self._index = 0
        self._lock  = threading.Lock()
        # Configure the first key immediately
        genai.configure(api_key=self._keys[0])
        logger.info("[RETRIEVER] Gemini keys loaded: %d key(s)", len(self._keys))

    @property
    def current_key(self) -> str:
        with self._lock:
            return self._keys[self._index]

    def rotate(self) -> bool:
        """Advance to the next key. Returns True if rotation succeeded."""
        with self._lock:
            if len(self._keys) <= 1:
                return False
            prev = self._index
            self._index = (self._index + 1) % len(self._keys)
            genai.configure(api_key=self._keys[self._index])
            logger.warning(
                "[RETRIEVER] 🔄 Rotated Gemini key #%d → #%d",
                prev + 1, self._index + 1,
            )
            return True


_key_mgr = _GeminiKeyManager(GEMINI_API_KEYS)


# ── Module-level singletons (loaded once at startup) ─────────────────────────
_faiss_index:       Optional[faiss.Index]             = None
_index_map:         Optional[Dict[int, Tuple[str, int]]] = None
_pickle_cache:      Dict[str, List[str]]               = {}
_loaded_successfully: bool                             = False

# LRU embedding cache – skip API call for repeated queries
_embed_cache:     Dict[str, np.ndarray] = {}
_EMBED_CACHE_MAX  = 512


# ── FAISS index loader ───────────────────────────────────────────────────────

def _ensure_loaded() -> None:
    """Load FAISS index and index map into memory on first call."""
    global _faiss_index, _index_map, _loaded_successfully

    if _loaded_successfully:
        return

    if not os.path.exists(FAISS_INDEX_PATH):
        logger.warning(
            "[RETRIEVER] FAISS index not found at %s. "
            "RAG disabled until the index is built.",
            FAISS_INDEX_PATH,
        )
        _faiss_index = None
        _index_map   = None
        return

    try:
        _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(INDEX_MAP_PATH, "rb") as f:
            _index_map = pickle.load(f)
        _loaded_successfully = True
        logger.info("[RETRIEVER] ✅ FAISS index loaded (%d vectors)", _faiss_index.ntotal)
    except Exception as exc:
        logger.error("[RETRIEVER] ❌ Failed to load FAISS index: %s", exc)
        _faiss_index = None
        _index_map   = None


# ── Pickle chunk loader (cached) ─────────────────────────────────────────────

def _load_pickle(pickle_filename: str) -> List[str]:
    if pickle_filename not in _pickle_cache:
        path = os.path.join(PICKLES_DIR, pickle_filename)
        with open(path, "rb") as f:
            _pickle_cache[pickle_filename] = pickle.load(f)
    return _pickle_cache[pickle_filename]


# ── Query embedder (Gemini API with key rotation) ────────────────────────────

def _embed_query(text: str) -> np.ndarray:
    """
    Embed a single query string using Gemini embedding API.
    • Returns a (1, dim) float32 array, L2-normalised.
    • Uses LRU cache to skip identical repeated queries.
    • Rotates to next Gemini key on rate-limit errors.
    """
    cache_key = text.strip().lower()
    if cache_key in _embed_cache:
        logger.debug("[RETRIEVER] Embedding cache HIT")
        return _embed_cache[cache_key]

    max_retries = len(GEMINI_API_KEYS)
    for attempt in range(max_retries):
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
            )
            vec = np.array([result["embedding"]], dtype=np.float32)
            faiss.normalize_L2(vec)

            # Store in LRU cache
            if len(_embed_cache) >= _EMBED_CACHE_MAX:
                oldest = next(iter(_embed_cache))
                del _embed_cache[oldest]
            _embed_cache[cache_key] = vec
            return vec

        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "rate" in err_str or "quota" in err_str
            if is_rate_limit and attempt < max_retries - 1:
                logger.warning(
                    "[RETRIEVER] Rate-limited on key #%d, rotating...",
                    _key_mgr._index + 1,
                )
                if _key_mgr.rotate():
                    time.sleep(0.5)
                    continue
            raise  # non-rate-limit error or all keys exhausted

    raise RuntimeError("[RETRIEVER] All Gemini keys exhausted for embedding")


# ── Public retrieval API ─────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> dict:
    """
    Find the top-k most relevant document chunks for a query.

    Returns:
        {
            "context": "<concatenated chunk text>",
            "sources": ["Achievements.txt", ...]
        }
    """
    _ensure_loaded()
    if _faiss_index is None or _index_map is None:
        logger.warning("[RETRIEVER] FAISS index not available – returning empty context")
        return {"context": "", "sources": []}

    t0 = time.perf_counter()
    query_vec = _embed_query(query)
    embed_ms  = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    distances, ids = _faiss_index.search(query_vec, top_k)
    search_ms = (time.perf_counter() - t1) * 1000

    logger.info("[RETRIEVER] embed=%.0fms  faiss=%.1fms", embed_ms, search_ms)

    chunks:       List[str] = []
    sources_seen: List[str] = []

    for faiss_id in ids[0]:
        if faiss_id == -1:
            continue
        pickle_filename, chunk_idx = _index_map[int(faiss_id)]
        pickle_chunks = _load_pickle(pickle_filename)
        chunks.append(pickle_chunks[chunk_idx])

        source_txt = pickle_filename.replace(".pkl", ".txt")
        if source_txt not in sources_seen:
            sources_seen.append(source_txt)

    return {
        "context": "\n\n".join(chunks),
        "sources": sources_seen,
    }


# ── Quick CLI smoke-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)
    _ensure_loaded()

    if _faiss_index is None:
        print("[RETRIEVER] ❌ No FAISS index found – run build_index first.")
    else:
        print(f"[RETRIEVER] Index loaded: {_faiss_index.ntotal} vectors")
        print(f"[RETRIEVER] Gemini keys: {len(GEMINI_API_KEYS)}")

        test_queries = [
            "What awards did students win?",
            "CSE Department BC cutoff mark?",
            "What are the hostel facilities?",
        ]
        for q in test_queries:
            start  = time.perf_counter()
            result = retrieve(q)
            ms     = (time.perf_counter() - start) * 1000
            print(f"\nQuery   : {q}")
            print(f"Sources : {result['sources']}")
            print(f"Latency : {ms:.1f} ms")
            print(f"Context : {result['context'][:200]} …")
