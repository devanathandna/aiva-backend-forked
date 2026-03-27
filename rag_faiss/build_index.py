"""
build_index.py – Read .txt sources, chunk them, embed via Gemini,
build a FAISS index, and persist pickle + mapping files.

Embedding: Google Gemini API (models/gemini-embedding-001)
  - 768-dim vectors, cloud-hosted, no local model download
  - Dual API-key rotation on rate-limit (429) errors
  - Batch embedding for fast index builds

Usage:
    python -m rag_faiss.build_index
"""

import os
import pickle
import time
import threading
from typing import Dict, List, Tuple

import numpy as np
import faiss
import google.generativeai as genai

from rag_faiss.config import (
    GEMINI_API_KEYS,
    KNOWLEDGE_FILES,
    EMBEDDINGS_DIR,
    PICKLES_DIR,
    FAISS_INDEX_PATH,
    INDEX_MAP_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)


# ── Gemini key rotation manager ──────────────────────────────────────────────

class _GeminiKeyManager:
    """Thread-safe round-robin over GEMINI_API_KEY_1 / _2."""

    def __init__(self, keys: List[str]):
        self._keys  = keys
        self._index = 0
        self._lock  = threading.Lock()
        # Configure the first key immediately
        genai.configure(api_key=self._keys[0])
        print(f"[GEMINI] Loaded {len(self._keys)} API key(s)")

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
            print(f"[GEMINI] 🔄 Rotated key #{prev + 1} → #{self._index + 1}")
            return True


_key_mgr = _GeminiKeyManager(GEMINI_API_KEYS)

# Chunk limits
_MAX_EMBED_CHARS = 9_000
_MIN_CHUNK_CHARS = 10


# ── Text helpers ─────────────────────────────────────────────────────────────

def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunk_size = max(100, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))
    step = max(1, chunk_size - overlap)

    chunks: List[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if len(chunk) >= _MIN_CHUNK_CHARS:
            chunks.append(chunk[:_MAX_EMBED_CHARS])
        start += step
    return chunks


# ── Gemini embedding with key rotation ───────────────────────────────────────

def _embed_texts(texts: List[str], batch_size: int = 10) -> np.ndarray:
    """Embed a list of texts using Gemini, with dual-key rotation on rate-limit."""
    # Filter invalid texts
    texts = [t for t in texts if t and t.strip() and len(t.strip()) >= _MIN_CHUNK_CHARS]
    if not texts:
        raise RuntimeError("All text chunks were empty/too-short after filtering.")

    all_embeddings: list = []
    requests_this_window = 0
    window_start = time.time()

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # If we're about to exceed 95 requests in this 60-second window, wait
        if requests_this_window + len(batch) > 95:
            elapsed = time.time() - window_start
            wait = max(0, 62 - elapsed)
            if wait > 0:
                print(f"[INDEX] Rate limit pause: {wait:.0f}s ...")
                time.sleep(wait)
            requests_this_window = 0
            window_start = time.time()

        # Try embedding with key rotation on failure
        max_retries = len(GEMINI_API_KEYS)
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=batch,
                )
                all_embeddings.extend(result["embedding"])
                requests_this_window += len(batch)
                break  # success
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate" in err_str or "quota" in err_str
                if is_rate_limit and attempt < max_retries - 1:
                    print(f"[INDEX] ⚠️ Rate-limited on key #{_key_mgr._index + 1}, rotating...")
                    if _key_mgr.rotate():
                        time.sleep(1)  # brief pause before retry
                        continue
                # Non-rate-limit error or all keys exhausted
                raise RuntimeError(f"Gemini embed failed after {attempt + 1} attempt(s): {e}")

        done = min(i + batch_size, len(texts))
        print(f"[INDEX] Embedded {done}/{len(texts)} chunks")

        # Small pause between batches
        if done < len(texts):
            time.sleep(0.5)

    arr = np.array(all_embeddings, dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr


# ── Index builder ────────────────────────────────────────────────────────────

def build_index() -> Tuple[int, int]:

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(PICKLES_DIR, exist_ok=True)

    index_map:   Dict[int, Tuple[str, int]] = {}
    all_vectors: List[np.ndarray] = []
    next_id      = 0
    total_chunks = 0

    for name, file_path in KNOWLEDGE_FILES.items():
        if not os.path.exists(file_path):
            print(f"[build_index] ⚠️  Skipping missing file: {file_path}")
            continue

        raw    = _read_text_file(file_path)
        chunks = _chunk_text(raw, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            print(f"[build_index] ⚠️  No valid chunks for: {file_path}")
            continue

        print(f"[build_index] 📄 {name}: {len(chunks)} chunks – embedding …")

        # Persist chunks to pickle BEFORE embedding
        pickle_filename = f"{name}.pkl"
        with open(os.path.join(PICKLES_DIR, pickle_filename), "wb") as pf:
            pickle.dump(chunks, pf)

        vecs = _embed_texts(chunks)
        all_vectors.append(vecs)

        for i in range(len(chunks)):
            index_map[next_id] = (pickle_filename, i)
            next_id += 1

        total_chunks += len(chunks)
        print(f"[build_index] ✅ {name}: {len(chunks)} chunks embedded")

    if not all_vectors:
        raise RuntimeError("No documents/chunks found. Check KNOWLEDGE_FILES paths.")

    vectors = np.vstack(all_vectors)
    dim     = int(vectors.shape[1])

    # IndexFlatIP = exact cosine search on L2-normalised vectors (ideal for <10K vecs)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(INDEX_MAP_PATH, "wb") as f:
        pickle.dump(index_map, f)

    return total_chunks, index.ntotal


if __name__ == "__main__":
    t0 = time.time()
    chunks, ntotal = build_index()
    elapsed = time.time() - t0
    print(f"\n[build_index] ✅ Done in {elapsed:.1f}s")
    print(f"[build_index]    Vectors : {ntotal}")
    print(f"[build_index]    Chunks  : {chunks}")
    print(f"[build_index]    Keys    : {len(GEMINI_API_KEYS)} Gemini key(s)")
    print(f"[build_index]    Index   → {FAISS_INDEX_PATH}")
    print(f"[build_index]    Map     → {INDEX_MAP_PATH}")
