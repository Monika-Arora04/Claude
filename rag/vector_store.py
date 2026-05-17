"""
Vector store with cosine similarity search, BM25 keyword search, and hybrid search.

Features:
- In-memory storage with numpy for fast cosine similarity
- BM25 keyword search using rank_bm25 (or fallback TF-IDF scoring)
- Hybrid search combining dense + sparse signals
- Save / load to disk using numpy + pickle
"""

from __future__ import annotations

import logging
import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .chunker import Chunk
from .embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single search result returned from the vector store."""

    chunk: Chunk
    score: float
    rank: int
    search_type: str = "hybrid"

    @property
    def content(self) -> str:
        return self.chunk.content

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def metadata(self) -> dict[str, Any]:
        return self.chunk.metadata

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return (
            f"SearchResult(rank={self.rank}, score={self.score:.4f}, "
            f"source={self.source!r}, content={preview!r}...)"
        )


# ---------------------------------------------------------------------------
# BM25 implementation (pure Python, no external deps required)
# ---------------------------------------------------------------------------


class _BM25:
    """
    Okapi BM25 implementation.

    Uses rank_bm25 library when available; otherwise falls back to a
    pure-Python implementation with the same interface.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._corpus_tokens: list[list[str]] = []
        self._doc_freqs: Counter[str] = Counter()
        self._idf: dict[str, float] = {}
        self._avg_dl: float = 0.0
        self._n: int = 0
        self._use_rank_bm25 = False
        self._bm25_obj: Any = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b[a-z0-9_\-\.]+\b", text.lower())

    def fit(self, texts: list[str]) -> "_BM25":
        """Build the BM25 index from a list of documents."""
        self._corpus_tokens = [self._tokenize(t) for t in texts]
        self._n = len(self._corpus_tokens)

        # Try rank_bm25
        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            self._bm25_obj = BM25Okapi(self._corpus_tokens, k1=self.k1, b=self.b)
            self._use_rank_bm25 = True
            logger.debug("BM25 index built with rank_bm25 (%d docs)", self._n)
            return self
        except ImportError:
            pass

        # Pure Python fallback
        total_tokens = sum(len(t) for t in self._corpus_tokens)
        self._avg_dl = total_tokens / max(self._n, 1)

        for tokens in self._corpus_tokens:
            seen = set(tokens)
            for tok in seen:
                self._doc_freqs[tok] += 1

        for tok, df in self._doc_freqs.items():
            self._idf[tok] = math.log(
                (self._n - df + 0.5) / (df + 0.5) + 1
            )

        logger.debug("BM25 index built (pure Python, %d docs)", self._n)
        return self

    def get_scores(self, query: str) -> np.ndarray:
        """Return BM25 scores for all documents."""
        if not self._corpus_tokens:
            return np.zeros(0, dtype=np.float32)

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return np.zeros(self._n, dtype=np.float32)

        if self._use_rank_bm25 and self._bm25_obj is not None:
            scores = self._bm25_obj.get_scores(query_tokens)
            return scores.astype(np.float32)

        # Pure Python BM25
        scores = np.zeros(self._n, dtype=np.float32)
        for i, doc_tokens in enumerate(self._corpus_tokens):
            dl = len(doc_tokens)
            tf_counts = Counter(doc_tokens)
            score = 0.0
            for tok in query_tokens:
                if tok not in self._idf:
                    continue
                tf = tf_counts.get(tok, 0)
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1))
                )
                score += self._idf[tok] * tf_norm
            scores[i] = score
        return scores


# ---------------------------------------------------------------------------
# Main VectorStore
# ---------------------------------------------------------------------------


class VectorStore:
    """
    In-memory vector store combining dense (cosine) and sparse (BM25) search.

    Usage::

        store = VectorStore(embedder)
        store.add_chunks(chunks)          # index chunks
        results = store.search("query")   # hybrid search
        store.save("index.pkl")           # persist
        store2 = VectorStore.from_disk("index.pkl", embedder)
    """

    def __init__(self, embedder: EmbeddingModel) -> None:
        self._embedder = embedder
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._bm25 = _BM25()
        self._is_indexed = False

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 128) -> None:
        """
        Add chunks to the vector store.

        Fits the embedder on the corpus if it has not been fitted yet.
        Computes and stores dense embeddings for all chunks.

        Args:
            chunks: Chunks to add to the index.
            batch_size: Embedding batch size (for memory management).
        """
        if not chunks:
            logger.warning("add_chunks called with empty list — nothing to index.")
            return

        texts = [c.content for c in chunks]

        # Fit embedder if needed (TF-IDF requires fitting)
        if not self._embedder.is_fitted:
            logger.info("Fitting embedder on %d texts...", len(texts))
            self._embedder.fit(texts)

        # Embed in batches
        logger.info("Embedding %d chunks...", len(texts))
        all_vecs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vecs = self._embedder.embed_texts(batch)
            all_vecs.append(vecs)

        new_embeddings = np.vstack(all_vecs).astype(np.float32)

        # Normalize for cosine similarity via dot product
        norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        new_embeddings = new_embeddings / norms

        # Append to existing store
        if len(self._chunks) == 0:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        self._chunks.extend(chunks)

        # Rebuild BM25 index
        self._bm25.fit([c.content for c in self._chunks])
        self._is_indexed = True

        logger.info(
            "Index now contains %d chunks (embedding_dim=%d)",
            len(self._chunks),
            self._embeddings.shape[1] if self._embeddings.ndim > 1 else 0,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_dense(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Cosine similarity search using dense embeddings.

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects sorted by descending score.
        """
        if not self._is_indexed or len(self._chunks) == 0:
            return []

        q_vec = self._embedder.embed_query(query).astype(np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        # Cosine similarity = dot product (both vectors are unit-normalized)
        scores = self._embeddings @ q_vec  # shape (N,)

        k = min(top_k, len(self._chunks))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append(
                SearchResult(
                    chunk=self._chunks[idx],
                    score=float(scores[idx]),
                    rank=rank + 1,
                    search_type="dense",
                )
            )
        return results

    def search_bm25(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        BM25 keyword search.

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects sorted by descending BM25 score.
        """
        if not self._is_indexed or len(self._chunks) == 0:
            return []

        scores = self._bm25.get_scores(query)
        k = min(top_k, len(self._chunks))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append(
                SearchResult(
                    chunk=self._chunks[idx],
                    score=float(scores[idx]),
                    rank=rank + 1,
                    search_type="bm25",
                )
            )
        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> list[SearchResult]:
        """
        Hybrid search combining dense cosine similarity and BM25.

        Scores are min-max normalised independently before combining.

        Args:
            query: Query string.
            top_k: Number of results to return.
            dense_weight: Weight for dense scores (0–1).
            bm25_weight: Weight for BM25 scores (0–1).

        Returns:
            List of SearchResult objects sorted by descending hybrid score.
        """
        if not self._is_indexed or len(self._chunks) == 0:
            return []

        n = len(self._chunks)

        # Dense scores
        q_vec = self._embedder.embed_query(query).astype(np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm
        dense_scores = self._embeddings @ q_vec

        # BM25 scores
        bm25_scores = self._bm25.get_scores(query)
        if len(bm25_scores) != n:
            bm25_scores = np.zeros(n, dtype=np.float32)

        # Min-max normalise
        def _minmax(arr: np.ndarray) -> np.ndarray:
            lo, hi = arr.min(), arr.max()
            if hi == lo:
                return np.zeros_like(arr)
            return (arr - lo) / (hi - lo)

        dense_norm = _minmax(dense_scores)
        bm25_norm = _minmax(bm25_scores)
        combined = dense_weight * dense_norm + bm25_weight * bm25_norm

        k = min(top_k, n)
        top_indices = np.argpartition(combined, -k)[-k:]
        top_indices = top_indices[np.argsort(combined[top_indices])[::-1]]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append(
                SearchResult(
                    chunk=self._chunks[idx],
                    score=float(combined[idx]),
                    rank=rank + 1,
                    search_type="hybrid",
                )
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Persist the vector store to disk.

        Saves:
        - chunks (list of Chunk objects)
        - embeddings matrix (numpy array)
        - BM25 index
        - embedder state (calls embedder.save alongside)

        Args:
            path: File path for the store (e.g. "index/store.pkl").
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "chunks": self._chunks,
            "embeddings": self._embeddings,
            "bm25": self._bm25,
            "is_indexed": self._is_indexed,
        }
        with p.open("wb") as fh:
            pickle.dump(state, fh)

        # Save embedder alongside
        embedder_path = p.with_suffix(".embedder.pkl")
        self._embedder.save(embedder_path)

        logger.info(
            "VectorStore saved to %s (%d chunks)", p, len(self._chunks)
        )

    @classmethod
    def from_disk(cls, path: str | Path, embedder: EmbeddingModel) -> "VectorStore":
        """
        Load a VectorStore from disk.

        Args:
            path: Path to the saved store file.
            embedder: An EmbeddingModel instance (will be loaded from disk
                      if a matching .embedder.pkl file exists).

        Returns:
            A fully-loaded VectorStore instance.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"VectorStore file not found: {p}")

        # Restore embedder state
        embedder_path = p.with_suffix(".embedder.pkl")
        if embedder_path.exists():
            try:
                embedder.load(embedder_path)
                logger.debug("Embedder state loaded from %s", embedder_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load embedder state: %s", exc)

        with p.open("rb") as fh:
            state = pickle.load(fh)  # noqa: S301

        store = cls(embedder)
        store._chunks = state["chunks"]
        store._embeddings = state["embeddings"]
        store._bm25 = state["bm25"]
        store._is_indexed = state["is_indexed"]

        logger.info(
            "VectorStore loaded from %s (%d chunks)", p, len(store._chunks)
        )
        return store

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def is_indexed(self) -> bool:
        return self._is_indexed

    def stats(self) -> dict[str, Any]:
        """Return statistics about the current index."""
        source_counts: Counter[str] = Counter(c.source for c in self._chunks)
        type_counts: Counter[str] = Counter(
            c.metadata.get("type", "unknown") for c in self._chunks
        )
        return {
            "num_chunks": len(self._chunks),
            "embedding_dim": (
                self._embeddings.shape[1]
                if self._embeddings.ndim > 1 and len(self._chunks) > 0
                else 0
            ),
            "is_indexed": self._is_indexed,
            "sources": dict(source_counts),
            "types": dict(type_counts),
            "embedder": repr(self._embedder),
        }

    def __repr__(self) -> str:
        return (
            f"VectorStore(chunks={len(self._chunks)}, "
            f"indexed={self._is_indexed}, embedder={self._embedder!r})"
        )
