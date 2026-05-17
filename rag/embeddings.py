"""
Embedding models for the RAG system.

Provides two backends with a consistent interface:
1. TFIDFEmbedder  — sklearn TF-IDF (no API key required, always available)
2. SentenceTransformerEmbedder — sentence-transformers (optional upgrade)

Both expose:
  embed_texts(texts: list[str]) -> np.ndarray   # shape (N, D)
  embed_query(text: str) -> np.ndarray           # shape (D,)
  is_fitted: bool
  save(path) / load(path)
"""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class EmbeddingModel(ABC):
    """Common interface for all embedding backends."""

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """True if the model has been fitted / loaded."""

    @abstractmethod
    def fit(self, texts: list[str]) -> "EmbeddingModel":
        """Fit the model on a corpus (no-op for pre-trained models)."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Returns:
            ndarray of shape (len(texts), embedding_dim)
        """

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns:
            ndarray of shape (embedding_dim,)
        """
        return self.embed_texts([text])[0]

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the model to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> "EmbeddingModel":
        """Load the model from disk."""

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of produced embeddings."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fitted={self.is_fitted})"


# ---------------------------------------------------------------------------
# TF-IDF backend (always available)
# ---------------------------------------------------------------------------


class TFIDFEmbedder(EmbeddingModel):
    """
    TF-IDF based embedder using sklearn.

    Produces sparse vectors converted to dense float32 arrays.
    Suitable for keyword-heavy documents like logs and sensor data.

    Features:
    - No API key required
    - Works offline
    - Interpretable features (terms)
    - Fast fit and transform
    """

    def __init__(
        self,
        max_features: int = 10_000,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        min_df: int = 1,
    ) -> None:
        """
        Initialise TF-IDF embedder.

        Args:
            max_features: Maximum vocabulary size.
            ngram_range: Range for n-gram extraction.
            sublinear_tf: Apply log normalization to term frequencies.
            min_df: Minimum document frequency for a term.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sklearn is required for TFIDFEmbedder. "
                "Install with: pip install scikit-learn"
            ) from exc

        self._max_features = max_features
        self._ngram_range = ngram_range
        self._vectorizer: Any = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            strip_accents="unicode",
            analyzer="word",
            stop_words=None,  # keep domain terms like unit names
        )
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, texts: list[str]) -> "TFIDFEmbedder":
        """Fit the TF-IDF vocabulary on the corpus."""
        if not texts:
            raise ValueError("Cannot fit on empty corpus")
        logger.debug("Fitting TF-IDF on %d texts...", len(texts))
        self._vectorizer.fit(texts)
        self._fitted = True
        logger.info(
            "TF-IDF fitted: vocab_size=%d",
            len(self._vectorizer.vocabulary_),
        )
        return self

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Transform texts to TF-IDF vectors (dense float32)."""
        if not self._fitted:
            raise RuntimeError("TFIDFEmbedder must be fitted before embedding. Call fit() first.")
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        sparse = self._vectorizer.transform(texts)
        dense = sparse.toarray().astype(np.float32)
        return dense

    @property
    def embedding_dim(self) -> int:
        if not self._fitted:
            return self._max_features
        return len(self._vectorizer.vocabulary_)

    def save(self, path: str | Path) -> None:
        """Pickle the vectorizer to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "vectorizer": self._vectorizer,
            "fitted": self._fitted,
            "max_features": self._max_features,
            "ngram_range": self._ngram_range,
        }
        with p.open("wb") as fh:
            pickle.dump(state, fh)
        logger.debug("TFIDFEmbedder saved to %s", p)

    def load(self, path: str | Path) -> "TFIDFEmbedder":
        """Load a previously saved vectorizer."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
        with p.open("rb") as fh:
            state = pickle.load(fh)  # noqa: S301
        self._vectorizer = state["vectorizer"]
        self._fitted = state["fitted"]
        self._max_features = state.get("max_features", self._max_features)
        self._ngram_range = state.get("ngram_range", self._ngram_range)
        logger.debug("TFIDFEmbedder loaded from %s", p)
        return self

    def get_feature_names(self) -> list[str]:
        """Return the vocabulary terms."""
        if not self._fitted:
            return []
        return list(self._vectorizer.get_feature_names_out())


# ---------------------------------------------------------------------------
# Sentence-Transformers backend (optional)
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder(EmbeddingModel):
    """
    Embedding model using the sentence-transformers library.

    Produces dense semantic embeddings suitable for semantic similarity.
    Much stronger than TF-IDF for natural language queries.

    Requires: pip install sentence-transformers
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        """
        Initialise the sentence-transformer embedder.

        Args:
            model_name: HuggingFace model identifier.
            device: 'cpu', 'cuda', 'mps', or None (auto-detect).
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self._model_name = model_name
        self._device = device
        logger.info("Loading sentence-transformer model: %s", model_name)
        self._model: Any = SentenceTransformer(model_name, device=device)
        self._dim: int = self._model.get_sentence_embedding_dimension()
        logger.info("Model loaded. Embedding dim: %d", self._dim)

    @property
    def is_fitted(self) -> bool:
        return True  # Pre-trained model, always ready

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        """No-op for pre-trained models."""
        return self

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts using the sentence-transformer model."""
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def save(self, path: str | Path) -> None:
        """Save model name metadata (model weights are cached by HuggingFace)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = {"model_name": self._model_name, "device": self._device, "dim": self._dim}
        with p.open("wb") as fh:
            pickle.dump(meta, fh)
        logger.debug("SentenceTransformerEmbedder metadata saved to %s", p)

    def load(self, path: str | Path) -> "SentenceTransformerEmbedder":
        """Load from saved metadata (re-downloads model weights if needed)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model metadata not found: {p}")
        with p.open("rb") as fh:
            meta = pickle.load(fh)  # noqa: S301

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError("sentence-transformers required") from exc

        self._model_name = meta["model_name"]
        self._device = meta.get("device")
        self._model = SentenceTransformer(self._model_name, device=self._device)
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.debug("SentenceTransformerEmbedder loaded from %s", p)
        return self


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def get_default_embedder() -> EmbeddingModel:
    """
    Return the best available embedding model.

    Tries sentence-transformers first; falls back to TF-IDF.
    """
    try:
        embedder = SentenceTransformerEmbedder()
        logger.info("Using SentenceTransformerEmbedder (%s)", embedder._model_name)
        return embedder
    except ImportError:
        logger.info("sentence-transformers not available. Using TFIDFEmbedder.")
        return TFIDFEmbedder()
