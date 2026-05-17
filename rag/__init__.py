"""
RAG (Retrieval-Augmented Generation) system for utility and sensor documents.

This package provides a complete pipeline for:
- Loading documents (TXT, CSV, JSON, PDF, LOG)
- Chunking text intelligently
- Embedding with TF-IDF or sentence-transformers
- Vector storage with cosine similarity and BM25 search
- Generation using Anthropic Claude API
- CLI interface for indexing and querying
"""

from .loaders import Document, load_document, load_directory
from .chunker import TextChunker, Chunk
from .embeddings import EmbeddingModel, TFIDFEmbedder, SentenceTransformerEmbedder
from .vector_store import VectorStore
from .pipeline import RAGPipeline

__all__ = [
    "Document",
    "load_document",
    "load_directory",
    "TextChunker",
    "Chunk",
    "EmbeddingModel",
    "TFIDFEmbedder",
    "SentenceTransformerEmbedder",
    "VectorStore",
    "RAGPipeline",
]

__version__ = "1.0.0"
