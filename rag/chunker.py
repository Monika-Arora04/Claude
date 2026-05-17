"""
Smart text chunking with multiple strategies.

Strategies:
- Fixed-size chunks with overlap
- Sentence-aware splitting (avoids breaking mid-sentence)
- Structured data preservation (CSV rows, log lines)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .loaders import Document


@dataclass
class Chunk:
    """A text chunk produced by the chunker."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata.setdefault("source", "unknown")
        self.metadata.setdefault("chunk_index", 0)
        self.metadata.setdefault("type", "text")

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return f"Chunk(source={self.source!r}, idx={self.metadata.get('chunk_index')}, {preview!r}...)"


class TextChunker:
    """
    Produces text chunks from Document objects with configurable strategies.

    For structured data (CSV, logs) the chunker preserves row/line context
    rather than breaking in the middle of a record.
    """

    # Sentence boundary: end of a sentence followed by whitespace + capital letter
    _SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
        sentence_aware: bool = True,
        rows_per_chunk: int = 20,
    ) -> None:
        """
        Initialise the chunker.

        Args:
            chunk_size: Target number of characters per chunk.
            chunk_overlap: Number of characters to overlap between consecutive chunks.
            min_chunk_size: Minimum characters required to keep a chunk.
            sentence_aware: If True, avoid splitting mid-sentence for prose documents.
            rows_per_chunk: For structured data (CSV/log), how many rows/lines per chunk.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.sentence_aware = sentence_aware
        self.rows_per_chunk = rows_per_chunk

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """
        Chunk a Document into a list of Chunk objects.

        Dispatches to the appropriate strategy based on document type.
        """
        doc_type = doc.metadata.get("type", "text")

        if doc_type == "csv":
            return self._chunk_csv(doc)
        elif doc_type == "log":
            return self._chunk_log(doc)
        else:
            return self._chunk_text(doc)

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk a list of documents and return all chunks."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

    # ------------------------------------------------------------------
    # Strategy: prose / generic text
    # ------------------------------------------------------------------

    def _chunk_text(self, doc: Document) -> list[Chunk]:
        """Fixed-size chunking with optional sentence-awareness."""
        text = doc.content
        if not text.strip():
            return []

        spans = self._split_into_spans(text)
        return self._spans_to_chunks(spans, doc)

    def _split_into_spans(self, text: str) -> list[str]:
        """
        Split text into a list of character spans that respect sentence
        boundaries when possible.
        """
        if not self.sentence_aware:
            return self._fixed_split(text)

        # First split at paragraph / double-newline boundaries
        paragraphs = re.split(r"\n\s*\n", text)
        spans: list[str] = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) <= self.chunk_size:
                spans.append(para)
            else:
                # Split paragraph at sentence boundaries
                sentences = self._SENTENCE_END.split(para)
                current: list[str] = []
                current_len = 0
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if current_len + len(sent) + 1 > self.chunk_size and current:
                        spans.append(" ".join(current))
                        # Start overlap: keep last sentence(s) that fit in overlap window
                        overlap_text = " ".join(current)[-self.chunk_overlap :]
                        current = [overlap_text.strip()] if overlap_text.strip() else []
                        current_len = len(current[0]) if current else 0
                    current.append(sent)
                    current_len += len(sent) + 1
                if current:
                    spans.append(" ".join(current))

        return [s for s in spans if len(s) >= self.min_chunk_size]

    def _fixed_split(self, text: str) -> list[str]:
        """Simple fixed-size split with overlap."""
        spans: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            spans.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return [s for s in spans if len(s) >= self.min_chunk_size]

    def _spans_to_chunks(self, spans: list[str], doc: Document) -> list[Chunk]:
        """Convert text spans to Chunk objects, inheriting document metadata."""
        chunks: list[Chunk] = []
        for i, span in enumerate(spans):
            meta = dict(doc.metadata)
            meta["chunk_index"] = i
            meta["total_chunks"] = len(spans)
            meta["chunk_chars"] = len(span)
            chunks.append(Chunk(content=span, metadata=meta))
        return chunks

    # ------------------------------------------------------------------
    # Strategy: CSV structured data
    # ------------------------------------------------------------------

    def _chunk_csv(self, doc: Document) -> list[Chunk]:
        """
        Chunk CSV documents by grouping rows together.

        Each chunk contains a header context + N data rows, preserving
        the column names for better retrieval.
        """
        raw_rows: list[dict[str, str]] = doc.metadata.get("raw_rows", [])
        columns: list[str] = doc.metadata.get("columns", [])

        if not raw_rows:
            # Fall back to text chunking if no structured data
            return self._chunk_text(doc)

        header_line = f"Columns: {', '.join(columns)}"
        source_line = f"Source: {doc.metadata.get('filename', doc.source)}"

        chunks: list[Chunk] = []
        total_rows = len(raw_rows)

        for batch_start in range(0, total_rows, self.rows_per_chunk):
            batch = raw_rows[batch_start : batch_start + self.rows_per_chunk]
            batch_end = min(batch_start + self.rows_per_chunk, total_rows)

            lines: list[str] = [
                source_line,
                header_line,
                f"Rows {batch_start + 1}–{batch_end} of {total_rows}:",
                "",
            ]
            for row_num, row in enumerate(batch, start=batch_start + 1):
                parts = [f"{k}: {v}" for k, v in row.items() if str(v).strip()]
                lines.append(f"  Row {row_num}: {' | '.join(parts)}")

            content = "\n".join(lines)

            meta = dict(doc.metadata)
            meta.pop("raw_rows", None)  # don't duplicate large data
            meta["chunk_index"] = len(chunks)
            meta["row_start"] = batch_start
            meta["row_end"] = batch_end
            meta["rows_in_chunk"] = len(batch)

            chunks.append(Chunk(content=content, metadata=meta))

        # Update total_chunks
        for c in chunks:
            c.metadata["total_chunks"] = len(chunks)

        return chunks

    # ------------------------------------------------------------------
    # Strategy: log files
    # ------------------------------------------------------------------

    def _chunk_log(self, doc: Document) -> list[Chunk]:
        """
        Chunk log files by grouping log lines together.

        Preserves the timestamp + severity + message structure within
        each chunk and includes a summary header.
        """
        text = doc.content
        lines = text.splitlines()

        # Skip the header lines we prepended during loading
        header_end = 0
        for i, line in enumerate(lines):
            if line.strip() == "" and i > 0:
                header_end = i + 1
                break
        header_block = "\n".join(lines[:header_end])
        log_lines = lines[header_end:]

        if not log_lines:
            return self._chunk_text(doc)

        chunks: list[Chunk] = []
        total_lines = len(log_lines)

        for batch_start in range(0, total_lines, self.rows_per_chunk):
            batch = log_lines[batch_start : batch_start + self.rows_per_chunk]
            batch_end = min(batch_start + self.rows_per_chunk, total_lines)

            # Include the file header in every chunk for context
            content_parts = []
            if header_block:
                content_parts.append(header_block)
            content_parts.append(f"Log lines {batch_start + 1}–{batch_end} of {total_lines}:")
            content_parts.append("")
            content_parts.extend(batch)

            content = "\n".join(content_parts)

            meta = dict(doc.metadata)
            meta["chunk_index"] = len(chunks)
            meta["line_start"] = batch_start
            meta["line_end"] = batch_end
            meta["lines_in_chunk"] = len(batch)

            chunks.append(Chunk(content=content, metadata=meta))

        for c in chunks:
            c.metadata["total_chunks"] = len(chunks)

        return chunks
