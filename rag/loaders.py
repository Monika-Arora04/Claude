"""
Document loaders for various file formats used in utility/sensor contexts.

Supports:
- Plain text files (.txt)
- CSV files (sensor logs with timestamps, values, units)
- JSON files
- PDF files (PyPDF2 if available, fallback to text)
- Log files (.log)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a loaded document with content and metadata."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure required metadata keys exist
        self.metadata.setdefault("source", "unknown")
        self.metadata.setdefault("type", "unknown")
        self.metadata.setdefault("page", 0)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def doc_type(self) -> str:
        return self.metadata.get("type", "unknown")

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return f"Document(source={self.source!r}, type={self.doc_type!r}, content={preview!r}...)"


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------


def _load_text(path: Path) -> list[Document]:
    """Load a plain text file."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise IOError(f"Cannot read text file {path}: {exc}") from exc

    return [
        Document(
            content=content,
            metadata={
                "source": str(path),
                "type": "text",
                "page": 0,
                "filename": path.name,
                "size_bytes": path.stat().st_size,
            },
        )
    ]


def _load_csv(path: Path) -> list[Document]:
    """
    Load a CSV file (sensor logs format).

    Each row is treated as a record; rows are grouped into logical batches
    for context-aware chunking later. Metadata captures column names,
    row count, and any detected timestamp/unit patterns.
    """
    try:
        rows: list[dict[str, str]] = []
        with path.open(encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames: list[str] = list(reader.fieldnames or [])
            for row in reader:
                rows.append(dict(row))
    except OSError as exc:
        raise IOError(f"Cannot read CSV file {path}: {exc}") from exc

    if not rows:
        return [
            Document(
                content="(empty CSV file)",
                metadata={"source": str(path), "type": "csv", "page": 0},
            )
        ]

    # Build human-readable representation
    lines: list[str] = []
    lines.append(f"CSV Data from: {path.name}")
    lines.append(f"Columns: {', '.join(fieldnames)}")
    lines.append(f"Total rows: {len(rows)}")
    lines.append("")

    # Format rows as readable text, preserving structured context
    for i, row in enumerate(rows):
        parts = [f"{k}={v}" for k, v in row.items() if v.strip()]
        lines.append(f"Row {i + 1}: {' | '.join(parts)}")

    content = "\n".join(lines)

    # Detect sensor-like columns
    timestamp_cols = [c for c in fieldnames if re.search(r"time|date|ts|timestamp", c, re.I)]
    value_cols = [c for c in fieldnames if re.search(r"value|reading|measure|sensor|temp|press|hum|volt|curr", c, re.I)]
    unit_cols = [c for c in fieldnames if re.search(r"unit|uom|measure_unit", c, re.I)]

    return [
        Document(
            content=content,
            metadata={
                "source": str(path),
                "type": "csv",
                "page": 0,
                "filename": path.name,
                "row_count": len(rows),
                "columns": fieldnames,
                "timestamp_columns": timestamp_cols,
                "value_columns": value_cols,
                "unit_columns": unit_cols,
                "raw_rows": rows,  # kept for structured queries
            },
        )
    ]


def _load_json(path: Path) -> list[Document]:
    """Load a JSON file and convert to readable text."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(text)
    except OSError as exc:
        raise IOError(f"Cannot read JSON file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    # Pretty-print for embedding
    pretty = json.dumps(data, indent=2, ensure_ascii=False, default=str)

    # Build summary lines for top-level structure
    summary_lines: list[str] = [f"JSON Data from: {path.name}"]
    if isinstance(data, dict):
        summary_lines.append(f"Top-level keys: {', '.join(str(k) for k in data.keys())}")
    elif isinstance(data, list):
        summary_lines.append(f"Array with {len(data)} items")

    summary = "\n".join(summary_lines)
    content = f"{summary}\n\n{pretty}"

    return [
        Document(
            content=content,
            metadata={
                "source": str(path),
                "type": "json",
                "page": 0,
                "filename": path.name,
                "data_type": type(data).__name__,
                "raw_data": data,
            },
        )
    ]


def _load_pdf(path: Path) -> list[Document]:
    """
    Load a PDF file.  Uses PyPDF2 when available; falls back to reading
    raw bytes and extracting printable text.
    """
    try:
        import PyPDF2  # type: ignore

        documents: list[Document] = []
        with path.open("rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            num_pages = len(reader.pages)
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                except Exception:  # noqa: BLE001
                    text = ""
                if text.strip():
                    documents.append(
                        Document(
                            content=text,
                            metadata={
                                "source": str(path),
                                "type": "pdf",
                                "page": page_num,
                                "total_pages": num_pages,
                                "filename": path.name,
                            },
                        )
                    )
        if not documents:
            documents.append(
                Document(
                    content=f"(PDF file {path.name} — no extractable text)",
                    metadata={"source": str(path), "type": "pdf", "page": 0},
                )
            )
        return documents

    except ImportError:
        logger.warning(
            "PyPDF2 not installed. Falling back to raw text extraction for %s. "
            "Install with: pip install PyPDF2",
            path,
        )
        # Fallback: read bytes, strip non-printable characters
        raw = path.read_bytes()
        printable = re.sub(rb"[^\x20-\x7e\n\t]", b" ", raw)
        text = printable.decode("ascii", errors="replace")
        # Remove excessive whitespace
        text = re.sub(r" {3,}", "  ", text)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        return [
            Document(
                content=text[:50_000],  # cap at 50k chars for the fallback
                metadata={
                    "source": str(path),
                    "type": "pdf",
                    "page": 0,
                    "filename": path.name,
                    "extraction_method": "fallback",
                },
            )
        ]


def _load_log(path: Path) -> list[Document]:
    """
    Load a log file.

    Preserves log structure (timestamps, severity levels, messages) and
    adds metadata about detected log patterns.
    """
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise IOError(f"Cannot read log file {path}: {exc}") from exc

    lines = content.splitlines()

    # Detect log patterns
    severity_pattern = re.compile(
        r"\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL|ALERT|NOTICE|SEVERE)\b",
        re.I,
    )
    timestamp_pattern = re.compile(
        r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
        r"|\b\d{2}/\d{2}/\d{4} \d{2}:\d{2}"
        r"|\b\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"
    )

    severity_counts: dict[str, int] = {}
    for line in lines:
        m = severity_pattern.search(line)
        if m:
            sev = m.group(1).upper()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

    has_timestamps = bool(timestamp_pattern.search(content[:2000]))

    # Prepend summary header
    header_lines = [
        f"Log file: {path.name}",
        f"Total lines: {len(lines)}",
    ]
    if severity_counts:
        sev_summary = ", ".join(f"{k}={v}" for k, v in sorted(severity_counts.items()))
        header_lines.append(f"Severity counts: {sev_summary}")
    if has_timestamps:
        header_lines.append("Timestamps: detected")
    header_lines.append("")

    full_content = "\n".join(header_lines) + content

    return [
        Document(
            content=full_content,
            metadata={
                "source": str(path),
                "type": "log",
                "page": 0,
                "filename": path.name,
                "line_count": len(lines),
                "severity_counts": severity_counts,
                "has_timestamps": has_timestamps,
            },
        )
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Map extensions to loader functions
_LOADERS: dict[str, Any] = {
    ".txt": _load_text,
    ".csv": _load_csv,
    ".json": _load_json,
    ".jsonl": _load_json,
    ".pdf": _load_pdf,
    ".log": _load_log,
    ".md": _load_text,
    ".rst": _load_text,
    ".xml": _load_text,
    ".yaml": _load_text,
    ".yml": _load_text,
    ".ini": _load_text,
    ".conf": _load_text,
    ".cfg": _load_text,
    ".tsv": _load_csv,
}


def load_document(path: str | Path) -> list[Document]:
    """
    Load a single document file and return a list of Document objects.

    For most formats this returns a single Document; PDFs may return one
    Document per page.

    Args:
        path: Path to the document file.

    Returns:
        List of Document objects extracted from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
        IOError: If the file cannot be read.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {p}")

    ext = p.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        # Try as plain text for unknown extensions
        logger.warning(
            "Unknown file extension %r for %s — attempting plain text load.", ext, p
        )
        return _load_text(p)

    logger.debug("Loading %s as type %s", p, ext)
    return loader(p)


def load_directory(
    directory: str | Path,
    extensions: list[str] | None = None,
    recursive: bool = True,
) -> list[Document]:
    """
    Load all supported documents from a directory.

    Args:
        directory: Path to the directory.
        extensions: Optional list of extensions to filter (e.g. ['.txt', '.csv']).
                    Defaults to all supported extensions.
        recursive: Whether to search subdirectories (default True).

    Returns:
        List of all Document objects loaded from the directory.
    """
    d = Path(directory)
    if not d.exists():
        raise FileNotFoundError(f"Directory not found: {d}")
    if not d.is_dir():
        raise ValueError(f"Path is not a directory: {d}")

    allowed_exts = set(extensions) if extensions else set(_LOADERS.keys())
    # Normalize to lowercase with leading dot
    allowed_exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in allowed_exts}

    glob_pattern = "**/*" if recursive else "*"
    all_files = sorted(d.glob(glob_pattern))

    documents: list[Document] = []
    errors: list[str] = []

    for fp in all_files:
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in allowed_exts:
            continue
        try:
            docs = load_document(fp)
            documents.extend(docs)
            logger.info("Loaded %d document(s) from %s", len(docs), fp)
        except Exception as exc:  # noqa: BLE001
            error_msg = f"Failed to load {fp}: {exc}"
            logger.warning(error_msg)
            errors.append(error_msg)

    if errors:
        logger.warning("%d file(s) failed to load.", len(errors))

    logger.info("Total documents loaded: %d", len(documents))
    return documents


def supported_extensions() -> list[str]:
    """Return the list of supported file extensions."""
    return sorted(_LOADERS.keys())
