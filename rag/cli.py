"""
CLI interface for the RAG system.

Commands:
  index       — Index a directory (or list of files) into a persistent store
  query       — Ask a single question against the index
  interactive — Start an interactive chat session
  stats       — Show index statistics

Usage examples:
  python -m rag.cli index sample_docs/ --index-path .rag_index
  python -m rag.cli query "What temperature anomalies were recorded?" --index-path .rag_index
  python -m rag.cli interactive --index-path .rag_index
  python -m rag.cli stats --index-path .rag_index
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

try:
    import click
except ImportError as exc:
    raise ImportError(
        "click is required for the CLI. Install with: pip install click"
    ) from exc

from .embeddings import TFIDFEmbedder, get_default_embedder
from .pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        format="%(levelname)s [%(name)s] %(message)s",
        level=level,
    )


# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

_index_path_option = click.option(
    "--index-path",
    "-i",
    default=".rag_index/store.pkl",
    show_default=True,
    help="Path to the persistent vector store file.",
)

_verbose_option = click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
)

_top_k_option = click.option(
    "--top-k",
    "-k",
    default=5,
    show_default=True,
    help="Number of chunks to retrieve.",
)

_model_option = click.option(
    "--model",
    "-m",
    default="claude-sonnet-4-6",
    show_default=True,
    help="Anthropic model for generation.",
)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option("1.0.0", prog_name="rag")
def cli() -> None:
    """RAG system for utility and sensor documents."""


# ---------------------------------------------------------------------------
# index command
# ---------------------------------------------------------------------------


@cli.command("index")
@click.argument("path", type=click.Path(exists=True))
@_index_path_option
@click.option(
    "--extensions",
    "-e",
    multiple=True,
    help="File extensions to include (e.g. -e .txt -e .csv). Defaults to all supported.",
)
@click.option(
    "--no-recursive",
    is_flag=True,
    default=False,
    help="Do not recurse into subdirectories.",
)
@click.option(
    "--chunk-size",
    default=512,
    show_default=True,
    help="Target chunk size in characters.",
)
@click.option(
    "--chunk-overlap",
    default=64,
    show_default=True,
    help="Character overlap between chunks.",
)
@click.option(
    "--rows-per-chunk",
    default=20,
    show_default=True,
    help="Rows per chunk for CSV/log files.",
)
@click.option(
    "--use-sentence-transformers",
    is_flag=True,
    default=False,
    help="Use sentence-transformers embedder (requires pip install sentence-transformers).",
)
@_verbose_option
def index_cmd(
    path: str,
    index_path: str,
    extensions: tuple[str, ...],
    no_recursive: bool,
    chunk_size: int,
    chunk_overlap: int,
    rows_per_chunk: int,
    use_sentence_transformers: bool,
    verbose: bool,
) -> None:
    """
    Index documents from PATH (directory or single file) into the vector store.

    PATH can be a directory (all supported files are loaded) or a single file.
    """
    _configure_logging(verbose)

    # Choose embedder
    if use_sentence_transformers:
        try:
            from .embeddings import SentenceTransformerEmbedder
            embedder = SentenceTransformerEmbedder()
            click.echo("Using SentenceTransformerEmbedder.")
        except ImportError:
            click.echo("sentence-transformers not installed. Falling back to TF-IDF.")
            embedder = TFIDFEmbedder()
    else:
        embedder = TFIDFEmbedder()
        click.echo("Using TFIDFEmbedder.")

    pipeline = RAGPipeline(
        embedder=embedder,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        rows_per_chunk=rows_per_chunk,
        index_path=index_path,
    )

    exts = list(extensions) if extensions else None
    p = Path(path)

    with click.progressbar(length=1, label="Indexing documents") as bar:
        if p.is_dir():
            n_chunks = pipeline.index_directory(
                p,
                extensions=exts,
                recursive=not no_recursive,
            )
        else:
            n_chunks = pipeline.index_files([p])
        bar.update(1)

    click.echo(f"Indexed {n_chunks} chunks from {path!r}.")
    click.echo(f"Index saved to: {index_path}")

    # Show quick stats
    s = pipeline.stats()
    click.echo(f"Files processed: {s['num_indexed_files']}")
    click.echo(f"Embedder: {s['embedder']}")


# ---------------------------------------------------------------------------
# query command
# ---------------------------------------------------------------------------


@cli.command("query")
@click.argument("question")
@_index_path_option
@_top_k_option
@_model_option
@click.option(
    "--show-context",
    is_flag=True,
    default=False,
    help="Print raw retrieved chunk content.",
)
@click.option(
    "--output-json",
    is_flag=True,
    default=False,
    help="Output result as JSON.",
)
@_verbose_option
def query_cmd(
    question: str,
    index_path: str,
    top_k: int,
    model: str,
    show_context: bool,
    output_json: bool,
    verbose: bool,
) -> None:
    """
    Ask QUESTION against the indexed documents.

    Requires an index to be created first with the 'index' command.
    Set ANTHROPIC_API_KEY for answer generation; otherwise returns retrieval results.
    """
    _configure_logging(verbose)

    ip = Path(index_path)
    if not ip.exists():
        click.echo(
            f"Index not found at {index_path!r}. "
            "Run the 'index' command first.",
            err=True,
        )
        sys.exit(1)

    embedder = TFIDFEmbedder()
    pipeline = RAGPipeline(
        embedder=embedder,
        top_k=top_k,
        model=model,
        index_path=index_path,
    )

    if not pipeline._vector_store.is_indexed:
        click.echo("Index is empty or failed to load.", err=True)
        sys.exit(1)

    answer = pipeline.query(question, top_k=top_k, include_context=show_context)

    if output_json:
        out = {
            "query": answer.query,
            "answer": answer.answer,
            "sources": answer.sources,
            "generation_used": answer.generation_used,
            "model": answer.model,
            "error": answer.error,
        }
        click.echo(json.dumps(out, indent=2, default=str))
        return

    # Human-readable output
    click.echo()
    click.echo(click.style("Question:", bold=True) + f" {answer.query}")
    click.echo()

    if answer.error:
        click.echo(click.style("Error:", fg="red") + f" {answer.error}")
    else:
        click.echo(click.style("Answer:", bold=True))
        click.echo(answer.answer)

    click.echo()
    click.echo(click.style("Sources:", bold=True))
    for src in answer.sources:
        src_name = Path(src.get("source", "?")).name
        score = src.get("score", 0.0)
        chunk_idx = src.get("chunk_index", "?")
        doc_type = src.get("type", "?")
        click.echo(f"  • {src_name}  chunk={chunk_idx}  type={doc_type}  score={score:.3f}")

    if show_context and answer.retrieved_chunks:
        click.echo()
        click.echo(click.style("Retrieved Context:", bold=True))
        for i, result in enumerate(answer.retrieved_chunks, 1):
            src_name = Path(result.source).name
            click.echo(click.style(f"\n[{i}] {src_name} (score={result.score:.3f})", fg="cyan"))
            click.echo(result.content[:500])
            if len(result.content) > 500:
                click.echo("    [... truncated ...]")

    if answer.generation_used:
        click.echo(
            click.style(f"\n[Generated by {answer.model}]", fg="green", dim=True)
        )
    else:
        click.echo(
            click.style("\n[Retrieval-only mode — set ANTHROPIC_API_KEY for generation]", fg="yellow", dim=True)
        )


# ---------------------------------------------------------------------------
# interactive command
# ---------------------------------------------------------------------------


@cli.command("interactive")
@_index_path_option
@_top_k_option
@_model_option
@_verbose_option
def interactive_cmd(
    index_path: str,
    top_k: int,
    model: str,
    verbose: bool,
) -> None:
    """
    Start an interactive chat session against the indexed documents.

    Type 'exit', 'quit', or press Ctrl-C to stop.
    Type '/help' to show available commands.
    Type '/stats' to show index statistics.
    Type '/sources' to toggle source display.
    """
    _configure_logging(verbose)

    ip = Path(index_path)
    if not ip.exists():
        click.echo(
            f"Index not found at {index_path!r}. "
            "Run the 'index' command first.",
            err=True,
        )
        sys.exit(1)

    embedder = TFIDFEmbedder()
    pipeline = RAGPipeline(
        embedder=embedder,
        top_k=top_k,
        model=model,
        index_path=index_path,
    )

    if not pipeline._vector_store.is_indexed:
        click.echo("Index is empty or failed to load.", err=True)
        sys.exit(1)

    s = pipeline.stats()
    click.echo(click.style("RAG Interactive Session", bold=True, fg="cyan"))
    click.echo(f"Index: {s['index_stats']['num_chunks']} chunks from {s['num_indexed_files']} file(s)")
    click.echo(f"Model: {model} | API: {'available' if s['api_available'] else 'not configured (retrieval-only)'}")
    click.echo("Type '/help' for commands, 'quit' to exit.\n")

    show_sources = True
    history: list[tuple[str, str]] = []

    while True:
        try:
            question = click.prompt(click.style("You", bold=True, fg="green"))
        except (KeyboardInterrupt, EOFError):
            click.echo("\nGoodbye!")
            break

        question = question.strip()
        if not question:
            continue

        # Built-in commands
        if question.lower() in ("exit", "quit", "q", ":q"):
            click.echo("Goodbye!")
            break

        if question == "/help":
            click.echo(
                "\nAvailable commands:\n"
                "  /stats   — Show index statistics\n"
                "  /sources — Toggle source display\n"
                "  /history — Show query history\n"
                "  /clear   — Clear history\n"
                "  quit     — Exit\n"
            )
            continue

        if question == "/stats":
            st = pipeline.stats()
            click.echo(json.dumps(st, indent=2, default=str))
            continue

        if question == "/sources":
            show_sources = not show_sources
            click.echo(f"Source display: {'ON' if show_sources else 'OFF'}")
            continue

        if question == "/history":
            if not history:
                click.echo("No history yet.")
            else:
                for i, (q, a) in enumerate(history, 1):
                    click.echo(f"\n[{i}] Q: {q}")
                    click.echo(f"    A: {a[:200]}{'...' if len(a) > 200 else ''}")
            continue

        if question == "/clear":
            history.clear()
            click.echo("History cleared.")
            continue

        # Process query
        click.echo(click.style("Thinking...", fg="yellow", dim=True))
        answer = pipeline.query(question, top_k=top_k)

        click.echo()
        click.echo(click.style("Assistant:", bold=True, fg="blue"))
        if answer.error:
            click.echo(click.style(f"Error: {answer.error}", fg="red"))
        else:
            click.echo(answer.answer)

        if show_sources and answer.sources:
            click.echo(click.style("\nSources:", dim=True))
            for src in answer.sources[:3]:  # show top 3
                src_name = Path(src.get("source", "?")).name
                score = src.get("score", 0.0)
                click.echo(click.style(f"  • {src_name}  score={score:.3f}", dim=True))

        click.echo()
        history.append((question, answer.answer))


# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------


@cli.command("stats")
@_index_path_option
@click.option(
    "--json-output",
    is_flag=True,
    default=False,
    help="Output statistics as JSON.",
)
@_verbose_option
def stats_cmd(index_path: str, json_output: bool, verbose: bool) -> None:
    """
    Show statistics about the current index.
    """
    _configure_logging(verbose)

    ip = Path(index_path)
    if not ip.exists():
        click.echo(
            f"Index not found at {index_path!r}. "
            "Run the 'index' command first.",
            err=True,
        )
        sys.exit(1)

    embedder = TFIDFEmbedder()
    pipeline = RAGPipeline(
        embedder=embedder,
        index_path=index_path,
    )

    s = pipeline.stats()

    if json_output:
        click.echo(json.dumps(s, indent=2, default=str))
        return

    idx = s["index_stats"]
    click.echo(click.style("RAG Index Statistics", bold=True))
    click.echo(f"  Chunks:        {idx['num_chunks']}")
    click.echo(f"  Embedding dim: {idx['embedding_dim']}")
    click.echo(f"  Indexed:       {idx['is_indexed']}")
    click.echo(f"  Embedder:      {idx['embedder']}")
    click.echo()

    click.echo(click.style("Document Types:", bold=True))
    for doc_type, count in sorted(idx["types"].items()):
        click.echo(f"  {doc_type:<15} {count} chunk(s)")
    click.echo()

    click.echo(click.style("Sources:", bold=True))
    for source, count in sorted(idx["sources"].items()):
        name = Path(source).name
        click.echo(f"  {name:<40} {count} chunk(s)")
    click.echo()

    click.echo(click.style("Pipeline Config:", bold=True))
    click.echo(f"  Model:         {s['model']}")
    click.echo(f"  API available: {s['api_available']}")
    click.echo(f"  Top-K:         {s['top_k']}")
    click.echo(f"  Dense weight:  {s['dense_weight']}")
    click.echo(f"  BM25 weight:   {s['bm25_weight']}")

    if s["indexed_files"]:
        click.echo()
        click.echo(click.style("Indexed Files:", bold=True))
        for fp in sorted(s["indexed_files"]):
            click.echo(f"  {fp}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
