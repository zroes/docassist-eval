"""
Command-line interface for the RAG data pipeline.

Intended to be run from the repository root, for example::

    .venv/bin/python src/cli.py --help

Ensures ``src/`` is on ``sys.path`` so imports resolve the same way as when scripts
are launched from inside ``src/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import click

import config
from chunk_corpus import build_chunks


@click.group()
def main():
    """DocAssist RAG pipeline commands."""
    pass


@main.command("ingest-html")
def ingest_html():
    """Convert ``data/raw/text/*.txt`` (URL + HTML) into ``data/raw/json/*.json``."""
    import extract_html_txt_to_json as extract

    extract.run()


@main.command("build-chunks")
def cmd_build_chunks():
    """Rebuild ``data/processed/chunks.jsonl`` from JSON and PDF inputs."""
    build_chunks()


@main.command("index")
def cmd_index():
    """Load ``chunks.jsonl`` into Chroma (replaces the ``rag_corpus`` collection)."""
    import index_chroma

    index_chroma.main()


@main.command("ingest-pdf")
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
def ingest_pdf(pdf_path: Path):
    """
    Copy a PDF into ``data/raw/pdf/`` so the next ``build-chunks`` run includes it.

    Args:
        pdf_path: Path to an existing PDF file on disk.
    """
    config.RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.RAW_PDF_DIR / pdf_path.name
    if dest.resolve() == pdf_path.resolve():
        click.echo(f"PDF already at {dest}")
        return
    if dest.exists():
        raise click.ClickException(f"Target exists: {dest}")
    dest.write_bytes(pdf_path.read_bytes())
    click.echo(f"Copied to {dest}. Run: python src/cli.py build-chunks")


@main.command("ask")
@click.argument("question", nargs=-1, required=True)
@click.option("--top-k", default=5, show_default=True, type=int, help="Number of chunks to retrieve from Chroma.")
@click.option(
    "--prompt-version",
    "prompt_version",
    type=click.Choice(["grounded", "plain"], case_sensitive=False),
    default="grounded",
    show_default=True,
    help="Grounded = cite-only from context; plain = lighter instructions.",
)
@click.option(
    "--show-context/--no-show-context",
    default=False,
    help="Print retrieved titles, distances, and a short text preview before the answer.",
)
@click.option(
    "--no-hybrid",
    is_flag=True,
    default=False,
    help="Dense Chroma retrieval only (skip BM25 + RRF when combined with defaults).",
)
@click.option(
    "--no-rerank",
    is_flag=True,
    default=False,
    help="Skip cross-encoder rerank; use fused dense+BM25 order only.",
)
def cmd_ask(
    question: tuple[str, ...],
    top_k: int,
    prompt_version: str,
    show_context: bool,
    no_hybrid: bool,
    no_rerank: bool,
):
    """
    Retrieve TOP_K chunks for QUESTION, then print a Gemini answer (needs GEMINI_API_KEY).

    Example::

        export GEMINI_API_KEY=...
        python src/cli.py ask "What is RAG used for?"
    """
    from generate import generate_answer
    from retrieve import get_chroma_collection, retrieve_for_grounding

    q = " ".join(question).strip()
    if not q:
        raise click.ClickException("Question is empty.")

    collection = get_chroma_collection()
    formatted = retrieve_for_grounding(
        collection,
        [q],
        top_k=top_k,
        hybrid=not no_hybrid,
        rerank=not no_rerank,
    )
    chunks = formatted[0]["top_chunks"]

    if not chunks:
        click.echo("(No chunks retrieved; index the corpus or broaden the question.)", err=True)

    if show_context:
        click.echo("--- Retrieved context ---\n")
        for i, c in enumerate(chunks, 1):
            title = c.get("title", "?")
            dist = c.get("distance")
            rs = c.get("rerank_score")
            score_bits = f"distance={dist}" + (f", rerank={rs:.3f}" if isinstance(rs, (int, float)) else "")
            text = (c.get("text") or "").strip()
            preview = text[:400] + ("…" if len(text) > 400 else "")
            click.echo(f"{i}. {title}  ({score_bits})")
            if preview:
                click.echo(preview)
            click.echo()

    try:
        answer = generate_answer(q, chunks, prompt_version=prompt_version.lower())
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(answer)


if __name__ == "__main__":
    main()
