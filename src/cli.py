"""
Command-line interface for the RAG (Retrieval Augmented Generation) data pipeline.

Run from the repository root::

    .venv/bin/python src/cli.py --help

Ensures ``src/`` is on ``sys.path`` so intra-project imports resolve whether the
script is launched from ``src/`` or from the repo root.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import click

import config
from chunk_corpus import build_chunks


# --- Config -----------------------------------------------------------------

PROJECT_ROOT = _SRC.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_ANSWER_FILE = RESULTS_DIR / "output.txt"

EXPECTED_PDFS: dict[str, str] = {
    "IBM watsonx.data version 2.0.3.pdf":
        "https://www.ibm.com/support/pages/system/files/inline-files/"
        "IBM%20watsonx.data%20version%202.0.3.pdf",
    "watson-assistant.pdf":
        "https://cloud.ibm.com/media/docs/pdf/watson-assistant/watson-assistant.pdf",
}

PROVIDER_CHOICES = ("gemini", "ibm")
PROMPT_VERSION_CHOICES = ("grounded", "plain")


# --- CLI group --------------------------------------------------------------

@click.group()
def main():
    """DocAssist RAG pipeline commands."""


# --- Ingestion / indexing ---------------------------------------------------

@main.command("ingest-html")
def ingest_html():
    """Convert ``data/raw/text/*.txt`` (URL + HTML) into ``data/raw/json/*.json``."""
    import extract_html_txt_to_json as extract
    extract.run()


@main.command("build-chunks")
def cmd_build_chunks():
    """Rebuild ``data/processed/chunks.jsonl`` from JSON and PDF inputs."""
    config.RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_expected_pdfs(EXPECTED_PDFS, config.RAW_PDF_DIR)
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

    shutil.copy2(pdf_path, dest)
    click.echo(f"Copied to {dest}. Run: python src/cli.py build-chunks")


# --- Ask command ------------------------------------------------------------

@main.command("ask")
@click.argument("question", nargs=-1, required=True)
@click.option("--top-k", default=5, show_default=True, type=int,
              help="Number of chunks to retrieve from Chroma.")
@click.option("--prompt-version", "prompt_version",
              type=click.Choice(PROMPT_VERSION_CHOICES, case_sensitive=False),
              default="grounded", show_default=True,
              help="Grounded = cite-only from context; plain = lighter instructions.")
@click.option("--show-context/--no-show-context", default=False,
              help="Print retrieved titles, distances, and a short text preview before the answer.")
@click.option("--no-hybrid", is_flag=True, default=False,
              help="Dense Chroma retrieval only (skip BM25 + RRF).")
@click.option("--no-rerank", is_flag=True, default=False,
              help="Skip cross-encoder rerank; use fused dense+BM25 order only.")
@click.option("--provider", type=click.Choice(PROVIDER_CHOICES, case_sensitive=False),
              default="gemini", show_default=True,
              help="LLM provider to use for generating the answer.")
@click.option("--output", "output_path", type=click.Path(path_type=Path),
              default=DEFAULT_ANSWER_FILE, show_default=True,
              help="File to write the answer to. Pass an empty string to skip writing.")
def cmd_ask(
    question: tuple[str, ...],
    top_k: int,
    prompt_version: str,
    show_context: bool,
    no_hybrid: bool,
    no_rerank: bool,
    provider: str,
    output_path: Path,
):
    """
    Retrieve TOP_K chunks for QUESTION, then print an LLM answer.

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
        collection, [q],
        top_k=top_k,
        hybrid=not no_hybrid,
        rerank=not no_rerank,
    )
    chunks = formatted[0]["top_chunks"]

    if not chunks:
        click.echo("(No chunks retrieved; index the corpus or broaden the question.)", err=True)

    if show_context:
        _print_context(chunks)

    try:
        answer = generate_answer(
            q, chunks,
            prompt_version=prompt_version.lower(),
            provider=provider.lower(),
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(answer)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(answer)
        click.echo(f"Answer saved to {output_path}")


# --- Helpers ----------------------------------------------------------------

def _ensure_expected_pdfs(pdfs: dict[str, str], dest_dir: Path) -> None:
    """Prompt to download any expected PDFs missing from ``dest_dir``."""
    import urllib.request

    for filename, url in pdfs.items():
        pdf_path = dest_dir / filename
        if pdf_path.exists():
            continue
        if not click.confirm(f"Missing '{filename}'. Download it now?"):
            continue
        click.echo(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, pdf_path)
            click.echo(f"Successfully downloaded {filename}.")
        except Exception as exc:  # noqa: BLE001 — surface any download failure
            click.echo(f"Failed to download {filename}: {exc}", err=True)


def _print_context(chunks: list[dict]) -> None:
    """Print retrieved chunks with their scores and a short preview."""
    click.echo("--- Retrieved context ---\n")
    for i, c in enumerate(chunks, 1):
        title = c.get("title", "?")
        score_bits = _format_scores(c)
        click.echo(f"{i}. {title}  ({score_bits})")

        text = (c.get("text") or "").strip()
        if text:
            preview = text[:400] + ("…" if len(text) > 400 else "")
            click.echo(preview)
        click.echo()


def _format_scores(chunk: dict) -> str:
    """Return a 'distance=..., rerank=...' string for display."""
    parts = [f"distance={chunk.get('distance')}"]
    rerank = chunk.get("rerank_score")
    if isinstance(rerank, (int, float)):
        parts.append(f"rerank={rerank:.3f}")
    return ", ".join(parts)


if __name__ == "__main__":
    main()