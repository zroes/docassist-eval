"""
Command-line interface for the RAG (Retrieval Augmented Generation) data pipeline.

This script is intended to be run from the repository root. For example, to see available commands, you would run:

    .venv/bin/python src/cli.py --help

It ensures that the `src/` directory is added to `sys.path` so that Python imports within the project resolve correctly,
whether scripts are launched from the `src/` directory itself or from the repository root.
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
    pass # No-op: This group merely serves as a container for subcommands.


@main.command("ingest-html")
def ingest_html():
    """Convert ``data/raw/text/*.txt`` (URL + HTML) into ``data/raw/json/*.json``."""
    # Lazily import the module to avoid circular dependencies and unnecessary imports when not using this command.
    import extract_html_txt_to_json as extract

    # Execute the main function within the extract_html_txt_to_json module.
    extract.run()


@main.command("build-chunks")
def cmd_build_chunks():
    """Rebuild ``data/processed/chunks.jsonl`` from JSON and PDF inputs."""
    # Lazily import urllib.request for downloading PDFs.
    import urllib.request
    
    # Define a dictionary of expected PDF files and their corresponding download URLs.
    expected_pdfs = {
        "IBM watsonx.data version 2.0.3.pdf": "https://www.ibm.com/support/pages/system/files/inline-files/IBM%20watsonx.data%20version%202.0.3.pdf",
        "watson-assistant.pdf": "https://cloud.ibm.com/media/docs/pdf/watson-assistant/watson-assistant.pdf"
    }
    
    # Ensure the raw PDF directory exists, creating it and any necessary parent directories if they don't.
    config.RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    
    # Iterate through the expected PDFs.
    for filename, url in expected_pdfs.items():
        pdf_path = config.RAW_PDF_DIR / filename
        # If a PDF file is missing, prompt the user to download it.
        if not pdf_path.exists():
            if click.confirm(f"Missing \'{filename}\'. Download it now?"):
                click.echo(f"Downloading {filename}...")
                try:
                    # Attempt to download the PDF from the given URL.
                    urllib.request.urlretrieve(url, pdf_path)
                    click.echo(f"Successfully downloaded {filename}.")
                except Exception as e:
                    # Report any errors during download.
                    click.echo(f"Failed to download {filename}: {e}", err=True)

    # Call the build_chunks function to process all input documents (JSON and downloaded PDFs) into chunks.
    build_chunks()


@main.command("index")
def cmd_index():
    """Load ``chunks.jsonl`` into Chroma (replaces the ``rag_corpus`` collection)."""
    # Lazily import the module to avoid unnecessary imports.
    import index_chroma

    # Execute the main indexing function.
    index_chroma.main()


@main.command("ingest-pdf")
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
def ingest_pdf(pdf_path: Path):
    """
    Copy a PDF into ``data/raw/pdf/`` so the next ``build-chunks`` run includes it.

    Args:
        pdf_path: Path to an existing PDF file on disk.
    """
    # Ensure the raw PDF directory exists, creating it and any necessary parent directories.
    config.RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    # Define the destination path for the PDF within the raw PDF directory.
    dest = config.RAW_PDF_DIR / pdf_path.name

    # Check if the source PDF is already at the destination.
    if dest.resolve() == pdf_path.resolve():
        click.echo(f"PDF already at {dest}")
        return
    # If a file with the same name already exists at the destination, raise an error to prevent overwriting.
    if dest.exists():
        raise click.ClickException(f"Target exists: {dest}")

    # Copy the PDF file by reading its bytes and writing them to the destination.
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
    # Lazily import necessary functions to avoid unnecessary imports until this command is called.
    from generate import generate_answer
    from retrieve import get_chroma_collection, retrieve_for_grounding

    # Join the question arguments into a single string and remove leading/trailing whitespace.
    q = " ".join(question).strip()
    # If the question is empty after stripping, raise an error.
    if not q:
        raise click.ClickException("Question is empty.")

    # Get the ChromaDB collection (vector store).
    collection = get_chroma_collection()
    # Retrieve relevant chunks for the question from the collection.
    # Options for hybrid retrieval (dense + BM25) and reranking are controlled by command-line flags.
    formatted = retrieve_for_grounding(
        collection,
        [q],
        top_k=top_k,
        hybrid=not no_hybrid,  # Enable hybrid retrieval unless --no-hybrid flag is used.
        rerank=not no_rerank,  # Enable reranking unless --no-rerank flag is used.
    )
    # Extract the top chunks from the retrieval results.
    chunks = formatted[0]["top_chunks"]

    # If no chunks were retrieved, inform the user.
    if not chunks:
        click.echo("(No chunks retrieved; index the corpus or broaden the question.)", err=True)

    # If --show-context flag is enabled, print details about the retrieved chunks.
    if show_context:
        click.echo("--- Retrieved context ---\n")
        for i, c in enumerate(chunks, 1):
            title = c.get("title", "?")  # Get the title of the chunk.
            dist = c.get("distance")  # Get the distance score from the dense retrieval.
            rs = c.get("rerank_score")  # Get the reranking score if available.
            # Format the score bits for display.
            score_bits = f"distance={dist}" + (f", rerank={rs:.3f}" if isinstance(rs, (int, float)) else "")
            text = (c.get("text") or "").strip()  # Get the text content of the chunk.
            # Create a short preview of the chunk text.
            preview = text[:400] + ("…" if len(text) > 400 else "")
            click.echo(f"{i}. {title}  ({score_bits})")
            if preview:
                click.echo(preview)
            click.echo()

    # Generate an answer using the retrieved chunks and the specified prompt version.
    try:
        answer = generate_answer(q, chunks, prompt_version=prompt_version.lower())
    except ValueError as exc:
        # If an error occurs during answer generation, raise a ClickException.
        raise click.ClickException(str(exc)) from exc

    # Print the generated answer to the console.
    click.echo(answer)

    # Save the answer to a file in the results directory.
    output_file_path = Path("../results/output.txt")
    # Ensure the parent directory for the output file exists.
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    # Write the answer text to the file.
    output_file_path.write_text(answer)
    click.echo(f"Answer saved to {output_file_path}")


if __name__ == "__main__":
    # Entry point for the CLI. This ensures that `main()` is called when the script is executed.
    main()
