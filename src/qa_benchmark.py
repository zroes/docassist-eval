"""
Batch evaluation: retrieve over a fixed question set, generate plain vs grounded answers, CSV out.

Uses :mod:`retrieve` for Chroma access and :mod:`generate` for Gemini calls. Requires
``GEMINI_API_KEY`` in the environment for each generation call.
"""

import csv
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import config
from generate import generate_answer
from retrieve import get_chroma_collection, retrieve_for_grounding


def save_to_csv(data: list[dict], filepath=None) -> None:
    """
    Write benchmark rows to CSV (UTF-8, newline-safe for Excel).

    Args:
        data: List of dicts with keys ``question``, ``prompt_version``, ``retrieved_docs``, ``answer``.
        filepath: Output path; defaults to ``config.RESULTS_DIR / "prompt_comparison.csv"``.

    Returns:
        None.
    """
    filepath = filepath or (config.RESULTS_DIR / "prompt_comparison.csv")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["question", "prompt_version", "retrieved_docs", "answer"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def run_benchmark() -> None:
    """
    Run embedded benchmark questions: retrieve, then two answers per question (plain + grounded).

    Returns:
        None. Appends rows to the results CSV and prints progress.
    """
    benchmark_questions = [
        "What are the main components and steps to implement a Retrieval-Augmented Generation (RAG) pattern solution?",
        "What are the key stages in the implementation workflow for a generative AI solution?",
        "What is the difference between working with tools versus writing custom code for foundation models?",
        "How do I interact with foundation models in a conversational format using Python?",
        "Is it possible to stream the generated text output instead of waiting for a single response? How do I do it?",
        "How can I create text embeddings programmatically using the API?",
        "How can I retrieve a list of supported foundation models and find their model IDs?",
        "How do I bake a classic chocolate chip cookie?",
        "What is the capital of France?",
        "Who won the Super Bowl in 2024?",
    ]

    collection = get_chroma_collection()
    results_list = []
    formatted_chunks = retrieve_for_grounding(collection, benchmark_questions, top_k=8)

    n = len(benchmark_questions)
    for i, question in enumerate(benchmark_questions):
        print(f"Processing question {i + 1}/{n}...")

        chunks_data = formatted_chunks[i].get("top_chunks", [])

        titles = [chunk.get("title") for chunk in chunks_data]
        titles_string = ", ".join(titles) if titles else "No docs retrieved (Filtered by distance)"

        plain_answer = generate_answer(
            question=question,
            retrieved_chunks=chunks_data,
            prompt_version="plain",
        )

        results_list.append(
            {
                "question": question,
                "prompt_version": "plain",
                "retrieved_docs": titles_string,
                "answer": plain_answer,
            }
        )

        grounded_answer = generate_answer(
            question=question,
            retrieved_chunks=chunks_data,
            prompt_version="grounded",
        )

        results_list.append(
            {
                "question": question,
                "prompt_version": "grounded",
                "retrieved_docs": titles_string,
                "answer": grounded_answer,
            }
        )

    save_to_csv(results_list)
    print(f"\nBenchmark complete! Wrote {config.RESULTS_DIR / 'prompt_comparison.csv'}")


if __name__ == "__main__":
    run_benchmark()
rompt_comparison.csv'}")


if __name__ == "__main__":
    run_benchmark()
