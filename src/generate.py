"""
Prompt construction and LLM calls for RAG answers (Gemini).

Takes retrieval results from :mod:`retrieve`, builds a grounded or plain prompt with
optional page hints, and returns model text.
"""

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import google.genai as genai


def build_prompt(
    question: str,
    retrieved_chunks: list[dict],
    prompt_version: str = "grounded",
    distance_threshold: float = 1.5,
) -> str:
    """
    Assemble an LLM prompt from the user question and retrieved chunk dicts.

    Chunks whose ``distance`` is greater than or equal to ``distance_threshold`` are
    skipped (treated as not relevant enough for grounded mode).

    Args:
        question: End-user question string.
        retrieved_chunks: List of dicts with ``title``, ``source``, ``text``, optional
            ``distance``, ``page_start``, ``page_end`` (as from :func:`retrieve.format_retrieval_results`).
        prompt_version: ``"grounded"`` (strict cite-from-context) or ``"plain"`` (lighter instructions).
        distance_threshold: Maximum Chroma distance to include a chunk (lower is closer).

    Returns:
        A single string prompt ready for :func:`call_llm`.
    """
    context_list = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        if chunk.get("distance", 10) >= distance_threshold:
            continue
        try:
            page_start = int(chunk.get("page_start", -1))
        except (TypeError, ValueError):
            page_start = -1
        page_line = ""
        if page_start > 0:
            try:
                page_end = int(chunk.get("page_end", page_start))
            except (TypeError, ValueError):
                page_end = page_start
            page_line = f"Page: {page_start}" + (f"–{page_end}" if page_end != page_start else "") + "\n"
        chunk_block = (
            f"--- Document {i} ---\n"
            f"Title: {chunk.get('title', 'Unknown')}\n"
            f"Source: {chunk.get('source', 'Unknown')}\n"
            f"{page_line}"
            f"Content: {chunk.get('text', '')}\n"
        )
        context_list.append(chunk_block)

    context_string = "\n".join(context_list)

    if prompt_version == "grounded":
        final_prompt = f"""
        You are a highly reliable, enterprise-grade AI assistant operating in a retrieval-augmented generation (RAG) setting.

        Your task is to answer the user's question using ONLY the information provided in the context below.

        STRICT RULES:
        1. Do NOT use any prior knowledge, assumptions, or external information.
        2. Only rely on the provided documents.
        3. If the answer cannot be found explicitly or inferred directly from the context, respond exactly with:
        "I don't know based on the provided documents."
        4. Do NOT hallucinate, fabricate, or guess.
        5. Be concise and precise. Avoid unnecessary elaboration.
        6. Always cite the document titles used in your answer.
        - Use clear inline citations like: (Title: <document title>)
        - If multiple documents are used, cite each relevant title.

        CONTEXT:
        {context_string}

        USER QUESTION:
        {question}

        ANSWER:
        """

    else:
        final_prompt = f"""
            You are a helpful assistant. Answer the user's question using the provided context.
            Context:
            {context_string}

            Question: {question}
            """

    print(final_prompt)
    return final_prompt


def call_llm(prompt: str, provider: str = "gemini") -> str:
    """
    Send a completed prompt string to a configured model provider.

    Args:
        prompt: Full prompt text from :func:`build_prompt`.
        provider: Currently only ``"gemini"`` is implemented.

    Returns:
        Model response text (first completion).

    Raises:
        ValueError: If ``GEMINI_API_KEY`` is unset when using Gemini.
    """
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("No Gemini API key found!")

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text

    raise ValueError(f"Unsupported provider: {provider}")


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    prompt_version: str = "grounded",
    provider: str = "gemini",
) -> str:
    """
    End-to-end: build prompt from chunks and return the model answer.

    Args:
        question: User question.
        retrieved_chunks: Formatted chunks for one query (not the full batch wrapper).
        prompt_version: ``"grounded"`` or ``"plain"``.
        provider: LLM provider id for :func:`call_llm`.

    Returns:
        Answer string from the model.
    """
    final_prompt = build_prompt(question, retrieved_chunks, prompt_version=prompt_version)
    answer = call_llm(final_prompt, provider)
    return answer


if __name__ == "__main__":
    mock_question = "How do I chat with a model in Python?"
    mock_chunks = [
        {
            "title": "Quick code tutorial: Chat with a model",
            "source": "https://dataplatform.cloud.ibm.com/docs/chat",
            "text": "You can use the ibm_watsonx_ai library to interact with foundation models. You pass a list of messages with 'role' and 'content'.",
        }
    ]

    print("Sending to Gemini...")
    final_answer = generate_answer(mock_question, mock_chunks, prompt_version="grounded", provider="gemini")

    print("\n--- FINAL ANSWER ---")
    print(final_answer)
