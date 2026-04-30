"""
Prompt construction and LLM calls for RAG answers.

Takes retrieval results from :mod:`retrieve`, builds a grounded or plain prompt
with optional page hints, and returns model text from Gemini or IBM watsonx.
"""

import json
import os
import sys
from pathlib import Path
from textwrap import dedent
from typing import Callable

import google.genai as genai
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference


_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# --- Provider configuration -------------------------------------------------

GEMINI_MODEL = "gemini-2.5-flash-lite"
WATSONX_MODEL = "ibm/granite-4-h-small"
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"


# --- Prompt templates -------------------------------------------------------

GROUNDED_TEMPLATE = dedent("""\
    You are a highly reliable, enterprise-grade AI assistant operating in a
    retrieval-augmented generation (RAG) setting.

    Your task is to answer the user's question precisely using ONLY the
    information provided in the context below. If the answer cannot be found
    explicitly or inferred directly from the context, state that you don't
    know based on the provided documents.

    - Do not rely on any external knowledge outside of the provided context.
    - Always cite the document titles used in your answer.
    - Use clear inline citations like: (Title: <document title>)
    - If multiple documents are used, cite each relevant title.

    CONTEXT:
    {context}

    USER QUESTION:
    {question}

    ANSWER:
""")

PLAIN_TEMPLATE = dedent("""\
    You are a helpful assistant. Answer the user's question using the
    provided context.

    Context:
    {context}

    Question: {question}
""")

PROMPT_TEMPLATES = {
    "grounded": GROUNDED_TEMPLATE,
    "plain": PLAIN_TEMPLATE,
}


# --- Credentials ------------------------------------------------------------

def _get_api_keys(api_keys_file: str = "apikeys.json") -> dict:
    """Load API keys from a JSON file, looking in the project root then CWD."""
    project_root = Path(__file__).resolve().parent.parent
    candidates = [project_root / api_keys_file, Path(api_keys_file)]

    for path in candidates:
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON from {path}") from e

    raise FileNotFoundError(f"API keys file not found: {api_keys_file}")


def _require_key(name: str) -> str:
    """Fetch a credential from apikeys.json (case-insensitive) or env vars."""
    keys = _get_api_keys()
    value = keys.get(name) or keys.get(name.lower()) or os.getenv(name.upper())
    if not value:
        raise ValueError(f"No {name} credential found!")
    return value


# --- Prompt construction ----------------------------------------------------

def _format_page_line(chunk: dict) -> str:
    """Return a 'Page: X' or 'Page: X–Y' line, or empty string if unavailable."""
    try:
        page_start = int(chunk.get("page_start", -1))
    except (TypeError, ValueError):
        return ""
    if page_start <= 0:
        return ""

    try:
        page_end = int(chunk.get("page_end", page_start))
    except (TypeError, ValueError):
        page_end = page_start

    range_str = f"{page_start}" if page_end == page_start else f"{page_start}–{page_end}"
    return f"Page: {range_str}\n"


def _format_chunk(index: int, chunk: dict) -> str:
    """Render a single retrieved chunk as a labeled context block."""
    return (
        f"--- Document {index} ---\n"
        f"Title: {chunk.get('title', 'Unknown')}\n"
        f"Source: {chunk.get('source', 'Unknown')}\n"
        f"{_format_page_line(chunk)}"
        f"Content: {chunk.get('text', '')}\n"
    )


def build_prompt(
    question: str,
    retrieved_chunks: list[dict],
    prompt_version: str = "grounded",
    distance_threshold: float = 1.5,
) -> str:
    """
    Assemble an LLM prompt from the user question and retrieved chunk dicts.

    Chunks whose ``distance`` is greater than or equal to ``distance_threshold``
    are skipped (treated as not relevant enough for grounded mode).

    Args:
        question: End-user question string.
        retrieved_chunks: List of dicts with ``title``, ``source``, ``text``,
            optional ``distance``, ``page_start``, ``page_end`` (as from
            :func:`retrieve.format_retrieval_results`).
        prompt_version: ``"grounded"`` (strict cite-from-context) or
            ``"plain"`` (lighter instructions).
        distance_threshold: Maximum Chroma distance to include a chunk
            (lower is closer).

    Returns:
        A single string prompt ready for :func:`call_llm`.
    """
    if prompt_version not in PROMPT_TEMPLATES:
        raise ValueError(
            f"Unknown prompt_version '{prompt_version}'. "
            f"Expected one of: {list(PROMPT_TEMPLATES)}"
        )

    relevant_chunks = [
        c for c in retrieved_chunks
        if c.get("distance", 10) < distance_threshold
    ]
    context = "\n".join(
        _format_chunk(i, c) for i, c in enumerate(relevant_chunks, start=1)
    )

    return PROMPT_TEMPLATES[prompt_version].format(
        context=context, question=question
    )


# --- LLM calls --------------------------------------------------------------

def _call_gemini(prompt: str) -> str:
    client = genai.Client(api_key=_require_key("Gemini"))
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


def _call_watsonx(prompt: str) -> str:
    model = ModelInference(
        model_id=WATSONX_MODEL,
        credentials=Credentials(api_key=_require_key("IBM"), url=WATSONX_URL),
        project_id=_require_key("WATSONX_PROJECT_ID"),
        params={"max_tokens": 1024},
    )
    response = model.chat(messages=[{"role": "user", "content": prompt}])
    return response["choices"][0]["message"]["content"]


_PROVIDERS: dict[str, Callable[[str], str]] = {
    "gemini": _call_gemini,
    "ibm": _call_watsonx,
}


def call_llm(prompt: str, provider: str = "gemini") -> str:
    """
    Send a completed prompt string to a configured model provider.

    Args:
        prompt: Full prompt text from :func:`build_prompt`.
        provider: One of ``"gemini"`` or ``"ibm"``.

    Returns:
        Model response text (first completion).

    Raises:
        ValueError: If the provider is unknown or required credentials are missing.
    """
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Expected one of: {list(_PROVIDERS)}"
        )
    return _PROVIDERS[provider](prompt)


# --- Public entry point -----------------------------------------------------

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
    prompt = build_prompt(question, retrieved_chunks, prompt_version=prompt_version)
    return call_llm(prompt, provider)


# --- Demo -------------------------------------------------------------------

if __name__ == "__main__":
    mock_question = "How do I chat with a model in Python?"
    mock_chunks = [
        {
            "title": "Quick code tutorial: Chat with a model",
            "source": "https://dataplatform.cloud.ibm.com/docs/chat",
            "text": (
                "You can use the ibm_watsonx_ai library to interact with "
                "foundation models. You pass a list of messages with 'role' "
                "and 'content'."
            ),
        }
    ]

    for provider in ("gemini", "ibm"):
        print(f"\nSending to {provider}...")
        answer = generate_answer(
            mock_question,
            mock_chunks,
            prompt_version="grounded",
            provider=provider,
        )
        print(f"\n--- FINAL ANSWER ({provider}) ---")
        print(answer)