"""
Convert scraped HTML-in-text files into structured JSON documents.

Each input ``.txt`` in ``data/raw/text/`` is expected to have a URL on the first line
and HTML body on the following lines (legacy Watsonx doc scrape format). Output files
are written to ``data/raw/json/<stem>.json`` for downstream :mod:`chunk_corpus`.
"""

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bs4 import BeautifulSoup

import config


def process_txt_file(file_path: Path) -> None:
    """
    Parse one ``.txt`` file: strip boilerplate HTML tags, derive title and plain text,
    and write a sibling JSON document under ``data/raw/json/``.

    Args:
        file_path: Path to a ``*.txt`` file under ``data/raw/text/``.

    Returns:
        None. Writes ``config.RAW_JSON_DIR / f"{stem}.json"`` or no-op if the file is empty.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return

    url = lines[0].strip()
    html_content = "".join(lines[1:])

    soup = BeautifulSoup(html_content, "html.parser")
    for junk in soup(["style", "script", "svg"]):
        junk.extract()

    h1_tag = soup.find("h1")
    if h1_tag:
        title = h1_tag.get_text(strip=True)
    else:
        title = f"Document — {file_path.stem}"

    text_content = soup.get_text(separator=" ", strip=True)

    doc_data = {
        "title": title,
        "source": url,
        "category": "Gen AI Solutions",
        "text": text_content,
    }

    json_path = config.RAW_JSON_DIR / f"{file_path.stem}.json"
    config.RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc_data, f, indent=4)

    print(f"Saved formatted JSON to {json_path}")


def run() -> None:
    """
    Process every ``*.txt`` in :data:`config.RAW_TEXT_DIR`.

    Returns:
        None. Prints discovered paths and per-file status.
    """
    config.RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    txt_files = sorted(config.RAW_TEXT_DIR.glob("*.txt"))
    print(txt_files)

    for file_path in txt_files:
        print(f"Processing {file_path}...")
        process_txt_file(file_path)

    print("\nAll data structured! Layer 1 is officially complete.")


if __name__ == "__main__":
    run()
