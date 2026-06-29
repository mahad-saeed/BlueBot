"""
Text chunker for BlueBot RAG pipeline.

Reads raw policy documents from data/raw/, cleans scraped web artifacts,
and splits them into topic-scoped sections for embedding/retrieval.
"""

from __future__ import annotations

import re
from pathlib import Path

# Default location of raw text files (project_root/data/raw/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Fixed-size fallback for sections that are still too long after structural split
CHUNK_SIZE_WORDS = 400
CHUNK_OVERLAP_WORDS = 50
MIN_WORDS_TO_CHUNK = 100  # Whole-file fallback: don't word-split if under this size

# Lines copied from the Airblue site navigation bar (appear at top of scraped pages)
NAV_HEADER_LINES = frozenset(
    {
        "reservations",
        "travel deals",
        "destinations",
        "bluemiles",
        "login",
        "signup",
        "help",
        "contact centre",
        "welcome",
    }
)

# Footer / site chrome lines (appear at bottom of scraped pages)
FOOTER_LINES = frozenset(
    {
        "airblue",
        "our journey",
        "corporate information",
        "blue news",
        "careers",
        "services",
        "travel info",
        "flight status",
        "travel agents",
        "customer service",
        "contact us",
        "privacy policy",
        "legal terms & conditions",
        "health and travel guidelines",
        "passenger rights",
        "subscribe/unsubscribe to emails",
        "stay connected",
        "subscribe to our special offers",
        "subscribe",
        "download our app to manage flights",
        "google play",
        "app store",
        "airblue on facebook",
        "airblue on linkedin",
        "airblue on twitter",
        "airblue on instagram",
        "airblue on tiktok",
    }
)

# Login-form UI text that sometimes appears in scraped loyalty-program pages
LOGIN_FORM_LINES = frozenset(
    {
        "bluemiles login",
        "member id",
        "password",
        "remember me",
        "signin",
        "booking reference",
        "passenger name",
        "find booking",
    }
)

# Breadcrumb navigation, e.g. "Home > Services > Baggage"
BREADCRUMB_PATTERN = re.compile(r"^[\w\s&/-]+(?:\s*>\s*[\w\s&/-]+)+$", re.IGNORECASE)

# Copyright / legal footer
COPYRIGHT_PATTERN = re.compile(r"©\s*airblue.*", re.IGNORECASE)

# Headquarters address line often repeated in footers
HQ_PATTERN = re.compile(r"airblue hdq:", re.IGNORECASE)

# Fallback: split flattened text before short Title-Case headings followed by body text
_FLAT_HEADING_SPLIT = re.compile(
    r"\s+(?=(?:[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,3})\s+[a-z])"
)


def _normalize_line(line: str) -> str:
    """Collapse internal whitespace and strip a single line."""
    return re.sub(r"\s+", " ", line).strip()


def _is_artifact_line(line: str) -> bool:
    """Return True if a line looks like site navigation or UI chrome."""
    normalized = _normalize_line(line).lower()

    if not normalized:
        return True

    if normalized == "+":
        return True

    if normalized in NAV_HEADER_LINES:
        return True

    if normalized in FOOTER_LINES:
        return True

    if normalized in LOGIN_FORM_LINES:
        return True

    if BREADCRUMB_PATTERN.match(normalized):
        return True

    if COPYRIGHT_PATTERN.search(normalized):
        return True

    if HQ_PATTERN.search(normalized):
        return True

    if "ise towers" in normalized and "jinnah" in normalized:
        return True

    return False


def clean_lines(raw_text: str) -> list[str]:
    """
    Remove scraped artifacts and return cleaned lines (structure preserved).

    Unlike the old clean_text(), this keeps one logical line per list item so
    section headings like "Value" / "Ticket Refund" remain separable.
    """
    if not raw_text or not raw_text.strip():
        return []

    cleaned_lines: list[str] = []
    previous_line: str | None = None

    for raw_line in raw_text.splitlines():
        line = _normalize_line(raw_line)
        if not line:
            continue

        if _is_artifact_line(line):
            continue

        if previous_line is not None and line.lower() == previous_line.lower():
            continue

        cleaned_lines.append(line)
        previous_line = line

    return cleaned_lines


def _is_section_heading(
    line: str,
    next_line: str | None = None,
    prev_line: str | None = None,
) -> bool:
    """
    Detect short Title-Case lines that start a new document section.

    Works on line-separated source files (Value, Flexi, Ticket Change, etc.).
    """
    normalized = _normalize_line(line)
    if not normalized:
        return False

    words = normalized.split()
    if len(words) > 6 or len(normalized) > 70:
        return False

    if re.match(r"^(\d+\.|-\s|•)", normalized):
        return False

    if normalized.endswith((".", "!", "?")) and len(words) >= 4:
        return False

    if normalized[0].islower():
        return False

    # Field labels (e.g. "Refunds & Exchanges:") and their values are not headings
    if ":" in normalized:
        return False

    if prev_line and _normalize_line(prev_line).endswith(":"):
        return False

    # Require Title Case words (reject value phrases like "Allowed with Higher Fee")
    for word in words:
        if word in {"&", "/", "-", "and", "or", "the", "a", "an"}:
            continue
        if not word[0].isupper():
            return False

    if next_line and len(words) <= 3:
        next_normalized = _normalize_line(next_line)
        next_words = next_normalized.split()
        if (
            next_normalized
            and len(next_words) <= 4
            and not next_normalized.endswith(".")
            and not next_normalized[0].islower()
        ):
            return False

    return True


def split_into_sections(lines: list[str]) -> list[str]:
    """Group cleaned lines into one section per detected heading boundary."""
    if not lines:
        return []

    grouped: list[list[str]] = []
    current: list[str] = []

    for index, line in enumerate(lines):
        next_line = lines[index + 1] if index + 1 < len(lines) else None
        prev_line = lines[index - 1] if index > 0 else None
        if current and _is_section_heading(line, next_line, prev_line):
            grouped.append(current)
            current = [line]
        else:
            current.append(line)

    if current:
        grouped.append(current)

    return [" ".join(section).strip() for section in grouped if section]


def split_on_flattened_headings(text: str) -> list[str]:
    """
    Fallback for text that was already flattened to one run-on string.

    Splits before short Title-Case phrases immediately followed by body text.
    """
    if not text.strip():
        return []

    parts = _FLAT_HEADING_SPLIT.split(text.strip())
    sections = [part.strip() for part in parts if part.strip()]
    return sections if sections else [text.strip()]


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """
    Split a long section into overlapping word-based chunks.

    Used only when a single structural section still exceeds CHUNK_SIZE_WORDS.
    """
    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks: list[str] = []
    stride = chunk_size - overlap
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        start += stride

    return chunks


def _sections_from_lines(lines: list[str]) -> list[str]:
    """Build topic sections from lines, with flattened-text fallback if needed."""
    sections = split_into_sections(lines)
    if not sections:
        return []

    # If line-based splitting produced one huge block, try heading detection on flat text
    if len(sections) == 1 and len(sections[0].split()) > CHUNK_SIZE_WORDS:
        flat_sections = split_on_flattened_headings(sections[0])
        if len(flat_sections) > 1:
            sections = flat_sections

    final_sections: list[str] = []
    for section in sections:
        if len(section.split()) > CHUNK_SIZE_WORDS:
            final_sections.extend(split_into_chunks(section))
        else:
            final_sections.append(section)

        # Filter out heading-only stubs (section title with no body content)
        MIN_CHUNK_WORDS = 10
        final_sections = [s for s in final_sections if len(s.split()) >= MIN_CHUNK_WORDS]
    return final_sections


def chunk_file(filepath: Path) -> list[dict[str, str]]:
    """
    Read, clean, and chunk a single .txt file.

    Returns a list of chunk dicts with keys: text, source, chunk_id.
    """
    source_name = filepath.name
    stem = filepath.stem

    try:
        raw_text = filepath.read_text(encoding="utf-8")
    except OSError:
        return []

    lines = clean_lines(raw_text)
    if not lines:
        return []

    text_chunks = _sections_from_lines(lines)

    # Very short whole documents that produced no sections still become one chunk
    if not text_chunks:
        joined = " ".join(lines)
        if len(joined.split()) < MIN_WORDS_TO_CHUNK:
            text_chunks = [joined]
        else:
            text_chunks = split_into_chunks(joined)

    return [
        {
            "text": chunk_text,
            "source": source_name,
            "chunk_id": f"{stem}_{index}",
        }
        for index, chunk_text in enumerate(text_chunks)
    ]


def create_chunks(raw_dir: Path | str = RAW_DATA_DIR) -> list[dict[str, str]]:
    """
    Process every .txt file in raw_dir and return all chunks.

    Files are processed in sorted filename order for deterministic output.
    """
    raw_path = Path(raw_dir)
    if not raw_path.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")

    all_chunks: list[dict[str, str]] = []

    for txt_file in sorted(raw_path.glob("*.txt")):
        all_chunks.extend(chunk_file(txt_file))

    return all_chunks


if __name__ == "__main__":
    chunks = create_chunks()

    print(f"Total chunks created: {len(chunks)}")

    if chunks:
        print("\nSample (first chunk):")
        sample = chunks[0]
        print(f"  chunk_id: {sample['chunk_id']}")
        print(f"  source:   {sample['source']}")
        preview = sample["text"]
        if len(preview) > 500:
            preview = preview[:500] + "..."
        print(f"  text:     {preview}")
    else:
        print("\nNo chunks were created. Check that data/raw/ contains .txt files.")
