"""Extract per-page text from a PDF, falling back to OCR when the text layer is unusable."""

import re
import unicodedata
from pathlib import Path

import pymupdf

from app.ingest.ocr import ocr_page

BENGALI_CHAR = re.compile(r"[ঀ-৿]")
LETTER = re.compile(r"\w")


def bengali_ratio(text: str) -> float:
    letters = len(LETTER.findall(text))
    return len(BENGALI_CHAR.findall(text)) / letters if letters else 0.0


def needs_ocr(text: str, language: str) -> bool:
    """True when a page has no text layer, or (for Bangla books) when the text is
    Bijoy/SutonnyMJ ANSI encoding, which reads as Latin gibberish like 'Avgvi'."""
    if len(text.strip()) < 30:
        return True
    return language == "bn" and bengali_ratio(text) < 0.5


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path: Path, language: str) -> list[tuple[int, str]]:
    """Return [(page_num, text), ...] with 1-indexed page numbers."""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text()
            if needs_ocr(text, language):
                text = ocr_page(page, language)
            text = clean(text)
            if text:
                pages.append((i, text))
    return pages
