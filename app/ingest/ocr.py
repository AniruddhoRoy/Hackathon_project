"""OCR backend for scanned pages and Bijoy-encoded Bangla PDFs.

Not chosen yet; pick one (PROJECT_GUIDELINE.md §5.2) and implement `ocr_page`:
  - Google Cloud Vision / Document AI: best Bangla accuracy, paid
  - Claude vision: send the page image, ask for a Unicode transcription
  - Surya OCR / EasyOCR / Tesseract (ben): free, run on a GPU (Colab) for whole books

Render the page to an image with:
    pix = page.get_pixmap(dpi=200)
    png_bytes = pix.tobytes("png")
"""

import pymupdf


class OCRNotConfigured(RuntimeError):
    pass


def ocr_page(page: pymupdf.Page, language: str) -> str:
    raise OCRNotConfigured(
        f"Page {page.number + 1} has no usable text layer ({language}); it needs OCR. "
        "Implement app/ingest/ocr.py:ocr_page with an OCR backend."
    )
