"""PDF → pages → chunks → vector store. Used by the admin upload and scripts/ingest_folder.py."""

from pathlib import Path

from app import db
from app.ingest.chunk import chunk_text
from app.ingest.extract import extract_pages
from app.rag import store


def ingest_pdf(pdf_path: Path, subject: str, class_num: int, language: str) -> int:
    """Index one PDF, replacing any previous version of it. Returns the chunk count."""
    source = pdf_path.name
    chunks: list[store.Chunk] = []
    for page_num, text in extract_pages(pdf_path, language):
        for i, piece in enumerate(chunk_text(text)):
            chunks.append(store.Chunk(
                id=f"{source}-p{page_num}-{i}",
                text=piece,
                source=source,
                subject=subject,
                class_num=class_num,
                language=language,
                page_num=page_num,
            ))
    if not chunks:
        raise ValueError(f"No text extracted from {source}")

    store.delete_source(source)
    store.add_chunks(chunks)
    db.upsert_book(source, subject, class_num, language, len(chunks), origin="upload")
    return len(chunks)


def run_job(job_id: int, pdf_path: Path, subject: str, class_num: int, language: str) -> None:
    db.update_job(job_id, "running")
    try:
        n = ingest_pdf(pdf_path, subject, class_num, language)
        db.update_job(job_id, "done", f"{n} chunks indexed")
    except Exception as e:  # surface any failure in the admin UI
        db.update_job(job_id, "failed", str(e))
