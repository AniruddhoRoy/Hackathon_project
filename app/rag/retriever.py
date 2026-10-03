"""Retrieval entry point.

Currently dense-only. Planned upgrades (see PROJECT_GUIDELINE.md §6):
  - hybrid search: add BM25 and merge with Reciprocal Rank Fusion
  - rerank top-20 with BAAI/bge-reranker-v2-m3
"""

from app.config import MIN_SIMILARITY, TOP_K
from app.rag.store import Chunk, search


def retrieve(question: str, class_num: int | None = None, subject: str | None = None,
             language: str | None = None, k: int = TOP_K) -> list[Chunk]:
    filters = {"class_num": class_num, "subject": subject, "language": language}
    chunks = search(question, k=k, filters=filters)
    return [c for c in chunks if c.score >= MIN_SIMILARITY]
