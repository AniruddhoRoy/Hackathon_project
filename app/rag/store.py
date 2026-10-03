"""ChromaDB vector store (persistent, stored under data/vectordb)."""

from dataclasses import dataclass
from functools import lru_cache

import chromadb

from app.config import COLLECTION_NAME, VECTOR_DIR
from app.rag.embeddings import embed_passages, embed_query


@dataclass
class Chunk:
    id: str
    text: str
    source: str  # PDF file name
    subject: str
    class_num: int
    language: str  # bn | en
    page_num: int
    score: float = 0.0  # cosine similarity, filled in on search

    def metadata(self) -> dict:
        return {
            "source": self.source,
            "subject": self.subject,
            "class_num": self.class_num,
            "language": self.language,
            "page_num": self.page_num,
        }


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    return client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def add_chunks(chunks: list[Chunk], batch_size: int = 256) -> None:
    collection = get_collection()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.upsert(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata() for c in batch],
            embeddings=embed_passages([c.text for c in batch]),
        )


def delete_source(source: str) -> None:
    get_collection().delete(where={"source": source})


def count() -> int:
    return get_collection().count()


def _where(filters: dict) -> dict | None:
    conditions = [{k: v} for k, v in filters.items() if v not in (None, "")]
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def search(query: str, k: int, filters: dict | None = None) -> list[Chunk]:
    result = get_collection().query(
        query_embeddings=[embed_query(query)],
        n_results=k,
        where=_where(filters or {}),
    )
    chunks = []
    for id_, text, meta, dist in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        chunks.append(Chunk(id=id_, text=text, score=1 - dist, **meta))
    return chunks
