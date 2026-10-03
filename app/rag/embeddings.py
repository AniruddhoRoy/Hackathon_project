"""multilingual-e5 embeddings. e5 models expect "query: " / "passage: " prefixes."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_passages(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    vectors = get_model().encode(
        [f"passage: {t}" for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    return get_model().encode(f"query: {text}", normalize_embeddings=True).tolist()
