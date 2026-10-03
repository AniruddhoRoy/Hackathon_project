"""Import the pre-extracted chunks from the ShihabReza/nctb-qa dataset (CC BY 4.0).

Run:  python -m app.ingest.load_hf
"""

from collections import Counter

from datasets import load_dataset

from app import db
from app.rag import store

DATASET = "ShihabReza/nctb-qa"


def main() -> None:
    db.init_db()
    ds = load_dataset(DATASET)

    chunks: dict[int, store.Chunk] = {}
    for split in ds.values():
        for row in split:
            if row["chunk_id"] in chunks:
                continue
            chunks[row["chunk_id"]] = store.Chunk(
                id=f"hf-{row['chunk_id']}",
                text=row["source_text"],
                source=row["source"],
                subject=row["subject"],
                class_num=int(row["class_num"]),
                language=row["language"],
                page_num=int(row["page_num"]),
            )

    print(f"Embedding {len(chunks)} unique chunks (first run downloads the embedding model)...")
    store.add_chunks(list(chunks.values()))

    per_book = Counter(c.source for c in chunks.values())
    for c in {c.source: c for c in chunks.values()}.values():
        db.upsert_book(c.source, c.subject, c.class_num, c.language, per_book[c.source], origin="hf-dataset")

    print(f"Done. {store.count()} chunks in the vector store across {len(per_book)} books.")


if __name__ == "__main__":
    main()
