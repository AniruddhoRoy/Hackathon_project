"""Retrieval evaluation on the nctb-qa test split (200 questions). No LLM calls, so it's free.

Run (after `python -m app.ingest.load_hf`):  python -m eval.run_eval
"""

from collections import defaultdict

from datasets import load_dataset

from app.rag.store import search

K = 5


def main() -> None:
    test = load_dataset("ShihabReza/nctb-qa", split="test")
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # group -> [hits, n]
    reciprocal_ranks = []

    for row in test:
        ids = [c.id for c in search(row["question"], k=K)]
        gold = f"hf-{row['chunk_id']}"
        rank = ids.index(gold) + 1 if gold in ids else None
        reciprocal_ranks.append(1 / rank if rank else 0)
        for group in ("all", f"lang={row['language']}", f"subject={row['subject']}", f"class={row['class_num']}"):
            totals[group][0] += rank is not None
            totals[group][1] += 1

    print(f"MRR@{K}: {sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}\n")
    print(f"{'group':<20} Hit@{K}")
    for group, (hits, n) in sorted(totals.items()):
        print(f"{group:<20} {hits / n:.3f}  ({hits}/{n})")


if __name__ == "__main__":
    main()
