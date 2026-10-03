"""Sentence-aware sliding-window chunking (~1000 chars, ~200 overlap)."""

import re

# Split after Bangla dari (।), ?, ! or a period followed by whitespace.
SENTENCE_END = re.compile(r"(?<=[।?!.])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in split_sentences(text):
        if current and length + len(sentence) > size:
            chunks.append(" ".join(current))
            # Carry the trailing sentences into the next chunk as overlap.
            carried: list[str] = []
            carried_len = 0
            for s in reversed(current):
                if carried_len + len(s) > overlap:
                    break
                carried.insert(0, s)
                carried_len += len(s) + 1
            current, length = carried, carried_len
        current.append(sentence)
        length += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks
