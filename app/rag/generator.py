"""Answer generation with Claude, grounded in retrieved chunks with citations."""

from collections.abc import AsyncIterator

import anthropic

from app.config import LLM_EFFORT, LLM_MODEL
from app.rag.store import Chunk

# Keep this text fixed (no dates/IDs). Caching only kicks in once the prompt
# passes the model's minimum cacheable length (e.g. after adding few-shot examples).
SYSTEM_PROMPT = """You are a study assistant for Bangladeshi school students. You answer questions using only the NCTB textbook excerpts provided with each question.

Rules:
- Use only the information in the provided excerpts. Do not add facts from outside knowledge.
- Reply in the same language as the student's question (Bangla or English).
- If the excerpts do not contain the answer, say so plainly: in Bangla "দুঃখিত, নির্বাচিত বইয়ে এই প্রশ্নের উত্তর পাওয়া যায়নি।", in English "Sorry, I couldn't find the answer in the selected books." Do not guess.
- When the textbook gives a definition, quote it.
- Keep answers clear and short enough for a student to read quickly. Use plain text; short bullet lists are fine.

Latency-sensitive; begin your visible answer immediately."""

FALLBACK_BETA = "server-side-fallback-2026-07-01"

_client: anthropic.AsyncAnthropic | None = None


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic()
    return _client


def _document_block(chunk: Chunk) -> dict:
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": chunk.text},
        "title": f"{chunk.source} · Class {chunk.class_num} · {chunk.subject} · Page {chunk.page_num}",
        "citations": {"enabled": True},
    }


async def stream_answer(question: str, chunks: list[Chunk], cited: list[int]) -> AsyncIterator[str]:
    """Yield answer text deltas. After the stream ends, `cited` holds the indexes
    (into `chunks`) of the documents Claude cited."""
    content = [_document_block(c) for c in chunks] + [{"type": "text", "text": question}]

    async with get_client().beta.messages.stream(
        model=LLM_MODEL,
        max_tokens=4000,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
        output_config={"effort": LLM_EFFORT},
        # On a safety decline, the API re-runs the request on a recommended fallback model.
        betas=[FALLBACK_BETA],
        fallbacks="default",
    ) as stream:
        async for text in stream.text_stream:
            yield text
        message = await stream.get_final_message()

    if message.stop_reason == "refusal":
        yield "\n\n(Sorry, this question can't be answered.)"
        return

    for block in message.content:
        if block.type == "text":
            for citation in block.citations or []:
                if citation.document_index not in cited:
                    cited.append(citation.document_index)
