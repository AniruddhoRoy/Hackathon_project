"""Answer generation with Gemma, grounded in numbered excerpts.

Providers (set LLM_PROVIDER in .env):
  - ollama: Ollama Cloud (https://ollama.com), e.g. gemma4:31b
  - google: Gemini API (Google AI Studio), e.g. gemma-4-26b-a4b-it

Gemma has no built-in citation feature, so excerpts are numbered [1]..[n] and the
model is asked to cite them; cited numbers are parsed from the answer text.
"""

import re
from collections.abc import AsyncIterator

from app.config import (
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
    LLM_PROVIDER,
    OLLAMA_API_KEY,
    OLLAMA_HOST,
    OLLAMA_MODEL,
)
from app.rag.store import Chunk

SYSTEM_PROMPT = """You are a study assistant for Bangladeshi school students. You answer questions using only the numbered NCTB textbook excerpts provided with each question.

Rules:
- Use only the information in the provided excerpts. Do not add facts from outside knowledge.
- After each fact, cite the excerpt it came from, like [1] or [2][3].
- Reply in the same language as the student's question (Bangla or English).
- If the excerpts do not contain the answer, say so plainly: in Bangla "দুঃখিত, নির্বাচিত বইয়ে এই প্রশ্নের উত্তর পাওয়া যায়নি।", in English "Sorry, I couldn't find the answer in the selected books." Do not guess.
- When the textbook gives a definition, quote it.
- Keep answers clear and short enough for a student to read quickly. Use plain text; short bullet lists are fine."""

TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 2048
CITATION = re.compile(r"\[(\d+)\]")


class LLMError(Exception):
    """Provider-independent error shown to the user."""


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    excerpts = "\n\n".join(
        f"[{i}] ({c.source}, Class {c.class_num}, {c.subject}, Page {c.page_num})\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    )
    return f"Textbook excerpts:\n\n{excerpts}\n\nStudent's question: {question}"


# ---------- Ollama Cloud ----------

_ollama = None


async def _ollama_stream(prompt: str) -> AsyncIterator[str]:
    global _ollama
    from ollama import AsyncClient, RequestError, ResponseError

    if _ollama is None:
        _ollama = AsyncClient(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"})
    try:
        async for part in await _ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": TEMPERATURE, "num_predict": MAX_OUTPUT_TOKENS},
        ):
            if part.message.content:
                yield part.message.content
    except (ResponseError, RequestError) as e:
        raise LLMError(f"Ollama: {e}") from e


# ---------- Google Gemini API ----------

_google = None


async def _google_stream(prompt: str) -> AsyncIterator[str]:
    global _google
    from google import genai
    from google.genai import errors, types

    if _google is None:
        _google = genai.Client(api_key=GOOGLE_API_KEY)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    started = False
    try:
        async for chunk in await _google.aio.models.generate_content_stream(
            model=GOOGLE_MODEL, contents=prompt, config=config
        ):
            if chunk.text:
                started = True
                yield chunk.text
    except errors.ServerError as e:
        # Some Gemma models intermittently fail to stream; retry once without streaming.
        if started:
            raise LLMError(f"Gemini API: {e.message or e}") from e
        try:
            response = await _google.aio.models.generate_content(model=GOOGLE_MODEL, contents=prompt, config=config)
        except errors.APIError as e2:
            raise LLMError(f"Gemini API: {e2.message or e2}") from e2
        if response.text:
            yield response.text
    except errors.APIError as e:
        raise LLMError(f"Gemini API: {e.message or e}") from e


PROVIDERS = {"ollama": _ollama_stream, "google": _google_stream}


async def stream_answer(question: str, chunks: list[Chunk], cited: list[int]) -> AsyncIterator[str]:
    """Yield answer text deltas. After the stream ends, `cited` holds the 0-based
    indexes (into `chunks`) of the excerpts the model cited."""
    if LLM_PROVIDER not in PROVIDERS:
        raise LLMError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' (use: {', '.join(PROVIDERS)})")

    parts: list[str] = []
    async for text in PROVIDERS[LLM_PROVIDER](build_prompt(question, chunks)):
        parts.append(text)
        yield text

    for n in CITATION.findall("".join(parts)):
        i = int(n) - 1
        if 0 <= i < len(chunks) and i not in cited:
            cited.append(i)
