"""NCTB Book Q&A: one FastAPI app serving pages, the ask API, and admin ingestion.

Run:  uvicorn app.main:app --reload
"""

import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app import db
from app.config import ADMIN_PASSWORD, PDF_DIR
from app.ingest.extract import BENGALI_CHAR
from app.ingest.pipeline import run_job
from app.rag import store
from app.rag.generator import LLMError, stream_answer
from app.rag.retriever import retrieve

APP_DIR = Path(__file__).parent

NOT_FOUND = {
    "bn": "দুঃখিত, নির্বাচিত বইয়ে এই প্রশ্নের উত্তর পাওয়া যায়নি।",
    "en": "Sorry, I couldn't find the answer in the selected books.",
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="NCTB Book Q&A", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")
security = HTTPBasic()


def require_admin(credentials: HTTPBasicCredentials = Depends(security)) -> None:
    ok = secrets.compare_digest(credentials.username, "admin") and secrets.compare_digest(
        credentials.password, ADMIN_PASSWORD
    )
    if not ok:
        raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Basic"})


# ---------- Student-facing ----------

@app.get("/")
def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html", {"options": db.filter_options()})


class AskRequest(BaseModel):
    question: str
    class_num: int | None = None
    subject: str | None = None
    language: str | None = None


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/ask")
async def ask(body: AskRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty")
    lang = "bn" if BENGALI_CHAR.search(question) else "en"

    async def events():
        chunks = await run_in_threadpool(retrieve, question, body.class_num, body.subject, body.language)
        if not chunks:
            yield sse("token", NOT_FOUND[lang])
            yield sse("done", {})
            return

        cited: list[int] = []
        try:
            async for text in stream_answer(question, chunks, cited):
                yield sse("token", text)
        except LLMError as e:
            yield sse("error", f"LLM error: {e}")
            return

        yield sse("sources", [
            {"n": i + 1, "source": chunks[i].source, "class_num": chunks[i].class_num,
             "subject": chunks[i].subject, "page_num": chunks[i].page_num, "text": chunks[i].text}
            for i in sorted(cited)
        ])
        yield sse("done", {})

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok", "chunks": store.count()}


# ---------- Admin ----------

@app.get("/admin", dependencies=[Depends(require_admin)])
def admin_page(request: Request):
    return templates.TemplateResponse(
        request, "admin/index.html", {"books": db.list_books(), "jobs": db.recent_jobs()}
    )


@app.post("/admin/upload", dependencies=[Depends(require_admin)])
async def admin_upload(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    subject: str = Form(...),
    class_num: int = Form(...),
    language: str = Form(...),
):
    name = Path(file.filename or "").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")
    if language not in ("bn", "en"):
        raise HTTPException(400, "language must be bn or en")

    pdf_path = PDF_DIR / name
    pdf_path.write_bytes(await file.read())
    job_id = db.create_job(name)
    background.add_task(run_job, job_id, pdf_path, subject.strip(), class_num, language)
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/books/{source}/delete", dependencies=[Depends(require_admin)])
def admin_delete_book(source: str):
    store.delete_source(source)
    db.delete_book(source)
    return RedirectResponse("/admin", status_code=303)
