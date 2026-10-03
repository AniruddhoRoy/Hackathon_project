# NCTB Book Q&A

Ask questions about NCTB textbooks in Bangla or English. Answers come only from the books, with book and page citations.
One FastAPI app (pages + API + admin), ChromaDB for retrieval, Claude for answers. See [PROJECT_GUIDELINE.md](PROJECT_GUIDELINE.md) for the full plan.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env        # then set ANTHROPIC_API_KEY and ADMIN_PASSWORD
```

## Load the starter books (16 ICT/Science books, Classes 6–9)

```bash
python -m app.ingest.load_hf
```

The first run downloads the embedding model (~1 GB for e5-base) and embeds about 6.9k chunks. That takes a few minutes on CPU.

## Run

```bash
uvicorn app.main:app --reload
```

- Chat: http://127.0.0.1:8000
- Admin (upload/delete books): http://127.0.0.1:8000/admin. Log in as `admin` with your `ADMIN_PASSWORD`.
- Health: http://127.0.0.1:8000/health

## Evaluate retrieval

```bash
python -m eval.run_eval
```

Prints Hit@5 and MRR on the dataset's 200 test questions, split by language, subject, and class.

## Add more books

- One at a time: upload in `/admin`.
- In bulk: put PDFs and a `manifest.csv` in `data/pdfs/`, then run `python -m scripts.ingest_folder data/pdfs/manifest.csv`.

Scanned PDFs and Bijoy-font Bangla PDFs need OCR. Implement a backend in [app/ingest/ocr.py](app/ingest/ocr.py).

## Project layout

```
app/
  main.py          routes: chat page, /api/ask (streaming), /admin
  config.py        settings from .env
  db.py            SQLite: books + ingestion jobs
  rag/             embeddings, Chroma store, retriever, Claude generator
  ingest/          PDF extract, OCR (stub), chunking, pipeline, HF dataset loader
  templates/       Jinja2 pages
  static/          CSS + small chat script
eval/run_eval.py   retrieval metrics on the test split
scripts/           bulk ingestion
data/              PDFs, vector DB, SQLite (git-ignored)
```

## Credits

Starter corpus: [ShihabReza/nctb-qa](https://huggingface.co/datasets/ShihabReza/nctb-qa) (NCTBench), CC BY 4.0. Textbooks © NCTB, Bangladesh.
