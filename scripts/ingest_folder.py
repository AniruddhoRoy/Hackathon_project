"""Bulk-index PDFs listed in a manifest CSV.

manifest.csv columns: file,subject,class_num,language
    ICT_ben_class8.pdf,ICT,8,bn

Run:  python -m scripts.ingest_folder data/pdfs/manifest.csv
"""

import csv
import sys
from pathlib import Path

from app import db
from app.ingest.pipeline import ingest_pdf


def main(manifest: Path) -> None:
    db.init_db()
    with manifest.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pdf = manifest.parent / row["file"]
            try:
                n = ingest_pdf(pdf, row["subject"], int(row["class_num"]), row["language"])
                print(f"OK    {pdf.name}: {n} chunks")
            except Exception as e:
                print(f"FAIL  {pdf.name}: {e}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
