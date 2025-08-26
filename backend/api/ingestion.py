"""
Ingestion module for multi-PDF RAG pipeline (Phase 1).

Responsibilities:
- Accept single/multiple PDF paths or folder(s).
- Extract text and (optionally) tables page-by-page with metadata.
- Persist processed pages as JSON into a structured output directory.
- Provide functions reusable from FastAPI (bytes-based ingestion) and CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Dict, Any, Union, Optional
from uuid import uuid4

import pdfplumber
import io

# ------------------------------
# Logging setup (industry-style)
# ------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ------------------------------
# Data structures
# ------------------------------
@dataclass
class PageRecord:
    doc_id: str
    filename: str
    page_number: int  # 1-indexed
    text: str
    tables: List[List[List[Optional[str]]]]


# ------------------------------
# Helpers
# ------------------------------
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")

def _safe_stem(stem: str) -> str:
    cleaned = SAFE_FILENAME_RE.sub("-", stem).strip("-._")
    return cleaned or f"doc-{uuid4().hex[:8]}"


def _ensure_outdir(outdir: Union[str, Path]) -> Path:
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _normalize_inputs(inputs: Iterable[Union[str, Path]]) -> List[Path]:
    pdfs: List[Path] = []
    for raw in inputs:
        p = Path(raw).expanduser().resolve()
        if p.is_file() and p.suffix.lower() == ".pdf":
            pdfs.append(p)
        elif p.is_dir():
            pdfs.extend(sorted(p.rglob("*.pdf")))
        else:
            logger.warning("Skipping non-PDF path: %s", p)
    seen = set()
    unique_pdfs: List[Path] = []
    for p in pdfs:
        if p not in seen:
            seen.add(p)
            unique_pdfs.append(p)
    if not unique_pdfs:
        logger.error("No PDF files found from the given inputs.")
    return unique_pdfs


# ------------------------------
# Core extraction
# ------------------------------
def _extract_page(page: pdfplumber.page.Page, *, extract_tables: bool) -> Dict[str, Any]:
    try:
        text = page.extract_text() or ""
    except Exception as e:
        logger.exception("Text extraction failed: %s", e)
        text = ""

    tables: List[List[List[Optional[str]]]] = []
    if extract_tables:
        try:
            raw_tables = page.extract_tables()
            for tbl in raw_tables or []:
                normalized = [[(cell if cell is not None else "") for cell in row] for row in tbl]
                tables.append(normalized)
        except Exception as e:
            logger.exception("Table extraction failed: %s", e)
    return {"text": text, "tables": tables}


def extract_pdf(
    pdf_path: Union[str, Path],
    *,
    extract_tables: bool = True,
    max_pages: int = 0,
    doc_id: Optional[str] = None,
) -> List[PageRecord]:
    p = Path(pdf_path).resolve()
    if not (p.exists() and p.is_file() and p.suffix.lower() == ".pdf"):
        raise FileNotFoundError(f"Not a valid PDF file: {p}")

    the_doc_id = doc_id or uuid4().hex
    records: List[PageRecord] = []

    with pdfplumber.open(p) as pdf:
        total = len(pdf.pages)
        limit = min(max_pages, total) if max_pages and max_pages > 0 else total
        logger.info("Extracting '%s' (%d pages; limit=%d)", p.name, total, limit)

        for idx in range(limit):
            page = pdf.pages[idx]
            payload = _extract_page(page, extract_tables=extract_tables)
            rec = PageRecord(
                doc_id=the_doc_id,
                filename=p.name,
                page_number=idx + 1,
                text=payload["text"],
                tables=payload["tables"],
            )
            records.append(rec)
    return records


def extract_pdf_from_bytes(
    file_bytes: bytes,
    filename: str,
    *,
    extract_tables: bool = True,
    max_pages: int = 0,
    doc_id: Optional[str] = None,
) -> List[PageRecord]:
    the_doc_id = doc_id or uuid4().hex
    records: List[PageRecord] = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        total = len(pdf.pages)
        limit = min(max_pages, total) if max_pages and max_pages > 0 else total
        logger.info("Extracting '%s' from bytes (%d pages; limit=%d)", filename, total, limit)

        for idx in range(limit):
            page = pdf.pages[idx]
            payload = _extract_page(page, extract_tables=extract_tables)
            rec = PageRecord(
                doc_id=the_doc_id,
                filename=filename,
                page_number=idx + 1,
                text=payload["text"],
                tables=payload["tables"],
            )
            records.append(rec)
    return records


# ------------------------------
# Persistence
# ------------------------------
def save_processed(pages: List[PageRecord], outdir: Union[str, Path]) -> Path:
    if not pages:
        raise ValueError("'pages' is empty; nothing to save.")

    out = _ensure_outdir(outdir)
    stem_source = Path(pages[0].filename).stem
    safe_stem = _safe_stem(stem_source)
    out_path = out / f"{safe_stem}.json"

    payload = [asdict(p) for p in pages]

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Saved processed JSON: %s (%d pages)", out_path, len(pages))
    return out_path


# ------------------------------
# New helper for FastAPI
# ------------------------------
def pdf_to_chunks(file_bytes: bytes, filename: str, output_dir: str = "data/processed") -> List[Dict[str, Any]]:
    """Convenience wrapper for FastAPI: takes file bytes, extracts pages, saves JSON, returns chunks."""
    pages = extract_pdf_from_bytes(file_bytes, filename)

    chunks = []
    for i, page in enumerate(pages):
        chunks.append({
            "filename": filename,
            "chunk_id": i,
            "content": page.content if hasattr(page, "content") else str(page)
        })

    # Save processed chunks to JSON
    os.makedirs(output_dir, exist_ok=True)
    safe_filename = filename.replace(" ", "-").replace("(", "").replace(")", "")
    out_path = os.path.join(output_dir, f"{os.path.splitext(safe_filename)[0]}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return chunks


# ------------------------------
# Orchestration
# ------------------------------
def process_pdfs(
    inputs: Iterable[Union[str, Path]],
    *,
    outdir: Union[str, Path] = "data/processed",
    extract_tables: bool = True,
    max_pages: int = 0,
) -> List[Path]:
    pdf_files = _normalize_inputs(inputs)
    outputs: List[Path] = []
    for pdf_path in pdf_files:
        try:
            pages = extract_pdf(pdf_path, extract_tables=extract_tables, max_pages=max_pages)
            json_path = save_processed(pages, outdir)
            outputs.append(json_path)
        except Exception:
            logger.exception("Failed to process: %s", pdf_path)
    return outputs


# ------------------------------
# CLI
# ------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDF(s) into processed JSON.")
    parser.add_argument("--input", "-i", nargs="+", required=True)
    parser.add_argument("--outdir", "-o", default="data/processed")
    parser.add_argument("--no-tables", action="store_true")
    parser.add_argument("--max-pages", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    outputs = process_pdfs(
        args.input,
        outdir=args.outdir,
        extract_tables=not args.no_tables,
        max_pages=args.max_pages,
    )
    if outputs:
        logger.info("Completed. JSON files written:\n%s", "\n".join(str(p) for p in outputs))
        return 0
    logger.error("No outputs produced.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
