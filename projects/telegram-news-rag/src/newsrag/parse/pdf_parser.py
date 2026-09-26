"""Convert downloaded newspaper PDFs into structured documents with Docling.

Usage:
    python -m newsrag.parse.pdf_parser                     # parse today's downloads
    python -m newsrag.parse.pdf_parser --date 2026-09-24
    python -m newsrag.parse.pdf_parser --pages 2           # quick test: first 2 pages of each

Digital PDFs (real text layer) are parsed without OCR. Scanned PDFs (page images)
go through Docling's layout model plus banded Apple Vision OCR (see banded_ocr.py).

Output in data/parsed/<date>/:
    <stem>.json       DoclingDocument (headings, paragraphs, reading order, page boxes)
    <stem>.md         Markdown export, for eyeballing
    manifest.jsonl    one record per paper: publication, method, pages, timing
"""

import argparse
import json
import re
import time
from datetime import date
from pathlib import Path

import pymupdf
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from newsrag.config import get_settings
from newsrag.parse.banded_ocr import BandedOcrMacOptions, register

# Text some uploaders stamp on every page of a scan; it doesn't count as real text
WATERMARKS = ("tradingref.com",)
MIN_CHARS_PER_PAGE = 200

MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec"
)
DATE_START = re.compile(rf"(\b\d{{1,2}}[\s_-]+)?\b({MONTHS})\b|\d{{4,}}", re.IGNORECASE)


def publication_name(original_name: str) -> str:
    """'The Wall Street Journal - 24 September 2026.pdf' -> 'The Wall Street Journal'."""
    stem = Path(original_name).stem
    stem = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+", "", stem)  # some uploads write the date in superscript
    match = DATE_START.search(stem)
    if match:
        stem = stem[: match.start()]
    if " " not in stem:
        stem = stem.replace("-", " ")  # 'The-i-Paper' -> 'The i Paper'
    return re.sub(r"\s+", " ", stem.replace("_", " ")).strip(" -,") or Path(original_name).stem


def is_scanned(pdf: Path) -> bool:
    with pymupdf.open(pdf) as doc:
        chars = 0
        for page in doc:
            text = page.get_text()
            for mark in WATERMARKS:
                text = text.replace(mark, "")
            chars += len(text.strip())
        return chars / max(len(doc), 1) < MIN_CHARS_PER_PAGE


def build_converters() -> dict[str, DocumentConverter]:
    register()
    digital = PdfPipelineOptions(do_ocr=False)
    scanned = PdfPipelineOptions(
        do_ocr=True,
        # scale 4 = 288 dpi, roughly the native resolution of these scans
        ocr_options=BandedOcrMacOptions(mode=OcrMode.FULL_PAGE, scale=4.0, lang=["en-US"]),
    )

    def converter(opts: PdfPipelineOptions) -> DocumentConverter:
        # pypdfium2 renders scans much more faithfully than the default docling-parse backend,
        # which garbled OCR on these pages
        fmt = PdfFormatOption(pipeline_options=opts, backend=PyPdfiumDocumentBackend)
        return DocumentConverter(format_options={InputFormat.PDF: fmt})

    return {"digital": converter(digital), "ocr": converter(scanned)}


def load_raw_manifest(raw_dir: Path) -> dict[str, dict]:
    path = raw_dir / "manifest.jsonl"
    if not path.exists():
        return {}
    with path.open() as f:
        return {rec["file"]: rec for rec in map(json.loads, filter(str.strip, f))}


def parse_day(day: date, max_pages: int | None = None, only: list[str] | None = None) -> None:
    settings = get_settings()
    raw_dir = settings.data_dir / "raw" / day.isoformat()
    out_dir = settings.data_dir / "parsed" / day.isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_manifest = load_raw_manifest(raw_dir)
    manifest_path = out_dir / "manifest.jsonl"
    done = set(load_raw_manifest(out_dir)) if max_pages is None else set()

    pdfs = sorted(raw_dir.glob("*.pdf"))
    if only:
        pdfs = [p for p in pdfs if any(s.lower() in p.name.lower() for s in only)]
    if not pdfs:
        print(f"No PDFs in {raw_dir}")
        return

    converters = build_converters()
    for pdf in pdfs:
        if pdf.name in done:
            print(f"skip   {pdf.name} (already parsed)")
            continue

        raw = raw_manifest.get(pdf.name, {})
        publication = publication_name(raw.get("original_name") or pdf.name.split("_", 1)[-1])
        method = "ocr" if is_scanned(pdf) else "digital"
        page_range = (1, max_pages) if max_pages else None

        print(f"parse  {publication:<28} [{method}] ...", end="", flush=True)
        started = time.time()
        kwargs = {"page_range": page_range} if page_range else {}
        doc = converters[method].convert(pdf, **kwargs).document
        seconds = time.time() - started

        doc.save_as_json(out_dir / f"{pdf.stem}.json")
        doc.save_as_markdown(out_dir / f"{pdf.stem}.md")
        pages = len(doc.pages)
        print(f" {pages} pages in {seconds:.0f}s ({seconds / max(pages, 1):.1f}s/page)")

        if max_pages is None:  # partial test runs shouldn't count as done
            record = {
                "file": pdf.name,
                "json": f"{pdf.stem}.json",
                "publication": publication,
                "edition_date": day.isoformat(),
                "method": method,
                "pages": pages,
                "seconds": round(seconds, 1),
                "telegram_message_id": raw.get("message_id"),
            }
            with manifest_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(), help="YYYY-MM-DD (default: today)")
    parser.add_argument("--pages", type=int, help="Only the first N pages of each PDF (for quick tests)")
    parser.add_argument("--only", nargs="+", help="Only files whose name contains one of these strings")
    args = parser.parse_args()
    parse_day(args.date, args.pages, args.only)


if __name__ == "__main__":
    main()
