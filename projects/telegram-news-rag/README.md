# telegram-news-rag

Ask questions like *"What are today's editorials saying about AI?"* over the
newspapers and magazines posted daily to a Telegram channel. A hands-on project
for learning RAG at scale.

## Stack

| Layer | Tool |
|---|---|
| Ingestion | Telethon |
| PDF parsing / OCR | Docling (layout) + Apple Vision via `ocrmac` (OCR), PyMuPDF |
| Framework | LangChain + LangGraph |
| Embeddings | sentence-transformers (`BAAI/bge-m3`) |
| Vector DB | Qdrant (hybrid dense + sparse) |
| Reranking | bge-reranker |
| LLM | Claude (Sonnet for answers, Haiku for tagging) |
| Metadata | PostgreSQL |
| Orchestration | Prefect |
| Serving / UI | FastAPI, Streamlit |
| Eval / tracing | RAGAS, Langfuse |

## Roadmap

- [x] **1. Ingest**: download each day's PDFs from the channel (Telethon)
- [x] **2. Parse**: PDF → structured Docling documents (OCR for scanned papers)
- [ ] **3. Basic RAG**: chunk → embed → Qdrant → Claude answers with citations
- [ ] **4. Better retrieval**: metadata filters, hybrid search, reranker
- [ ] **5. Measure**: RAGAS eval set, Langfuse tracing
- [ ] **6. Agent**: LangGraph agent that picks filters and searches iteratively
- [ ] **7. Productionize**: Prefect schedule, FastAPI, Streamlit, Docker Compose

## Setup

```bash
cd projects/telegram-news-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # then fill in your values
```

Get `TELEGRAM_API_ID` and `TELEGRAM_API_HASH` from <https://my.telegram.org>
under **API development tools**. This logs in as *your* account, so it can read
any channel you've joined (a bot could not).

## Step 1: Ingest

```bash
# See what was posted today (the first run asks for your phone number + login code)
python -m newsrag.ingest.telegram_fetcher --dry-run

# Download today's PDFs
python -m newsrag.ingest.telegram_fetcher

# A specific day
python -m newsrag.ingest.telegram_fetcher --date 2026-09-23
```

Output:

```
data/
├── telegram.session          # your logged-in session (keep private!)
└── raw/2026-09-24/
    ├── 1234_TheHindu_24Sep.pdf
    ├── 1235_TheEconomist.pdf
    └── manifest.jsonl        # message id, caption, timestamp per file
```

Re-running is safe: files already in the manifest are skipped.

## Step 2: Parse

```bash
python -m newsrag.parse.pdf_parser                  # parse today's downloads
python -m newsrag.parse.pdf_parser --pages 2        # quick test on the first 2 pages
python -m newsrag.parse.pdf_parser --only NYT WSJ   # just some papers
```

Output goes to `data/parsed/<date>/`: a `DoclingDocument` JSON and a Markdown file per paper, plus a `manifest.jsonl` recording the publication name, whether OCR was used, and timing.

What we learned about the data:

- **Most papers are scans.** NYT, WSJ, Guardian and FT are 300 dpi page images whose only text is a `tradingref.com` watermark. The Washington Post and The Independent have real text.
- **OCR a full newspaper page in one go and you get garbage.** Apple Vision downscales large images, so the page is split into overlapping full-width bands, and the duplicate lines are removed ([banded_ocr.py](src/newsrag/parse/banded_ocr.py)). This took body-text OCR from unreadable to near-perfect.
- **The PDF backend matters.** Docling's default `docling-parse` backend rendered these scans badly for OCR, while `pypdfium2` is clean.
- Docling's built-in `LAYOUT_REGIONS` OCR mode produced garbage on these pages, so it isn't used.
- Speed on an Apple Silicon Mac: about 2 s per page with OCR, about 0.3 s per page for digital PDFs.
