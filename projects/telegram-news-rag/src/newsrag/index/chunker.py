"""Turn parsed newspapers (DoclingDocument JSON) into LangChain Documents ready to embed.

Docling's HybridChunker splits along the document structure (headings, paragraphs)
and then fits pieces to the embedding model's token limit. Each chunk keeps the
headings above it, so "Xi Pushes All-Out Drive..." travels with its paragraphs.
"""

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DocItemLabel, DoclingDocument
from langchain_core.documents import Document

from newsrag.config import get_settings

# Chunks shorter than this are page furniture: bylines, section labels, "Continued on Page A17"
MIN_WORDS = 40
# Market data tables, TV listings, etc. are not what we ask questions about
SKIP_LABELS = {DocItemLabel.TABLE, DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}


def build_chunker() -> HybridChunker:
    settings = get_settings()
    tokenizer = HuggingFaceTokenizer.from_pretrained(settings.embed_model, max_tokens=settings.chunk_max_tokens)
    return HybridChunker(tokenizer=tokenizer, merge_peers=True)


def chunk_id(file: str, index: int) -> str:
    """Stable id, so re-indexing the same paper overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"newsrag:{file}:{index}"))


def chunk_paper(json_path: Path, paper: dict, chunker: HybridChunker) -> Iterator[Document]:
    doc = DoclingDocument.load_from_json(json_path)
    for index, chunk in enumerate(chunker.chunk(doc)):
        items = chunk.meta.doc_items
        if all(item.label in SKIP_LABELS for item in items):
            continue
        if len(chunk.text.split()) < MIN_WORDS:
            continue

        pages = sorted({prov.page_no for item in items for prov in item.prov})
        headings = chunk.meta.headings or []
        yield Document(
            id=chunk_id(paper["file"], index),
            # contextualize() prepends the headings, so the embedding "knows" which article this is
            page_content=chunker.contextualize(chunk),
            metadata={
                "source": "news",
                "publication": paper["publication"],
                "edition_date": paper["edition_date"],
                "page": pages[0] if pages else None,
                "pages": pages,
                "headline": headings[-1] if headings else None,
                "headings": headings,
                "file": paper["file"],
                "ocr": paper["method"] == "ocr",
                "chunk_index": index,
            },
        )


def iter_day_chunks(day: str) -> Iterator[Document]:
    parsed_dir = get_settings().parsed_dir / day
    manifest = parsed_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"No parsed papers for {day}. Run: python -m newsrag.parse.pdf_parser --date {day}")

    chunker = build_chunker()
    with manifest.open() as f:
        for paper in map(json.loads, filter(str.strip, f)):
            yield from chunk_paper(parsed_dir / paper["json"], paper, chunker)
