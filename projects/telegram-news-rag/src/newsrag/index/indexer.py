"""Chunk a day's parsed papers, embed them, and store them in Qdrant.

Usage:
    python -m newsrag.index.indexer                    # index today
    python -m newsrag.index.indexer --date 2026-09-24

Re-running is safe: chunk ids are deterministic, so existing chunks are overwritten.
"""

import argparse
import time
from datetime import date

from rich.progress import track

from newsrag.index.chunker import iter_day_chunks
from newsrag.index.store import get_vector_store

BATCH = 64


def index_day(day: date) -> None:
    started = time.time()
    docs = list(iter_day_chunks(day.isoformat()))
    print(f"{len(docs)} chunks from {len({d.metadata['publication'] for d in docs})} papers")

    store = get_vector_store()
    for i in track(range(0, len(docs), BATCH), description="Embedding + storing"):
        batch = docs[i : i + BATCH]
        store.add_documents(batch, ids=[d.id for d in batch])

    print(f"Indexed {len(docs)} chunks for {day} in {time.time() - started:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(), help="YYYY-MM-DD (default: today)")
    index_day(parser.parse_args().date)


if __name__ == "__main__":
    main()
