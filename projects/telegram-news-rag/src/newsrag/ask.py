"""Ask a question about the indexed newspapers.

Usage:
    python -m newsrag.ask "What are the editorials saying about AI?"
    python -m newsrag.ask "..." --date 2026-09-24 --paper NYT "FT US"
    python -m newsrag.ask "..." --show-chunks        # print the full retrieved text
    python -m newsrag.ask "..." --retrieve-only      # skip the LLM, just see what retrieval finds

Retrieval: embed the question with bge-m3 and fetch the nearest chunks from Qdrant.
Answer: an LLM answers from those chunks only, citing them as [1], [2], ...
The LLM is set by LLM_PROVIDER in .env: "ollama" (free, local) or "anthropic" (Claude).
"""

import argparse
import re
from dataclasses import dataclass
from datetime import date

from langchain_core.documents import Document
from qdrant_client import models
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from newsrag.config import get_settings
from newsrag.index.store import get_vector_store

console = Console()

SYSTEM_PROMPT = """You answer questions about today's newspapers using only the excerpts provided.

- Name the publication when you use something from it (e.g. "The FT argues...").
- If the excerpts don't cover the question, say so plainly instead of guessing.
- Several excerpts come from OCR of scanned pages and may contain small recognition errors; \
read past them, but don't invent details to fill gaps.
- Distinguish news reporting from opinion or editorial pieces when that's clear from the text."""

# Local models have no built-in citation feature, so we number the excerpts and ask for [n] markers
CITE_INSTRUCTION = "- After each claim, cite the excerpt(s) it came from as [1], [2], etc."


@dataclass
class Answer:
    text: str  # may contain [n] citation markers
    model: str
    usage: str


def build_filter(day: date | None, papers: list[str] | None) -> models.Filter | None:
    must = []
    if day:
        must.append(models.FieldCondition(key="metadata.edition_date", match=models.MatchValue(value=day.isoformat())))
    if papers:
        must.append(models.FieldCondition(key="metadata.publication", match=models.MatchAny(any=papers)))
    return models.Filter(must=must) if must else None


def retrieve(question: str, day: date | None, papers: list[str] | None, k: int) -> list[tuple[Document, float]]:
    store = get_vector_store()
    return store.similarity_search_with_score(question, k=k, filter=build_filter(day, papers))


def source_label(doc: Document) -> str:
    m = doc.metadata
    return f"{m['publication']} · p.{m['page']} · {m.get('headline') or '(no headline)'}"


def answer_with_ollama(question: str, hits: list[tuple[Document, float]]) -> Answer:
    from langchain_ollama import ChatOllama

    settings = get_settings()
    llm = ChatOllama(
        model=settings.ollama_model,
        temperature=0.2,
        num_ctx=16384,  # Ollama's default context is too small for 8 excerpts
        reasoning=False,  # skip qwen3's "thinking" phase: much faster, fine for grounded Q&A
    )
    excerpts = "\n\n".join(f"[{n}] ({source_label(doc)})\n{doc.page_content}" for n, (doc, _) in enumerate(hits, 1))
    reply = llm.invoke(
        [
            ("system", f"{SYSTEM_PROMPT}\n{CITE_INSTRUCTION}"),
            ("human", f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"),
        ]
    )
    u = reply.usage_metadata or {}
    return Answer(reply.content, settings.ollama_model, f"{u.get('input_tokens', '?')} in / {u.get('output_tokens', '?')} out tokens · local, free")


def answer_with_claude(question: str, hits: list[tuple[Document, float]]) -> Answer:
    import anthropic

    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
    # Claude's citations feature: each chunk is a document, and the answer points back to the ones it used
    documents = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": doc.page_content},
            "title": source_label(doc),
            "citations": {"enabled": True},
        }
        for doc, _ in hits
    ]
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": [*documents, {"type": "text", "text": question}]}],
    )
    if response.stop_reason == "refusal":
        return Answer("Claude declined to answer this question.", response.model, "")

    parts = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
            parts.extend(f"[{n}]" for n in sorted({c.document_index + 1 for c in (block.citations or [])}))
    u = response.usage
    return Answer("".join(parts).strip(), response.model, f"{u.input_tokens} in / {u.output_tokens} out tokens")


def render_answer(ans: Answer) -> None:
    # Escape the model's text for rich, then colour the [n] citation markers
    text = re.sub(r"\\\[(\d+)\]", r"[cyan]\\[\1][/cyan]", escape(ans.text))
    console.print(Panel(text, title="Answer", border_style="green"))
    console.print(f"[dim]{ans.model} · {ans.usage}[/dim]")


def render_sources(hits: list[tuple[Document, float]], show_chunks: bool) -> None:
    console.print("\n[bold]Retrieved chunks[/bold]")
    for n, (doc, score) in enumerate(hits, 1):
        ocr = " [dim](OCR)[/dim]" if doc.metadata.get("ocr") else ""
        console.print(f"[cyan]\\[{n}][/cyan] {escape(source_label(doc))}{ocr}  [dim]score {score:.3f}[/dim]")
        body = doc.page_content if show_chunks else doc.page_content[:220].replace("\n", " ") + "…"
        console.print(f"    [dim]{escape(body)}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--date", type=date.fromisoformat, help="Only this edition date (YYYY-MM-DD)")
    parser.add_argument("--paper", nargs="+", help="Only these publications, e.g. NYT 'FT US'")
    parser.add_argument("-k", type=int, default=8, help="How many chunks to retrieve (default 8)")
    parser.add_argument("--show-chunks", action="store_true", help="Print full text of retrieved chunks")
    parser.add_argument("--retrieve-only", action="store_true", help="Skip the LLM; only show retrieval")
    args = parser.parse_args()
    settings = get_settings()

    with console.status("Searching the papers..."):
        hits = retrieve(args.question, args.date, args.paper, args.k)
    if not hits:
        console.print("[yellow]No matching chunks. Is anything indexed for that date/paper?[/yellow]")
        return

    if not args.retrieve_only:
        if settings.llm_provider == "anthropic" and not settings.anthropic_api_key:
            console.print("[yellow]LLM_PROVIDER=anthropic but no ANTHROPIC_API_KEY in .env; showing retrieval only.[/yellow]")
        else:
            model = settings.ollama_model if settings.llm_provider == "ollama" else settings.anthropic_model
            with console.status(f"Asking {model}..."):
                ask = answer_with_ollama if settings.llm_provider == "ollama" else answer_with_claude
                render_answer(ask(args.question, hits))

    render_sources(hits, args.show_chunks)


if __name__ == "__main__":
    main()
