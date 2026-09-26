"""Browser chat over the indexed newspapers.

Run from the project folder:
    streamlit run src/newsrag/app.py

Note: Qdrant local mode allows one process at a time, so stop this app before
running the indexer or `python -m newsrag.ask` (and vice versa).
"""

import json
from datetime import date

import pymupdf
import streamlit as st

from newsrag.ask import answer_with_claude, answer_with_ollama, retrieve
from newsrag.config import get_settings

st.set_page_config(page_title="News RAG", page_icon="📰", layout="wide")
settings = get_settings()


@st.cache_data
def available_editions() -> dict[str, list[str]]:
    """{edition_date: [publications]} for every day that has been parsed."""
    editions = {}
    for manifest in sorted(settings.parsed_dir.glob("*/manifest.jsonl"), reverse=True):
        with manifest.open() as f:
            editions[manifest.parent.name] = sorted({json.loads(line)["publication"] for line in f if line.strip()})
    return editions


@st.cache_data(max_entries=64)
def page_image(edition_date: str, file: str, page: int) -> bytes:
    with pymupdf.open(settings.raw_dir / edition_date / file) as doc:
        return doc[page - 1].get_pixmap(dpi=70).tobytes("png")


# ---- Sidebar: filters -------------------------------------------------------
editions = available_editions()
with st.sidebar:
    st.title("📰 News RAG")
    if not editions:
        st.warning("Nothing parsed yet. Run the fetcher, parser and indexer first.")
        st.stop()

    day = st.selectbox("Edition", list(editions), format_func=lambda d: date.fromisoformat(d).strftime("%a %d %b %Y"))
    papers = st.multiselect("Papers", editions[day], default=editions[day])
    k = st.slider("Chunks to retrieve", 3, 20, 8)
    use_reranker = st.toggle("Rerank (cross-encoder)", value=True, help="Re-scores a wider candidate pool for relevance; slower but much more precise than raw embedding similarity")
    show_pages = st.toggle("Show page images", value=False)

    model = settings.ollama_model if settings.llm_provider == "ollama" else settings.anthropic_model
    st.caption(f"LLM: **{model}** ({settings.llm_provider})  \nEmbeddings: {settings.embed_model}")
    if st.button("Clear chat"):
        st.session_state.messages = []


def render_hits(hits: list[dict], score_label: str = "score") -> None:
    with st.expander(f"Retrieved chunks ({len(hits)})"):
        for n, hit in enumerate(hits, 1):
            meta = hit["metadata"]
            ocr = " · OCR" if meta.get("ocr") else ""
            st.markdown(f"**[{n}] {source_label_from(meta)}**{ocr} · {score_label} `{hit['score']:.3f}`")
            if show_pages and meta.get("page"):
                st.image(page_image(meta["edition_date"], meta["file"], meta["page"]), width=320)
            st.text(hit["text"])
            st.divider()


def source_label_from(meta: dict) -> str:
    return f"{meta['publication']} · p.{meta['page']} · {meta.get('headline') or '(no headline)'}"


# ---- Chat -------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("hits"):
            render_hits(msg["hits"], msg.get("score_label", "score"))
            st.caption(msg["footer"])

if question := st.chat_input("Ask about the papers…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the papers…"):
            hits = retrieve(question, date.fromisoformat(day), papers or None, k, use_reranker=use_reranker)
        if not hits:
            st.warning("No matching chunks for these filters.")
            st.stop()

        if settings.llm_provider == "anthropic" and not settings.anthropic_api_key:
            content, footer = "_No ANTHROPIC_API_KEY set; showing retrieval only._", ""
        else:
            with st.spinner(f"Asking {model}…"):
                ask = answer_with_ollama if settings.llm_provider == "ollama" else answer_with_claude
                ans = ask(question, hits)
            content, footer = ans.text, f"{ans.model} · {ans.usage}"

        stored_hits = [{"text": d.page_content, "metadata": d.metadata, "score": s} for d, s in hits]
        score_label = "relevance" if use_reranker else "score"
        st.markdown(content)
        render_hits(stored_hits, score_label)
        st.caption(footer)

    st.session_state.messages.append(
        {"role": "assistant", "content": content, "hits": stored_hits, "footer": footer, "score_label": score_label}
    )
