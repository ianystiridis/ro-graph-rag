from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Make local src imports work when running: streamlit run src/app.py
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Avoid conflict with Microsoft's installed package named `graphrag`.
LOCAL_GRAPHRAG_DIR = SRC_DIR / "graphrag"
if str(LOCAL_GRAPHRAG_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_GRAPHRAG_DIR))

from graph_retriever import GraphRetriever
from baseline.embedding_backends import get_embedder
from baseline.entity_reranker import rerank_vector_results
from generation.answer_generation import generate_answer_openai

import chromadb

import streamlit.components.v1 as components
from knowledge_graph.knowledge_graph import (
    extract_knowledge_graph,
    build_vector_retrieval_kg_visualization,
    build_graph_retrieval_visualization,
)


CHROMA_DIR = "vectordb/chroma_baseline"
COLLECTION_NAME = "rowiki_baseline"
GRAPH_OUTPUT_DIR = "src/graphrag/output"


@st.cache_resource(show_spinner=False)
def load_graph_retriever(output_dir: str) -> GraphRetriever:
    return GraphRetriever(output_dir)


@st.cache_resource(show_spinner=False)
def load_embedder(backend: str, model: str):
    return get_embedder(backend, model)


@st.cache_resource(show_spinner=False)
def load_chroma_collection(chroma_dir: str, collection_name: str):
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_collection(collection_name)


def retrieve_vector(query: str, backend: str, model: str, top_k: int, candidate_k: int) -> list[dict[str, Any]]:
    embedder = load_embedder(backend, model)
    collection = load_chroma_collection(CHROMA_DIR, COLLECTION_NAME)

    query_embedding = embedder.encode([query])[0]
    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(top_k, candidate_k),
    )
    return rerank_vector_results(query, raw_results, top_k=top_k)


def retrieve_graph(query: str, top_k: int) -> list[dict[str, Any]]:
    retriever = load_graph_retriever(GRAPH_OUTPUT_DIR)
    return retriever.retrieve(query, top_k=top_k)


def short_id(value: Any, limit: int = 14) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "..."


def render_vector_results(
    results: list[dict[str, Any]],
    vector_kg_viz: dict[str, Any] | None = None,
) -> None:
    st.subheader("Vector retrieval")

    if not results:
        st.warning("No vector results returned.")
        return

    query_entities = results[0].get("query_entities", [])

    if query_entities:
        st.caption("Query entities: " + ", ".join(query_entities))

    for item in results:
        title = (
            f"#{item.get('rank')} | Doc: {item.get('document_id')} | "
            f"Chunk: {short_id(item.get('chunk_id'))} | "
            f"Score: {float(item.get('score', 0.0)):.3f}"
        )

        with st.expander(title, expanded=item.get("rank") == 1):
            if item.get("matched_query_entities"):
                st.markdown(
                    "**Matched query entities:** "
                    + ", ".join(item["matched_query_entities"])
                )

            if "distance" in item:
                st.markdown(f"**Vector distance:** `{float(item['distance']):.3f}`")

            st.write(item.get("text", ""))

    if not vector_kg_viz:
        return

    st.markdown("#### Knowledge graph from vector-retrieved documents")

    st.caption(
        f"Traditional dependency-based KG | "
        f"Nodes: {vector_kg_viz.get('node_count', 0)} | "
        f"Edges extracted: {vector_kg_viz.get('edge_count', 0)} | "
        f"Edges shown: {vector_kg_viz.get('shown_edge_count', 0)}"
    )

    if vector_kg_viz.get("html"):
        components.html(
            vector_kg_viz["html"],
            height=510,
            scrolling=True,
        )
    else:
        st.info(vector_kg_viz.get("empty_reason", "No KG visualization available."))

    with st.expander("Vector KG triples"):
        triples = vector_kg_viz.get("triples", [])

        if triples:
            st.dataframe(triples, use_container_width=True)
        else:
            st.write("No triples available above the current confidence threshold.")


def render_graph_results(
    results: list[dict[str, Any]],
    graph_retrieval_viz: dict[str, Any] | None = None,
) -> None:
    st.subheader("Graph retrieval")

    if not results:
        st.warning("No graph results returned.")
        return

    for item in results:
        title = (
            f"#{item.get('rank')} | Doc: {item.get('document_id')} | "
            f"Text unit: {short_id(item.get('text_unit_id'))} | "
            f"Score: {float(item.get('score', 0.0)):.3f}"
        )

        with st.expander(title, expanded=item.get("rank") == 1):
            entities = item.get("matched_entities", [])

            if entities:
                st.markdown("**Matched entities:** " + ", ".join(entities[:8]))

            relationships = item.get("matched_relationships", [])

            if relationships:
                rel = relationships[0]

                st.markdown(
                    f"**Top relation:** `{rel.get('source', '')}` → `{rel.get('target', '')}`"
                )

                if rel.get("description"):
                    st.caption(rel.get("description"))

            st.write(item.get("text", ""))

    if not graph_retrieval_viz:
        return

    st.markdown("#### Graph retrieval visualization")

    st.caption(
        f"Direct visualization from retrieved graph relationships | "
        f"Nodes: {graph_retrieval_viz.get('node_count', 0)} | "
        f"Edges: {graph_retrieval_viz.get('edge_count', 0)}"
    )

    if graph_retrieval_viz.get("html"):
        components.html(
            graph_retrieval_viz["html"],
            height=510,
            scrolling=True,
        )
    else:
        st.info(
            graph_retrieval_viz.get(
                "empty_reason",
                "No graph retrieval visualization available.",
            )
        )

def render_assistant_message(message: dict[str, Any], message_idx: int, llm_model: str) -> None:
    with st.chat_message("assistant"):
        if message.get("error"):
            st.error(message["error"])
            return

        col1, col2 = st.columns(2)

        with col1:
            render_vector_results(
                message.get("vector_results", []),
                vector_kg_viz=message.get("vector_kg_viz"),
            )

        with col2:
            render_graph_results(
                message.get("graph_results", []),
                graph_retrieval_viz=message.get("graph_retrieval_viz"),
            )

        st.divider()

        if message.get("answer"):
            st.markdown("### Final answer")
            st.write(message["answer"])
        else:
            if st.button("Generate final answer with OpenAI", key=f"generate_answer_{message_idx}"):
                with st.spinner("Generating answer..."):
                    answer = generate_answer_openai(
                        question=message.get("question", ""),
                        vector_results=message.get("vector_results", []),
                        graph_results=message.get("graph_results", []),
                        model=llm_model,
                    )

                st.session_state.messages[message_idx]["answer"] = answer
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="Ro GraphRAG Demo", layout="wide")

    st.title("Romanian GraphRAG Retrieval Demo")
    st.caption(
        "Enter a Romanian question and compare the top retrieved evidence "
        "from vector RAG and graph retrieval."
    )

    with st.sidebar:
        st.header("Settings")

        top_k = st.slider("Top K", min_value=1, max_value=10, value=3)
        candidate_k = st.slider(
            "Vector candidates before reranking",
            min_value=top_k,
            max_value=100,
            value=max(50, top_k),
        )

        backend = st.selectbox("Embedding backend", options=["ollama", "hf"], index=0)
        default_model = "intfloat/multilingual-e5-large" if backend == "hf" else "qwen3-embedding:0.6b"
        model = st.text_input("Embedding model", value=default_model)

        llm_model = st.text_input("OpenAI answer model", value="gpt-4.1-mini")

        st.divider()

        run_vector = st.checkbox("Run vector retrieval", value=True)
        run_graph = st.checkbox("Run graph retrieval", value=True)
        st.divider()

        show_vector_kg = st.checkbox(
            "Show KG from vector retrieval",
            value=True,
        )

        vector_kg_min_confidence = st.slider(
            "Vector KG min confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.60,
            step=0.05,
        )

        vector_kg_max_chars = st.slider(
            "Vector KG max characters",
            min_value=1000,
            max_value=20000,
            value=8000,
            step=1000,
        )

        show_graph_visualization = st.checkbox(
            "Show graph retrieval visualization",
            value=True,
        )

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render previous chat messages
    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        elif message["role"] == "assistant":
            render_assistant_message(message, idx, llm_model)

    prompt = st.chat_input("Ask a Romanian question...")

    if not prompt:
        return

    # Store and render user message
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.write(prompt)

    # Build assistant message
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "question": prompt,
        "vector_results": [],
        "graph_results": [],
        "vector_kg_viz": None,
        "graph_retrieval_viz": None,
        "answer": None,
    }

    try:
        with st.spinner("Retrieving evidence..."):
            if run_vector:
                assistant_message["vector_results"] = retrieve_vector(
                    prompt,
                    backend=backend,
                    model=model,
                    top_k=top_k,
                    candidate_k=candidate_k,
                )

            if run_graph:
                assistant_message["graph_results"] = retrieve_graph(
                    prompt,
                    top_k=top_k,
                )

            if show_vector_kg and assistant_message["vector_results"]:
                assistant_message["vector_kg_viz"] = build_vector_retrieval_kg_visualization(
                    vector_results=assistant_message["vector_results"],
                    extractor=extract_knowledge_graph,
                    min_confidence=vector_kg_min_confidence,
                    max_chars=vector_kg_max_chars,
                    height="460px",
                )

            if show_graph_visualization and assistant_message["graph_results"]:
                assistant_message["graph_retrieval_viz"] = build_graph_retrieval_visualization(
                    graph_results=assistant_message["graph_results"],
                    height="460px",
                )

    except Exception as exc:
        assistant_message["error"] = str(exc)

    # Store assistant message
    st.session_state.messages.append(assistant_message)

    # Render the new assistant message
    render_assistant_message(
        assistant_message,
        len(st.session_state.messages) - 1,
        llm_model,
    )

if __name__ == "__main__":
    main()
