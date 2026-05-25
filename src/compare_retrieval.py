from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow imports when running: python src/compare_retrieval.py
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Avoid conflict with Microsoft's installed package named `graphrag`.
LOCAL_GRAPHRAG_DIR = SRC_DIR / "graphrag"
if str(LOCAL_GRAPHRAG_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_GRAPHRAG_DIR))

from graph_retriever import GraphRetriever

CHROMA_DIR = "vectordb/chroma_baseline"
COLLECTION_NAME = "rowiki_baseline"
GRAPH_OUTPUT_DIR = "src/graphrag/output"


def retrieve_vector(query: str, backend: str, model: str, top_k: int, candidate_k: int) -> list[dict]:
    import chromadb
    from baseline.embedding_backends import get_embedder
    from baseline.entity_reranker import rerank_vector_results

    embedder = get_embedder(backend, model)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = embedder.encode([query])[0]

    # Retrieve more than top_k, then rerank using entity/keyword matches.
    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(top_k, candidate_k),
    )

    return rerank_vector_results(query, raw_results, top_k=top_k)


def print_vector(results: list[dict]) -> None:
    print("\nVECTOR RESULTS WITH SPACY/ENTITY RERANK")
    if results:
        print(f"Query entities: {results[0].get('query_entities', [])}")
    for item in results:
        print("=" * 80)
        print(
            f"Rank: {item['rank']} | Rerank score: {item['score']:.3f} | "
            f"Distance: {item['distance']:.3f} | Doc: {item['document_id']} | Chunk: {item['chunk_id']}"
        )
        if item.get("matched_query_entities"):
            print(f"Matched query entities: {item['matched_query_entities']}")
        print("-" * 80)
        print(item["text"][:900])
        print()


def print_graph(results: list[dict]) -> None:
    print("\nGRAPH RESULTS")
    for item in results:
        print("=" * 80)
        print(f"Rank: {item['rank']} | Score: {item['score']} | Doc: {item['document_id']} | Text unit: {item['text_unit_id']}")
        print(f"Matched entities: {', '.join(item['matched_entities'][:5])}")
        if item["matched_relationships"]:
            rel = item["matched_relationships"][0]
            print(f"Top relation: {rel['source']} -> {rel['target']} | {rel['description']}")
        print("-" * 80)
        print(item["text"][:900])
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--candidate_k", type=int, default=50, help="Vector candidates retrieved before entity reranking")
    parser.add_argument("--backend", choices=["ollama", "hf"], default="hf")
    parser.add_argument("--model", default="intfloat/multilingual-e5-large")
    parser.add_argument("--skip_vector", action="store_true")
    parser.add_argument("--skip_graph", action="store_true")
    args = parser.parse_args()

    print(f"\nQUERY: {args.query}")

    if not args.skip_vector:
        vector_results = retrieve_vector(args.query, args.backend, args.model, args.top_k, args.candidate_k)
        print_vector(vector_results)

    if not args.skip_graph:
        graph_retriever = GraphRetriever(GRAPH_OUTPUT_DIR)
        graph_results = graph_retriever.retrieve(args.query, top_k=args.top_k)
        print_graph(graph_results)


if __name__ == "__main__":
    main()
