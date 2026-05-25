import argparse
import chromadb
from baseline.embedding_backends import get_embedder
from baseline.entity_reranker import rerank_vector_results


CHROMA_DIR = "vectordb/chroma_baseline"
COLLECTION_NAME = "rowiki_baseline"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["ollama", "hf"], default="hf")
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-large")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--candidate_k", type=int, default=50)
    args = parser.parse_args()

    embedder = get_embedder(args.backend, args.model)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = embedder.encode([args.query])[0]

    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(args.top_k, args.candidate_k),
    )

    results = rerank_vector_results(args.query, raw_results, top_k=args.top_k)

    print(f"\nQuery: {args.query}")
    if results:
        print(f"Query entities: {results[0].get('query_entities', [])}")
    print()

    for item in results:
        print("=" * 80)
        print(
            f"Rank: {item['rank']} | Rerank score: {item['score']:.3f} | "
            f"Distance: {item['distance']:.3f} | Doc: {item['document_id']} | Chunk: {item['chunk_id']}"
        )
        if item.get("matched_query_entities"):
            print(f"Matched query entities: {item['matched_query_entities']}")
        print("-" * 80)
        print(item["text"][:1200])
        print()


if __name__ == "__main__":
    main()
