
import argparse
import json
import os
from pathlib import Path

import chromadb
from tqdm import tqdm

from baseline.embedding_backends import get_embedder


DATA_PATH = Path("data/processed/chunks_100.jsonl")
COLLECTION_NAME = "rowiki_baseline"
BATCH_SIZE = 16


def iter_chunks(path: Path):
    """
    Stream chunks from JSONL instead of loading the whole file into memory.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def batchify(iterator, batch_size: int):
    batch = []

    for item in iterator:
        batch.append(item)

        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def count_lines(path: Path) -> int:
    """
    Used only for tqdm progress.
    This avoids keeping the JSONL content in memory.
    """
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--backend", choices=["ollama", "hf"], default="ollama")
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--data-path", type=str, default=str(DATA_PATH))
    parser.add_argument("--collection-name", type=str, default=COLLECTION_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)

    parser.add_argument("--chroma-host", type=str, default=os.getenv("CHROMA_HOST", "localhost"))
    parser.add_argument("--chroma-port", type=int, default=int(os.getenv("CHROMA_PORT", "8000")))

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the Chroma collection before ingesting."
    )

    args = parser.parse_args()

    data_path = Path(args.data_path)

    print(f"Using backend={args.backend}, model={args.model}")
    embedder = get_embedder(args.backend, args.model)

    print(f"Connecting to Chroma at {args.chroma_host}:{args.chroma_port}")
    client = chromadb.HttpClient(
        host=args.chroma_host,
        port=args.chroma_port,
    )

    client.heartbeat()
    print("Connected to Chroma server.")

    existing_collections = [c.name for c in client.list_collections()]

    if args.reset and args.collection_name in existing_collections:
        print(f"Deleting existing collection: {args.collection_name}")
        client.delete_collection(args.collection_name)

    collection = client.get_or_create_collection(name=args.collection_name)

    total_chunks = count_lines(data_path)
    total_batches = (total_chunks + args.batch_size - 1) // args.batch_size

    print(f"Ingesting {total_chunks} chunks into collection '{args.collection_name}'")

    for batch in tqdm(
        batchify(iter_chunks(data_path), args.batch_size),
        total=total_batches,
        desc="Embedding and storing chunks",
    ):
        documents = [row["text"] for row in batch]
        ids = [str(row["chunk_id"]) for row in batch]

        metadatas = [
            {
                "doc_id": str(row["doc_id"]),
                "source": row.get("source", ""),
                "topic": row.get("topic", ""),
                "char_len": row.get("char_len", 0),
            }
            for row in batch
        ]

        embeddings = embedder.encode(documents)

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    print("Done.")
    print(f"Collection '{args.collection_name}' is stored in the local Chroma Docker volume.")


if __name__ == "__main__":
    main()