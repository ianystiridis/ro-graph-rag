import argparse
import json
from pathlib import Path

import chromadb
from tqdm import tqdm

from baseline.embedding_backends import get_embedder


DATA_PATH = Path("data/processed/chunks_100.jsonl")
CHROMA_DIR = Path("vectordb/chroma_baseline")
COLLECTION_NAME = "rowiki_baseline"
BATCH_SIZE = 16


def load_chunks(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def batchify(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["ollama", "hf"], default="ollama")
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    print("Loading chunks...")
    chunks = load_chunks(DATA_PATH)
    print(f"Loaded {len(chunks)} chunks")

    print(f"Using backend={args.backend}, model={args.model}")
    embedder = get_embedder(args.backend, args.model)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME)

    print("Embedding and storing chunks...")
    for batch in tqdm(list(batchify(chunks, BATCH_SIZE))):
        documents = [row["text"] for row in batch]
        ids = [row["chunk_id"] for row in batch]
        metadatas = [
            {
                "doc_id": row["doc_id"],
                "source": row.get("source", ""),
                "topic": row.get("topic", ""),
                "char_len": row.get("char_len", 0),
            }
            for row in batch
        ]

        embeddings = embedder.encode(documents)

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    print("Done.")
    print(f"Saved collection '{COLLECTION_NAME}' to {CHROMA_DIR}")


if __name__ == "__main__":
    main()