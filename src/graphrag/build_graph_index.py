from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import graphrag.api as api
from graphrag.config.load_config import load_config

PROJECT_ROOT = Path("src/graphrag")
CORPUS_PATH = Path("data/processed/corpus_100.jsonl")


def load_corpus_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rows.append(
                {
                    "id": item["doc_id"],
                    "text": item["text"],
                    "title": item.get("doc_id", ""),
                    "creation_date": None,
                    "metadata": {
                        "source": item.get("source", ""),
                        "topic": item.get("topic", ""),
                    },
                }
            )
    return pd.DataFrame(rows)


async def run_index():
    docs_df = load_corpus_jsonl(CORPUS_PATH)
    print(f"Loaded {len(docs_df)} documents from {CORPUS_PATH}")

    config = load_config(PROJECT_ROOT)

    results = await api.build_index(
        config=config,
        verbose=True,
        input_documents=docs_df,
    )

    return results


def summarize_results(results):
    print("\nIndexing workflow results:")
    for result in results:
        status = "success" if result.error is None else f"error: {result.error}"
        print(f" - {result.workflow}: {status}")


if __name__ == "__main__":
    outputs = asyncio.run(run_index())
    summarize_results(outputs)
    print(f"\nDone. Check outputs in: {PROJECT_ROOT / 'output'}")