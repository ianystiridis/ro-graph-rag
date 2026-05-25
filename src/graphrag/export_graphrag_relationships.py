from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def export_graphrag_relationships(graph_dir: Path, out_path: Path) -> None:
    relationships_path = graph_dir / "relationships.parquet"

    print(f"Looking for relationships file at: {relationships_path.resolve()}")

    if not relationships_path.exists():
        raise FileNotFoundError(f"Missing file: {relationships_path.resolve()}")

    relationships = pd.read_parquet(relationships_path)

    print("Available columns:")
    print(list(relationships.columns))

    columns_to_export = [
        "source",
        "target",
        "description",
        "weight",
    ]

    existing_columns = [col for col in columns_to_export if col in relationships.columns]

    if not existing_columns:
        raise ValueError("None of the expected relationship columns were found.")

    exported = relationships[existing_columns].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    exported.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\nExported {len(exported)} relationships.")
    print(f"CSV created at: {out_path.resolve()}")
    print("\nPreview:")
    print(exported.head(20).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph_dir",
        default="src/graphrag/output",
        help="Path to GraphRAG output folder containing relationships.parquet",
    )
    parser.add_argument(
        "--out",
        default="data/processed/graphrag_relationships.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    export_graphrag_relationships(
        graph_dir=Path(args.graph_dir),
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()