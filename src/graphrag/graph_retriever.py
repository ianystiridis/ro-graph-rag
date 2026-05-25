from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

GRAPH_OUTPUT_DIR = Path("src/graphrag/output")

STOPWORDS = {
    "care", "cine", "unde", "cand", "când", "este", "sunt", "fost", "fosta", "fostă",
    "intr", "într", "din", "dintre", "pentru", "prin", "spre", "despre", "dupa", "după",
    "in", "în", "la", "pe", "cu", "si", "și", "sau", "ale", "al", "ai", "a", "o", "un",
    "unei", "unui", "este", "era", "au", "asupra", "ce", "cum", "cat", "cât",
    "romania", "româniei", "roman", "român", "romana", "română",
}


def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> set[str]:
    return {tok for tok in normalize(text).split() if len(tok) >= 3 and tok not in STOPWORDS}


def ensure_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    return [value]


class GraphRetriever:
    def __init__(self, output_dir: str | Path = GRAPH_OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.entities = pd.read_parquet(self.output_dir / "entities.parquet")
        self.relationships = pd.read_parquet(self.output_dir / "relationships.parquet")
        self.text_units = pd.read_parquet(self.output_dir / "text_units.parquet")

        self.entities["_norm_title"] = self.entities["title"].map(normalize)
        self.entities["_title_tokens"] = self.entities["title"].map(tokens)
        self.entities["_desc_tokens"] = self.entities["description"].fillna("").map(tokens)

        self.text_units["_text_tokens"] = self.text_units["text"].fillna("").map(tokens)
        self.text_by_id = self.text_units.set_index("id", drop=False)

    def retrieve(self, query: str, top_k: int = 5, max_entities: int = 8) -> list[dict]:
        query_norm = normalize(query)
        query_tokens = tokens(query)

        matched_entities = self._match_entities(query_norm, query_tokens, max_entities)
        if matched_entities.empty:
            candidate_units = self.text_units.copy()
            matched_titles: set[str] = set()
            matched_relationships = pd.DataFrame(columns=self.relationships.columns)
        else:
            matched_titles = set(matched_entities["title"].astype(str))
            matched_relationships = self._connected_relationships(matched_titles)
            candidate_unit_ids = self._candidate_text_unit_ids(matched_entities, matched_relationships)
            candidate_units = self.text_units[self.text_units["id"].isin(candidate_unit_ids)].copy()

        # Always add a small lexical fallback so generic/missed entities still work.
        lexical_fallback = self._top_lexical_units(query_tokens, limit=25)
        candidate_units = pd.concat([candidate_units, lexical_fallback], ignore_index=True)
        candidate_units = candidate_units.drop_duplicates(subset=["id"])

        if candidate_units.empty:
            candidate_units = self.text_units.copy()

        ranked = self._rank_text_units(
            candidate_units=candidate_units,
            query_tokens=query_tokens,
            matched_entity_ids=set(matched_entities["id"].astype(str)) if not matched_entities.empty else set(),
            matched_relationship_ids=set(matched_relationships["id"].astype(str)) if not matched_relationships.empty else set(),
        )

        return self._format_results(ranked.head(top_k), matched_entities, matched_relationships)

    def _match_entities(self, query_norm: str, query_tokens: set[str], max_entities: int) -> pd.DataFrame:
        rows = []
        for _, row in self.entities.iterrows():
            title_norm = row["_norm_title"]
            title_tokens = row["_title_tokens"]
            desc_tokens = row["_desc_tokens"]

            title_overlap = len(query_tokens & title_tokens)
            desc_overlap = len(query_tokens & desc_tokens)

            score = 0.0
            if title_norm and title_norm in query_norm:
                score += 20.0
            if query_norm and query_norm in title_norm:
                score += 8.0
            score += title_overlap * 4.0
            score += desc_overlap * 0.3
            score += min(float(row.get("frequency", 0)), 10.0) * 0.05

            if score > 0:
                rows.append((score, row))

        if not rows:
            return self.entities.iloc[0:0].copy()

        rows.sort(key=lambda item: item[0], reverse=True)
        df = pd.DataFrame([r[1] for r in rows[:max_entities]])
        df["match_score"] = [r[0] for r in rows[:max_entities]]
        return df

    def _connected_relationships(self, entity_titles: set[str]) -> pd.DataFrame:
        rels = self.relationships[
            self.relationships["source"].astype(str).isin(entity_titles)
            | self.relationships["target"].astype(str).isin(entity_titles)
        ].copy()
        if "weight" in rels.columns:
            rels = rels.sort_values("weight", ascending=False)
        return rels.head(30)

    def _candidate_text_unit_ids(self, entities_df: pd.DataFrame, rels_df: pd.DataFrame) -> set[str]:
        ids: set[str] = set()

        for value in entities_df.get("text_unit_ids", []):
            ids.update(map(str, ensure_list(value)))

        if not rels_df.empty:
            for value in rels_df.get("text_unit_ids", []):
                ids.update(map(str, ensure_list(value)))

        return ids


    def _top_lexical_units(self, query_tokens: set[str], limit: int = 25) -> pd.DataFrame:
        if not query_tokens:
            return self.text_units.iloc[0:0].copy()

        df = self.text_units.copy()
        df["_lexical_score"] = df["_text_tokens"].map(lambda unit_tokens: len(query_tokens & unit_tokens))
        return df[df["_lexical_score"] > 0].sort_values("_lexical_score", ascending=False).head(limit)

    def _rank_text_units(
        self,
        candidate_units: pd.DataFrame,
        query_tokens: set[str],
        matched_entity_ids: set[str],
        matched_relationship_ids: set[str],
    ) -> pd.DataFrame:
        scored = candidate_units.copy()

        def score_row(row) -> float:
            lexical = len(query_tokens & row["_text_tokens"])
            entity_hits = len(set(map(str, ensure_list(row.get("entity_ids")))) & matched_entity_ids)
            rel_hits = len(set(map(str, ensure_list(row.get("relationship_ids")))) & matched_relationship_ids)
            return lexical * 2.0 + entity_hits * 3.0 + rel_hits * 4.0

        scored["score"] = scored.apply(score_row, axis=1)
        return scored.sort_values("score", ascending=False)

    def _format_results(self, ranked_units: pd.DataFrame, matched_entities: pd.DataFrame, matched_relationships: pd.DataFrame) -> list[dict]:
        entity_titles = matched_entities["title"].astype(str).head(8).tolist() if not matched_entities.empty else []

        rel_preview = []
        if not matched_relationships.empty:
            for _, rel in matched_relationships.head(8).iterrows():
                rel_preview.append(
                    {
                        "source": str(rel.get("source", "")),
                        "target": str(rel.get("target", "")),
                        "description": str(rel.get("description", "")),
                        "weight": float(rel.get("weight", 0.0)),
                    }
                )

        results = []
        for rank, (_, row) in enumerate(ranked_units.iterrows(), start=1):
            results.append(
                {
                    "method": "graph",
                    "rank": rank,
                    "score": float(row.get("score", 0.0)),
                    "text_unit_id": str(row.get("id", "")),
                    "document_id": str(row.get("document_id", "")),
                    "text": str(row.get("text", "")),
                    "matched_entities": entity_titles,
                    "matched_relationships": rel_preview,
                }
            )
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_dir", default=str(GRAPH_OUTPUT_DIR))
    args = parser.parse_args()

    retriever = GraphRetriever(args.output_dir)
    results = retriever.retrieve(args.query, top_k=args.top_k)

    print(f"\nQuery: {args.query}")
    print("\nGRAPH RESULTS")
    for item in results:
        print("=" * 80)
        print(f"Rank: {item['rank']} | Score: {item['score']} | Doc: {item['document_id']}")
        print(f"Text unit: {item['text_unit_id']}")
        print(f"Matched entities: {', '.join(item['matched_entities'][:5])}")
        if item["matched_relationships"]:
            rel = item["matched_relationships"][0]
            print(f"Top relation: {rel['source']} -> {rel['target']} | {rel['description']}")
        print("-" * 80)
        print(item["text"][:1200])
        print()


if __name__ == "__main__":
    main()
