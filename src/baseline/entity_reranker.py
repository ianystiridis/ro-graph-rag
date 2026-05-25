from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Any

# No custom Romanian stopword list here.
# We rely on spaCy token.is_stop + POS tags when a Romanian model is installed.
CONTENT_POS = {"PROPN", "NOUN", "ADJ", "NUM"}
KEYWORD_POS = {"PROPN", "NOUN", "ADJ", "VERB", "NUM"}
BRIDGE_POS = {"ADP", "DET"}


def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def load_spacy_model():
    """Load a Romanian spaCy pipeline if available; fallback to blank tokenizer."""
    try:
        import spacy
    except Exception:
        return None

    for model_name in ("ro_core_news_lg", "ro_core_news_md", "ro_core_news_sm"):
        try:
            return spacy.load(model_name)
        except Exception:
            pass

    # Tokenizer only: no POS/NER, but still better than nothing.
    try:
        return spacy.blank("ro")
    except Exception:
        return None


def has_pos(doc: Any) -> bool:
    return any(bool(tok.pos_) for tok in doc)


def fallback_capitalized_phrases(text: str) -> list[str]:
    """Fallback only. Extract proper-name-like spans without a custom stopword list."""
    tokens = re.findall(r"[A-ZĂÂÎȘȚ][\wĂÂÎȘȚăâîșț\-]*|[a-zăâîșț]{1,4}|\d+", text)
    phrases: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            norm = normalize(" ".join(current))
            # keep multi-token phrases or strong uppercase single tokens like USR
            if len(norm.split()) >= 2 or any(tok.isupper() and len(tok) >= 2 for tok in current):
                phrases.append(" ".join(current))
        current = []

    for tok in tokens:
        starts_upper = bool(re.match(r"^[A-ZĂÂÎȘȚ]", tok)) or tok.isupper()
        short_lower_bridge = tok.islower() and len(tok) <= 4 and bool(current)

        if starts_upper or short_lower_bridge:
            current.append(tok)
        else:
            flush()
    flush()
    return phrases


def extract_pos_phrases(doc: Any) -> list[str]:
    """Extract entity-like noun/proper-noun phrases using POS tags."""
    phrases: list[str] = []
    current: list[Any] = []
    has_content = False

    def flush() -> None:
        nonlocal current, has_content
        if current and has_content:
            # Trim weak bridge tokens from edges.
            while current and current[0].pos_ in BRIDGE_POS:
                current.pop(0)
            while current and current[-1].pos_ in BRIDGE_POS:
                current.pop()
            if current:
                text = " ".join(tok.text for tok in current)
                norm_parts = normalize(text).split()
                # Keep multi-token phrases and strong named tokens.
                if len(norm_parts) >= 2 or any(tok.pos_ == "PROPN" for tok in current):
                    phrases.append(text)
        current = []
        has_content = False

    for tok in doc:
        if tok.is_space or tok.is_punct:
            flush()
            continue

        if tok.pos_ in CONTENT_POS:
            current.append(tok)
            has_content = True
        elif tok.pos_ in BRIDGE_POS and current:
            # Allows "Palatul de Justiție din Suceava" as one phrase.
            current.append(tok)
        else:
            flush()
    flush()
    return phrases


def extract_query_entities(query: str) -> list[str]:
    """Extract query entities using spaCy NER + POS phrase extraction.

    No hand-written stopword list is used. If Romanian POS/NER model is missing,
    fallback to simple capitalized phrase extraction.
    """
    entities: list[str] = []
    nlp = load_spacy_model()

    if nlp is not None:
        try:
            doc = nlp(query)

            # 1) NER entities when the model has NER.
            if "ner" in getattr(nlp, "pipe_names", []):
                entities.extend(ent.text for ent in doc.ents)

            # 2) Noun/proper-noun spans from POS.
            if has_pos(doc):
                entities.extend(extract_pos_phrases(doc))
            else:
                entities.extend(fallback_capitalized_phrases(query))
        except Exception:
            entities.extend(fallback_capitalized_phrases(query))
    else:
        entities.extend(fallback_capitalized_phrases(query))

    # Deduplicate by normalized form.
    cleaned: list[str] = []
    seen: set[str] = set()
    for ent in entities:
        ent = ent.strip()
        norm = normalize(ent)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        cleaned.append(ent)

    return cleaned


def extract_keywords(text: str) -> set[str]:
    """Extract content keywords using spaCy POS and token.is_stop.

    Fallback uses only length/alphanumeric filtering, not a custom stopword list.
    """
    nlp = load_spacy_model()
    if nlp is not None:
        try:
            doc = nlp(text)
            if has_pos(doc):
                keywords: set[str] = set()
                for tok in doc:
                    if tok.is_space or tok.is_punct or tok.is_stop:
                        continue
                    if tok.pos_ in KEYWORD_POS:
                        lemma = tok.lemma_ if tok.lemma_ and tok.lemma_ != "-PRON-" else tok.text
                        norm = normalize(lemma)
                        if len(norm) >= 3:
                            keywords.add(norm)
                return keywords
        except Exception:
            pass

    # Last fallback: no hardcoded stopwords, only token length.
    return {tok for tok in normalize(text).split() if len(tok) >= 4}


def rerank_vector_results(query: str, raw_results: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    """Rerank Chroma candidates using vector distance + spaCy entity/POS overlap."""
    ids = raw_results["ids"][0]
    docs = raw_results["documents"][0]
    metas = raw_results["metadatas"][0]
    distances = raw_results.get("distances", [[]])[0]

    query_entities = extract_query_entities(query)
    query_entity_norms = [normalize(e) for e in query_entities]
    query_keywords = extract_keywords(query)

    ranked: list[dict[str, Any]] = []
    for i, (chunk_id, text, meta) in enumerate(zip(ids, docs, metas)):
        distance = float(distances[i]) if distances else 0.0
        text_norm = normalize(text)
        text_keywords = extract_keywords(text)

        # Chroma distance: lower is better.
        vector_score = 1.0 / (1.0 + max(distance, 0.0))

        exact_entity_hits = [
            ent for ent, ent_norm in zip(query_entities, query_entity_norms)
            if ent_norm and ent_norm in text_norm
        ]

        entity_token_overlap = 0
        for ent_norm in query_entity_norms:
            entity_token_overlap += len(set(ent_norm.split()) & text_keywords)

        keyword_overlap = len(query_keywords & text_keywords)

        # Entity exact match matters most; POS keyword overlap is a smaller tie-breaker.
        # describe the process
        entity_score = 4.0 * len(exact_entity_hits) + 0.5 * entity_token_overlap
        keyword_score = 0.12 * keyword_overlap
        final_score = vector_score + entity_score + keyword_score

        ranked.append(
            {
                "method": "vector_pos_entity_rerank",
                "rank": 0,
                "score": final_score,
                "distance": distance,
                "chunk_id": chunk_id,
                "document_id": meta.get("doc_id"),
                "topic": meta.get("topic", ""),
                "text": text,
                "query_entities": query_entities,
                "matched_query_entities": exact_entity_hits,
                "keyword_overlap": keyword_overlap,
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    for rank, item in enumerate(ranked[:top_k], start=1):
        item["rank"] = rank
    return ranked[:top_k]
