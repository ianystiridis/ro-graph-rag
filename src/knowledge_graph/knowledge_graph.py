from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

import networkx as nx
import stanza

# ---------------------------------------------------------------------
# Visualization utilities for Streamlit integration
# ---------------------------------------------------------------------

import html
import hashlib
import os
import tempfile
from pyvis.network import Network


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class Node:
    id: str
    label: str
    type: str
    source: str
    mentions: List[str]


@dataclass
class Edge:
    source: str
    target: str
    label: str
    evidence: str
    sentence_id: int
    confidence: float
    rule: str


# ---------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------

def strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_surface(text: str) -> str:
    """
    Used only for merging equivalent local surface forms.
    Example:
        "București" -> "bucuresti"
    """
    text = strip_diacritics(text.lower())
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_safe_id(prefix: str, label: str) -> str:
    """
    Creates stable IDs for local graph nodes.
    Example:
        "Ion Popescu" -> "ent_ion_popescu_1a2b3c"
    """
    norm = normalize_surface(label)
    slug = re.sub(r"\W+", "_", norm).strip("_")[:48] or "node"
    digest = hashlib.md5(norm.encode("utf-8")).hexdigest()[:6]
    return f"{prefix}_{slug}_{digest}"


def safe_relation_label(label: str) -> str:
    """
    Converts extracted relation labels into graph-safe labels.
    Example:
        "lucra la" -> "lucra_la"
    """
    label = strip_diacritics(label.lower())
    label = re.sub(r"[^\w]+", "_", label).strip("_")
    return label or "related_to"


# ---------------------------------------------------------------------
# Stanza loading
# ---------------------------------------------------------------------

def load_stanza_pipeline(use_ner: bool = True) -> Tuple[Any, bool]:
    """
    Loads Romanian Stanza pipeline.

    Returns:
        nlp, ner_enabled

    If NER is unavailable, it falls back to dependency parsing only.
    """
    if use_ner:
        processors = "tokenize,pos,lemma,depparse,ner"
    else:
        processors = "tokenize,pos,lemma,depparse"

    try:
        nlp = stanza.Pipeline(
            lang="ro",
            processors=processors,
            use_gpu=False,
            verbose=False,
        )
        return nlp, use_ner

    except Exception:
        stanza.download("ro", verbose=False)

    try:
        nlp = stanza.Pipeline(
            lang="ro",
            processors=processors,
            use_gpu=False,
            verbose=False,
        )
        return nlp, use_ner

    except Exception:
        fallback_processors = "tokenize,pos,lemma,depparse"
        nlp = stanza.Pipeline(
            lang="ro",
            processors=fallback_processors,
            use_gpu=False,
            verbose=False,
        )
        return nlp, False


# ---------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------

def sentence_text(sent: Any) -> str:
    if getattr(sent, "text", None):
        return sent.text
    return " ".join(w.text for w in sent.words)


def build_children_index(sent: Any) -> Dict[int, List[Any]]:
    children: Dict[int, List[Any]] = {}

    for word in sent.words:
        head = int(word.head)
        children.setdefault(head, []).append(word)

    return children


def deprel(word: Any) -> str:
    return getattr(word, "deprel", "") or ""


def deprel_base(word: Any) -> str:
    return deprel(word).split(":")[0]


def children_with_deprel(
    children: Dict[int, List[Any]],
    head_id: int,
    prefixes: Sequence[str],
) -> List[Any]:
    result = []

    for child in children.get(head_id, []):
        rel = deprel(child)

        if any(rel == p or rel.startswith(p + ":") for p in prefixes):
            result.append(child)

    return result


def get_case_marker(word: Any, children: Dict[int, List[Any]]) -> Optional[str]:
    """
    Gets preposition/case marker for an oblique or nominal modifier.

    Examples:
        "în București" -> "în"
        "de la companie" -> "de_la"
    """
    markers = []

    for child in children.get(int(word.id), []):
        if deprel_base(child) == "case":
            markers.append(child)

            for fixed in children.get(int(child.id), []):
                if deprel_base(fixed) == "fixed":
                    markers.append(fixed)

    if not markers:
        return None

    markers = sorted(markers, key=lambda w: int(w.id))
    return "_".join(w.text.lower() for w in markers)


def is_content_head(word: Any) -> bool:
    return getattr(word, "upos", "") in {
        "NOUN",
        "PROPN",
        "PRON",
        "NUM",
        "ADJ",
    }


def is_entity_like_head(word: Any) -> bool:
    return getattr(word, "upos", "") in {
        "NOUN",
        "PROPN",
        "NUM",
    }


def phrase_for_word(sent: Any, word: Any, children: Dict[int, List[Any]]) -> str:
    """
    Builds a compact phrase around a syntactic head.

    This uses only generic dependency labels, not dictionaries.

    Examples:
        Ion Popescu
        Universitatea Babeș-Bolyai
        7000 de euro
        avocatul Ion Popescu
    """
    allowed_base = {
        "flat",
        "compound",
        "fixed",
        "name",
        "amod",
        "det",
        "nummod",
    }

    allowed_full = {
        "nmod:poss",
        "nummod:gov",
    }

    collected = {int(word.id)}
    stack = [word]

    while stack:
        current = stack.pop()

        for child in children.get(int(current.id), []):
            rel = deprel(child)

            if rel in allowed_full or deprel_base(child) in allowed_base:
                collected.add(int(child.id))
                stack.append(child)

    id_to_word = {int(w.id): w for w in sent.words}
    pieces = [
        id_to_word[i].text
        for i in sorted(collected)
        if i in id_to_word
    ]

    phrase = " ".join(pieces)
    phrase = re.sub(r"\s+", " ", phrase).strip()

    return phrase or word.text


# ---------------------------------------------------------------------
# NER helpers
# ---------------------------------------------------------------------

def extract_ner_spans(doc: Any) -> List[Dict[str, str]]:
    """
    Extracts NER spans from Stanza, if available.
    """
    spans: List[Dict[str, str]] = []

    for ent in getattr(doc, "ents", []) or []:
        text = getattr(ent, "text", "")
        ent_type = getattr(ent, "type", "NAMED_ENTITY")

        if text:
            spans.append(
                {
                    "text": text,
                    "type": ent_type,
                }
            )

    return spans


def infer_node_type(
    label: str,
    head_word: Optional[Any],
    ner_spans: List[Dict[str, str]],
) -> Tuple[str, str]:
    """
    Infers node type using:
    1. Stanza NER, if available
    2. Universal POS tag

    No domain dictionaries are used.
    """
    norm = normalize_surface(label)

    for ent in ner_spans:
        ent_norm = normalize_surface(ent["text"])

        if norm == ent_norm or ent_norm in norm or norm in ent_norm:
            return ent.get("type", "NAMED_ENTITY"), "stanza_ner"

    if head_word is not None:
        upos = getattr(head_word, "upos", "")

        if upos == "PROPN":
            return "PROPER_NOUN", "dependency_phrase"

        if upos == "NUM":
            return "NUMBER", "dependency_phrase"

        if upos == "NOUN":
            return "NOUN_PHRASE", "dependency_phrase"

        if upos == "ADJ":
            return "ATTRIBUTE", "dependency_phrase"

        if upos == "PRON":
            return "PRONOUN", "dependency_phrase"

    return "UNKNOWN", "dependency_phrase"


# ---------------------------------------------------------------------
# Graph accumulator
# ---------------------------------------------------------------------

class GraphAccumulator:
    def __init__(self, ner_spans: List[Dict[str, str]]) -> None:
        self.nodes_by_norm: Dict[str, Node] = {}
        self.edges_seen: set[Tuple[str, str, str, str]] = set()
        self.edges: List[Edge] = []
        self.ner_spans = ner_spans

    def get_or_create_node(
        self,
        label: str,
        head_word: Optional[Any] = None,
    ) -> Node:
        label = re.sub(r"\s+", " ", label).strip()

        if not label:
            label = "UNKNOWN"

        norm = normalize_surface(label)

        if not norm:
            norm = label.lower().strip()

        if norm in self.nodes_by_norm:
            node = self.nodes_by_norm[norm]

            if label not in node.mentions:
                node.mentions.append(label)

            return node

        node_type, source = infer_node_type(
            label=label,
            head_word=head_word,
            ner_spans=self.ner_spans,
        )

        node = Node(
            id=make_safe_id("ent", label),
            label=label,
            type=node_type,
            source=source,
            mentions=[label],
        )

        self.nodes_by_norm[norm] = node

        return node

    def add_edge(
        self,
        source_node: Node,
        target_node: Node,
        label: str,
        evidence: str,
        sentence_id: int,
        confidence: float,
        rule: str,
    ) -> None:
        if source_node.id == target_node.id:
            return

        clean_label = safe_relation_label(label)

        key = (
            source_node.id,
            target_node.id,
            clean_label,
            evidence,
        )

        if key in self.edges_seen:
            return

        self.edges_seen.add(key)

        self.edges.append(
            Edge(
                source=source_node.id,
                target=target_node.id,
                label=clean_label,
                evidence=evidence,
                sentence_id=sentence_id,
                confidence=round(float(confidence), 3),
                rule=rule,
            )
        )

    def to_graph_dict(
        self,
        include_isolated_ner_nodes: bool = False,
    ) -> Dict[str, Any]:
        if include_isolated_ner_nodes:
            for ent in self.ner_spans:
                self.get_or_create_node(ent["text"], None)

        used_ids = {
            edge.source
            for edge in self.edges
        } | {
            edge.target
            for edge in self.edges
        }

        nodes = [
            asdict(node)
            for node in self.nodes_by_norm.values()
            if include_isolated_ner_nodes or node.id in used_ids
        ]

        return {
            "nodes": nodes,
            "edges": [asdict(edge) for edge in self.edges],
        }


# ---------------------------------------------------------------------
# Relation extraction rules
# ---------------------------------------------------------------------

def add_active_verb_relations(
    sent: Any,
    sent_id: int,
    verb: Any,
    children: Dict[int, List[Any]],
    graph: GraphAccumulator,
) -> None:
    """
    Pattern 1:
        subject + verb + object

    Example:
        Compania Alfa a cumpărat firma Beta.
        Compania Alfa --cumpăra--> firma Beta

    Pattern 2:
        subject + verb + oblique/prepositional phrase

    Example:
        Maria lucrează la universitate.
        Maria --lucra_la--> universitate
    """
    subjects = [
        w
        for w in children_with_deprel(children, int(verb.id), ["nsubj", "csubj"])
        if "pass" not in deprel(w)
    ]

    objects = children_with_deprel(
        children,
        int(verb.id),
        ["obj", "iobj"],
    )

    obliques = children_with_deprel(
        children,
        int(verb.id),
        ["obl"],
    )

    if not subjects:
        return

    evidence = sentence_text(sent)
    verb_label = getattr(verb, "lemma", None) or verb.text

    for subj in subjects:
        subj_label = phrase_for_word(sent, subj, children)
        subj_node = graph.get_or_create_node(subj_label, subj)

        for obj in objects:
            if not is_content_head(obj):
                continue

            obj_label = phrase_for_word(sent, obj, children)
            obj_node = graph.get_or_create_node(obj_label, obj)

            graph.add_edge(
                source_node=subj_node,
                target_node=obj_node,
                label=verb_label,
                evidence=evidence,
                sentence_id=sent_id,
                confidence=0.85,
                rule="active_subject_verb_object",
            )

        for obl in obliques:
            if not is_content_head(obl):
                continue

            prep = get_case_marker(obl, children)

            if prep:
                rel = f"{verb_label}_{prep}"
            else:
                rel = verb_label

            obl_label = phrase_for_word(sent, obl, children)
            obl_node = graph.get_or_create_node(obl_label, obl)

            graph.add_edge(
                source_node=subj_node,
                target_node=obl_node,
                label=rel,
                evidence=evidence,
                sentence_id=sent_id,
                confidence=0.74,
                rule="active_subject_verb_oblique",
            )


def add_passive_verb_relations(
    sent: Any,
    sent_id: int,
    verb: Any,
    children: Dict[int, List[Any]],
    graph: GraphAccumulator,
) -> None:
    """
    Passive pattern:
        passive subject + verb + oblique agent

    Example:
        Avocatul a fost reținut de procurori.
        procurori --reține_de--> avocatul

    Direction is reversed so the agent points to the patient.
    """
    passive_subjects = [
        w
        for w in children_with_deprel(children, int(verb.id), ["nsubj"])
        if "pass" in deprel(w)
    ]

    if not passive_subjects:
        return

    obliques = children_with_deprel(
        children,
        int(verb.id),
        ["obl"],
    )

    if not obliques:
        return

    evidence = sentence_text(sent)
    verb_label = getattr(verb, "lemma", None) or verb.text

    for patient in passive_subjects:
        patient_label = phrase_for_word(sent, patient, children)
        patient_node = graph.get_or_create_node(patient_label, patient)

        for agent in obliques:
            if not is_content_head(agent):
                continue

            prep = get_case_marker(agent, children)

            if prep:
                rel = f"{verb_label}_{prep}"
            else:
                rel = verb_label

            agent_label = phrase_for_word(sent, agent, children)
            agent_node = graph.get_or_create_node(agent_label, agent)

            graph.add_edge(
                source_node=agent_node,
                target_node=patient_node,
                label=rel,
                evidence=evidence,
                sentence_id=sent_id,
                confidence=0.78,
                rule="passive_agent_verb_patient",
            )


def add_copula_relations(
    sent: Any,
    sent_id: int,
    predicate: Any,
    children: Dict[int, List[Any]],
    graph: GraphAccumulator,
) -> None:
    """
    Copula pattern:
        subject + copula + predicate

    Example:
        Ion Popescu este avocat.
        Ion Popescu --fi--> avocat
    """
    copulas = children_with_deprel(
        children,
        int(predicate.id),
        ["cop"],
    )

    subjects = children_with_deprel(
        children,
        int(predicate.id),
        ["nsubj", "csubj"],
    )

    if not copulas or not subjects:
        return

    evidence = sentence_text(sent)
    copula_label = getattr(copulas[0], "lemma", None) or copulas[0].text

    for subj in subjects:
        subj_label = phrase_for_word(sent, subj, children)
        pred_label = phrase_for_word(sent, predicate, children)

        subj_node = graph.get_or_create_node(subj_label, subj)
        pred_node = graph.get_or_create_node(pred_label, predicate)

        graph.add_edge(
            source_node=subj_node,
            target_node=pred_node,
            label=copula_label,
            evidence=evidence,
            sentence_id=sent_id,
            confidence=0.82,
            rule="copula_subject_predicate",
        )


def add_nominal_modifier_relations(
    sent: Any,
    sent_id: int,
    head: Any,
    children: Dict[int, List[Any]],
    graph: GraphAccumulator,
) -> None:
    """
    Nominal modifier pattern.

    Example:
        avocat în București
        avocat --în--> București
    """
    if not is_entity_like_head(head):
        return

    modifiers = children_with_deprel(
        children,
        int(head.id),
        ["nmod"],
    )

    if not modifiers:
        return

    evidence = sentence_text(sent)

    for mod in modifiers:
        if not is_content_head(mod):
            continue

        prep = get_case_marker(mod, children)
        rel = prep or "related_to"

        head_label = phrase_for_word(sent, head, children)
        mod_label = phrase_for_word(sent, mod, children)

        head_node = graph.get_or_create_node(head_label, head)
        mod_node = graph.get_or_create_node(mod_label, mod)

        graph.add_edge(
            source_node=head_node,
            target_node=mod_node,
            label=rel,
            evidence=evidence,
            sentence_id=sent_id,
            confidence=0.62,
            rule="nominal_modifier",
        )


def add_apposition_relations(
    sent: Any,
    sent_id: int,
    head: Any,
    children: Dict[int, List[Any]],
    graph: GraphAccumulator,
) -> None:
    """
    Apposition pattern.

    Example:
        Ion Popescu, avocat
        Ion Popescu --appos--> avocat
    """
    appositives = children_with_deprel(
        children,
        int(head.id),
        ["appos"],
    )

    if not appositives:
        return

    evidence = sentence_text(sent)

    for app in appositives:
        head_label = phrase_for_word(sent, head, children)
        app_label = phrase_for_word(sent, app, children)

        head_node = graph.get_or_create_node(head_label, head)
        app_node = graph.get_or_create_node(app_label, app)

        graph.add_edge(
            source_node=head_node,
            target_node=app_node,
            label="appos",
            evidence=evidence,
            sentence_id=sent_id,
            confidence=0.72,
            rule="apposition",
        )


# ---------------------------------------------------------------------
# Main KG extraction function
# ---------------------------------------------------------------------

def extract_knowledge_graph(
    text: str,
    nlp: Optional[Any] = None,
    include_isolated_ner_nodes: bool = False,
    use_ner: bool = True,
) -> Dict[str, Any]:
    """
    Extracts a property knowledge graph from Romanian text.

    Parameters
    ----------
    text:
        Romanian input text.

    nlp:
        Optional already-loaded Stanza pipeline.
        Useful if you call this many times from a web app.

    include_isolated_ner_nodes:
        If True, includes named entities even when no relation was extracted.

    use_ner:
        Whether to try loading Stanza NER.

    Returns
    -------
    dict:
        {
            "nodes": [...],
            "edges": [...],
            "metadata": {...}
        }
    """
    if not text or not text.strip():
        return {
            "nodes": [],
            "edges": [],
            "metadata": {
                "language": "ro",
                "method": "traditional_nlp_dependency_rules",
                "llm_used": False,
                "manual_domain_dictionaries_used": False,
                "error": "Empty input text",
            },
        }

    ner_enabled = False

    if nlp is None:
        nlp, ner_enabled = load_stanza_pipeline(use_ner=use_ner)
    else:
        ner_enabled = use_ner

    doc = nlp(text)

    ner_spans = extract_ner_spans(doc) if ner_enabled else []
    graph = GraphAccumulator(ner_spans=ner_spans)

    for sent_id, sent in enumerate(doc.sentences, start=1):
        children = build_children_index(sent)

        for word in sent.words:
            upos = getattr(word, "upos", "")

            if upos in {"VERB", "AUX"}:
                add_active_verb_relations(
                    sent=sent,
                    sent_id=sent_id,
                    verb=word,
                    children=children,
                    graph=graph,
                )

                add_passive_verb_relations(
                    sent=sent,
                    sent_id=sent_id,
                    verb=word,
                    children=children,
                    graph=graph,
                )

            if upos in {"NOUN", "ADJ", "PROPN"}:
                add_copula_relations(
                    sent=sent,
                    sent_id=sent_id,
                    predicate=word,
                    children=children,
                    graph=graph,
                )

                add_nominal_modifier_relations(
                    sent=sent,
                    sent_id=sent_id,
                    head=word,
                    children=children,
                    graph=graph,
                )

                add_apposition_relations(
                    sent=sent,
                    sent_id=sent_id,
                    head=word,
                    children=children,
                    graph=graph,
                )

    result = graph.to_graph_dict(
        include_isolated_ner_nodes=include_isolated_ner_nodes
    )

    result["metadata"] = {
        "language": "ro",
        "method": "traditional_nlp_dependency_rules",
        "llm_used": False,
        "manual_domain_dictionaries_used": False,
        "ner_enabled": ner_enabled,
        "node_count": len(result["nodes"]),
        "edge_count": len(result["edges"]),
    }

    return result


# ---------------------------------------------------------------------
# Optional NetworkX conversion
# ---------------------------------------------------------------------

def to_networkx(
    graph_data: Dict[str, Any],
    min_confidence: float = 0.0,
) -> nx.DiGraph:
    """
    Converts the extracted property graph to a NetworkX directed graph.
    Useful for visualization later.
    """
    G = nx.DiGraph()

    filtered_edges = [
        edge
        for edge in graph_data.get("edges", [])
        if float(edge.get("confidence", 0.0)) >= min_confidence
    ]

    used_node_ids = {
        edge["source"]
        for edge in filtered_edges
    } | {
        edge["target"]
        for edge in filtered_edges
    }

    for node in graph_data.get("nodes", []):
        if node["id"] in used_node_ids:
            G.add_node(
                node["id"],
                label=node.get("label", node["id"]),
                type=node.get("type", "UNKNOWN"),
                source=node.get("source", ""),
                mentions=node.get("mentions", []),
            )

    for edge in filtered_edges:
        if edge["source"] in G.nodes and edge["target"] in G.nodes:
            G.add_edge(
                edge["source"],
                edge["target"],
                label=edge.get("label", ""),
                evidence=edge.get("evidence", ""),
                confidence=edge.get("confidence", 0.0),
                rule=edge.get("rule", ""),
                sentence_id=edge.get("sentence_id", ""),
            )

    return G


# ---------------------------------------------------------------------
# Optional helpers for app integration
# ---------------------------------------------------------------------

def graph_to_triples(graph_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Converts graph JSON into readable triples.

    Returns:
        [
            {
                "source": "...",
                "relation": "...",
                "target": "...",
                "confidence": ...,
                "evidence": "..."
            }
        ]
    """
    node_by_id = {
        node["id"]: node
        for node in graph_data.get("nodes", [])
    }

    triples = []

    for edge in graph_data.get("edges", []):
        source_node = node_by_id.get(edge["source"], {})
        target_node = node_by_id.get(edge["target"], {})

        triples.append(
            {
                "source": source_node.get("label", edge["source"]),
                "relation": edge.get("label", ""),
                "target": target_node.get("label", edge["target"]),
                "confidence": edge.get("confidence", 0.0),
                "rule": edge.get("rule", ""),
                "sentence_id": edge.get("sentence_id", ""),
                "evidence": edge.get("evidence", ""),
            }
        )

    return triples


def _vis_strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _vis_normalize_surface(text: str) -> str:
    text = _vis_strip_diacritics(str(text).lower())
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _vis_make_safe_id(prefix: str, label: str) -> str:
    norm = _vis_normalize_surface(label)
    slug = re.sub(r"\W+", "_", norm).strip("_")[:48] or "node"
    digest = hashlib.md5(norm.encode("utf-8")).hexdigest()[:6]
    return f"{prefix}_{slug}_{digest}"


def _vis_safe_relation_label(label: str) -> str:
    label = _vis_strip_diacritics(str(label).lower())
    label = re.sub(r"[^\w]+", "_", label).strip("_")
    return label or "related_to"


def vector_results_to_text(
    results: list[dict[str, Any]],
    max_chars: int = 8000,
) -> str:
    """
    Converts vector-retrieved chunks into one text input for KG creation.

    This is the text that will be passed to your traditional KG extractor.
    """
    pieces: list[str] = []

    for item in results:
        text = str(item.get("text", "")).strip()

        if not text:
            continue

        doc_id = item.get("document_id", "unknown_doc")
        chunk_id = item.get("chunk_id", "unknown_chunk")

        pieces.append(
            f"[Document: {doc_id} | Chunk: {chunk_id}]\n{text}"
        )

    combined = "\n\n".join(pieces)

    return combined[:max_chars]


def kg_to_networkx(
    graph_data: dict[str, Any],
    min_confidence: float = 0.0,
) -> nx.DiGraph:
    """
    Converts your extracted property graph into a NetworkX directed graph.

    Expected graph_data shape:
    {
        "nodes": [
            {"id": "...", "label": "...", "type": "..."}
        ],
        "edges": [
            {"source": "...", "target": "...", "label": "...", "confidence": ...}
        ]
    }
    """
    G = nx.DiGraph()

    filtered_edges = [
        edge
        for edge in graph_data.get("edges", [])
        if float(edge.get("confidence", 0.0)) >= min_confidence
    ]

    used_node_ids = {edge["source"] for edge in filtered_edges} | {
        edge["target"] for edge in filtered_edges
    }

    for node in graph_data.get("nodes", []):
        node_id = node.get("id")

        if not node_id or node_id not in used_node_ids:
            continue

        G.add_node(
            node_id,
            label=node.get("label", node_id),
            type=node.get("type", "UNKNOWN"),
            source=node.get("source", ""),
            mentions=node.get("mentions", []),
        )

    for edge in filtered_edges:
        source = edge.get("source")
        target = edge.get("target")

        if source in G.nodes and target in G.nodes:
            G.add_edge(
                source,
                target,
                label=edge.get("label", ""),
                evidence=edge.get("evidence", ""),
                confidence=edge.get("confidence", 0.0),
                rule=edge.get("rule", ""),
                sentence_id=edge.get("sentence_id", ""),
            )

    return G


def graph_retrieval_results_to_networkx(
    results: list[dict[str, Any]],
) -> nx.DiGraph:
    """
    Builds a graph directly from graph retrieval output.

    This is intentionally different from the vector-side KG.

    Vector side:
        retrieved text -> traditional KG extraction -> visualization

    Graph side:
        retrieved graph relationships -> visualization directly
    """
    G = nx.DiGraph()

    for item in results:
        rank = item.get("rank", "")
        doc_id = item.get("document_id", "")
        score = float(item.get("score", 0.0) or 0.0)

        for entity in item.get("matched_entities", []) or []:
            entity_label = str(entity).strip()

            if not entity_label:
                continue

            entity_id = _vis_make_safe_id("graph_ent", entity_label)

            G.add_node(
                entity_id,
                label=entity_label,
                type="MATCHED_ENTITY",
                source=f"graph_result_rank_{rank}",
                title=f"Matched entity<br>Doc: {doc_id}<br>Rank: {rank}",
            )

        for rel in item.get("matched_relationships", []) or []:
            source_label = str(rel.get("source", "")).strip()
            target_label = str(rel.get("target", "")).strip()

            if not source_label or not target_label:
                continue

            source_id = _vis_make_safe_id("graph_ent", source_label)
            target_id = _vis_make_safe_id("graph_ent", target_label)

            G.add_node(
                source_id,
                label=source_label,
                type="GRAPH_ENTITY",
                source=f"graph_result_rank_{rank}",
            )

            G.add_node(
                target_id,
                label=target_label,
                type="GRAPH_ENTITY",
                source=f"graph_result_rank_{rank}",
            )

            description = str(rel.get("description", "") or "")

            relation_label = (
                rel.get("relation")
                or rel.get("label")
                or rel.get("type")
                or "retrieved_relation"
            )

            G.add_edge(
                source_id,
                target_id,
                label=_vis_safe_relation_label(str(relation_label)),
                evidence=description,
                confidence=score,
                rule="graph_retrieval_relationship",
                sentence_id=rank,
            )

    return G


def make_pyvis_html(
    G: nx.DiGraph,
    height: str = "460px",
) -> str:
    """
    Converts a NetworkX graph into embeddable PyVis HTML.
    """
    net = Network(
        height=height,
        width="100%",
        directed=True,
        notebook=False,
        cdn_resources="in_line",
    )

    net.barnes_hut()

    for node_id, attrs in G.nodes(data=True):
        label = str(attrs.get("label", node_id))
        node_type = str(attrs.get("type", "UNKNOWN"))
        source = str(attrs.get("source", ""))

        title = (
            f"<b>{html.escape(label)}</b><br>"
            f"Type: {html.escape(node_type)}<br>"
            f"Source: {html.escape(source)}"
        )

        if attrs.get("mentions"):
            title += f"<br>Mentions: {html.escape(str(attrs.get('mentions')))}"

        if attrs.get("title"):
            title += "<br>" + str(attrs["title"])

        net.add_node(
            node_id,
            label=label,
            title=title,
            group=node_type,
        )

    for source, target, attrs in G.edges(data=True):
        label = str(attrs.get("label", ""))
        evidence = str(attrs.get("evidence", ""))
        confidence = attrs.get("confidence", "")

        title = (
            f"<b>Relation:</b> {html.escape(label)}<br>"
            f"<b>Confidence:</b> {html.escape(str(confidence))}<br>"
            f"<b>Rule:</b> {html.escape(str(attrs.get('rule', '')))}<br>"
            f"<b>Evidence:</b> {html.escape(evidence[:1200])}"
        )

        net.add_edge(
            source,
            target,
            label=label,
            title=title,
            arrows="to",
        )

    fd, path = tempfile.mkstemp(suffix=".html")
    os.close(fd)

    try:
        net.save_graph(path)

        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def kg_graph_to_triples(
    graph_data: dict[str, Any],
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Converts extracted KG JSON into a table-friendly triple list.
    """
    node_by_id = {
        node["id"]: node
        for node in graph_data.get("nodes", [])
        if node.get("id")
    }

    triples: list[dict[str, Any]] = []

    for edge in graph_data.get("edges", []):
        if float(edge.get("confidence", 0.0)) < min_confidence:
            continue

        source = node_by_id.get(edge.get("source"), {})
        target = node_by_id.get(edge.get("target"), {})

        triples.append(
            {
                "source": source.get("label", edge.get("source", "")),
                "relation": edge.get("label", ""),
                "target": target.get("label", edge.get("target", "")),
                "confidence": edge.get("confidence", 0.0),
                "rule": edge.get("rule", ""),
                "evidence": edge.get("evidence", ""),
            }
        )

    return triples


def build_vector_retrieval_kg_visualization(
    vector_results: list[dict[str, Any]],
    extractor: Callable[[str], dict[str, Any]],
    min_confidence: float = 0.60,
    max_chars: int = 8000,
    height: str = "460px",
) -> dict[str, Any]:
    """
    High-level helper for the left side of the app.

    It takes vector retrieval results, extracts a KG from the retrieved text,
    and returns HTML + graph metadata + triples.
    """
    retrieved_text = vector_results_to_text(
        vector_results,
        max_chars=max_chars,
    )

    if not retrieved_text.strip():
        return {
            "html": "",
            "graph_data": {
                "nodes": [],
                "edges": [],
                "metadata": {
                    "node_count": 0,
                    "edge_count": 0,
                },
            },
            "triples": [],
            "node_count": 0,
            "edge_count": 0,
            "shown_edge_count": 0,
            "empty_reason": "No retrieved text available.",
        }

    graph_data = extractor(retrieved_text)

    G = kg_to_networkx(
        graph_data,
        min_confidence=min_confidence,
    )

    triples = kg_graph_to_triples(
        graph_data,
        min_confidence=min_confidence,
    )

    html_output = make_pyvis_html(G, height=height) if G.number_of_edges() > 0 else ""

    return {
        "html": html_output,
        "graph_data": graph_data,
        "triples": triples,
        "node_count": len(graph_data.get("nodes", [])),
        "edge_count": len(graph_data.get("edges", [])),
        "shown_edge_count": G.number_of_edges(),
        "empty_reason": "" if G.number_of_edges() > 0 else "No extracted edges above confidence threshold.",
    }


def build_graph_retrieval_visualization(
    graph_results: list[dict[str, Any]],
    height: str = "460px",
) -> dict[str, Any]:
    """
    High-level helper for the right side of the app.

    It visualizes the graph retrieval output directly.
    """
    G = graph_retrieval_results_to_networkx(graph_results)

    html_output = make_pyvis_html(G, height=height) if G.number_of_edges() > 0 else ""

    return {
        "html": html_output,
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "empty_reason": "" if G.number_of_edges() > 0 else "No graph relationships available to visualize.",
    }