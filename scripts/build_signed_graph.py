"""Build the signed legal citation graph and compute a Proof-Support Score (Phase II).

Three edge sources get merged here:
  - citations_classified.jsonl: reliable doc-ID-to-doc-ID case citation edges (from the
    provided metadata), but with NO sign/polarity. This is the graph's skeleton.
  - headnote_signed_citations.jsonl: sign labels for pre-2000 cases (from HEADNOTE text),
    but the cited side is a free-text case name/citation string, not a doc ID -- headnote
    extraction never sees the cited judgment's own PDF, only the citing one.
  - modern_signed_citations.jsonl: sign labels for 2000-2025 cases (from the LLM
    classifier), which DOES carry both citing_doc_id and cited_doc_id directly (it was
    matched against the same metadata this script starts from), so no fuzzy matching
    is needed for this source -- a direct (citing_doc_id, cited_doc_id) key join.

For the headnote source, we match each edge against the OTHER edges the *same citing
document* already has in the metadata (same citing_doc_id), picking the closest
case-name match -- a much easier problem than searching the whole corpus, since a
citing case typically has ~19 outbound edges, not thousands. Edges that fail to match,
or aren't covered by either source, stay in the graph as "unsigned" rather than being
dropped -- they still count structurally.

Usage: python3 scripts/build_signed_graph.py
Outputs:
  Data/processed/signed_graph_edges.csv   -- every edge with its sign + source
  Data/processed/proof_support_scores.csv -- per-case Proof-Support Score
"""

import csv
import json
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
CITATIONS_PATH = ROOT / "Data" / "processed" / "citations_classified.jsonl"
HEADNOTE_PATH = ROOT / "Data" / "processed" / "headnote_signed_citations.jsonl"
MODERN_PATH = ROOT / "Data" / "processed" / "modern_signed_citations.jsonl"
SAMPLE_PATH = ROOT / "Data" / "processed" / "sample_cases.csv"
EDGES_OUT = ROOT / "Data" / "processed" / "signed_graph_edges.csv"
SCORES_OUT = ROOT / "Data" / "processed" / "proof_support_scores.csv"

MATCH_THRESHOLD = 0.6  # SequenceMatcher ratio below this is treated as "no match"


def normalize_name(s: str) -> str:
    return " ".join(s.lower().replace(".", " ").split())


def load_sample_doc_ids() -> set[str]:
    with SAMPLE_PATH.open() as f:
        return {row["doc_id"] for row in csv.DictReader(f)}


def load_case_edges_by_citing_doc(sample_doc_ids: set[str]) -> dict:
    """citing_doc_id -> list of metadata edges, restricted to our 500-case sample."""
    by_doc = defaultdict(list)
    with CITATIONS_PATH.open() as f:
        for line in f:
            e = json.loads(line)
            if e["edge_type"] == "case" and e["citing_doc_id"] in sample_doc_ids:
                by_doc[e["citing_doc_id"]].append(e)
    return by_doc


def load_headnote_edges() -> list[dict]:
    edges = []
    with HEADNOTE_PATH.open() as f:
        for line in f:
            e = json.loads(line)
            if e["citing_doc_id"]:
                edges.append(e)
    return edges


def load_modern_signs() -> dict:
    """(citing_doc_id, cited_doc_id) -> {polarity, verb} -- direct join, no fuzzy matching
    needed since both doc IDs were already resolved when this source was produced."""
    sign_by_edge_key = {}
    if not MODERN_PATH.exists():
        return sign_by_edge_key
    with MODERN_PATH.open() as f:
        for line in f:
            e = json.loads(line)
            if e["citing_doc_id"] and e["cited_doc_id"]:
                sign_by_edge_key[(e["citing_doc_id"], e["cited_doc_id"])] = {
                    "polarity": e["polarity"],
                    "verb": e["label"],
                }
    return sign_by_edge_key


def match_headnote_to_metadata(headnote_edges, case_edges_by_doc):
    """Attach a sign to the metadata edge whose cited_text best matches each headnote
    edge's cited_case, restricted to candidates from the same citing_doc_id."""
    sign_by_edge_key = {}  # (citing_doc_id, cited_doc_id) -> {polarity, verb, match_score}
    matched, unmatched = 0, 0

    for hn in headnote_edges:
        candidates = case_edges_by_doc.get(hn["citing_doc_id"], [])
        if not candidates:
            unmatched += 1
            continue
        target = normalize_name(hn["cited_case"])
        best, best_score = None, 0.0
        for cand in candidates:
            score = SequenceMatcher(None, target, normalize_name(cand["cited_text"])).ratio()
            if score > best_score:
                best, best_score = cand, score
        if best and best_score >= MATCH_THRESHOLD:
            key = (hn["citing_doc_id"], best["cited_doc_id"])
            sign_by_edge_key[key] = {
                "polarity": hn["polarity"],
                "verb": hn["verb"],
                "match_score": round(best_score, 2),
            }
            matched += 1
        else:
            unmatched += 1

    print(f"Headnote-to-metadata matching: {matched} matched, {unmatched} unmatched "
          f"(threshold={MATCH_THRESHOLD})")
    return sign_by_edge_key


def signed_pagerank(G: nx.DiGraph, damping: float = 0.85, iterations: int = 100, tol: float = 1e-8):
    """PageRank variant where each edge contributes score * sign(edge) from source to
    target, instead of always-positive contribution. A negative resulting score means
    the node is net *contested* by what cites it, not merely under-cited."""
    nodes = list(G.nodes())
    n = len(nodes)
    score = {node: 1.0 / n for node in nodes}
    sign_val = {"positive": 1.0, "negative": -1.0, "neutral": 0.3, "unsigned": 0.5}

    for _ in range(iterations):
        new_score = {node: (1 - damping) / n for node in nodes}
        for u, v, data in G.edges(data=True):
            out_deg = G.out_degree(u)
            if out_deg == 0:
                continue
            contribution = damping * sign_val[data["polarity"]] * score[u] / out_deg
            new_score[v] += contribution
        delta = sum(abs(new_score[node] - score[node]) for node in nodes)
        score = new_score
        if delta < tol:
            break
    return score


def main():
    sample_doc_ids = load_sample_doc_ids()
    case_edges_by_doc = load_case_edges_by_citing_doc(sample_doc_ids)
    headnote_edges = load_headnote_edges()
    sign_by_edge_key = match_headnote_to_metadata(headnote_edges, case_edges_by_doc)

    modern_signs = load_modern_signs()
    print(f"Modern-era signed edges available: {len(modern_signs)}")
    overlap = set(sign_by_edge_key) & set(modern_signs)
    if overlap:
        print(f"  ({len(overlap)} edges have both a headnote and modern label -- "
              f"modern (exact doc-ID match) takes precedence)")
    sign_by_edge_key.update(modern_signs)  # modern signs are an exact join; prefer them

    G = nx.DiGraph()
    edge_rows = []
    for citing_doc_id, edges in case_edges_by_doc.items():
        for e in edges:
            key = (citing_doc_id, e["cited_doc_id"])
            sign_info = sign_by_edge_key.get(key)
            polarity = sign_info["polarity"] if sign_info else "unsigned"
            verb = sign_info["verb"] if sign_info else ""
            G.add_edge(citing_doc_id, e["cited_doc_id"], polarity=polarity)
            edge_rows.append(
                {
                    "citing_doc_id": citing_doc_id,
                    "citing_case": e["citing_case"],
                    "cited_doc_id": e["cited_doc_id"],
                    "cited_text": e["cited_text"],
                    "polarity": polarity,
                    "verb": verb,
                }
            )

    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    from collections import Counter
    polarity_counts = Counter(row["polarity"] for row in edge_rows)
    for p, c in polarity_counts.items():
        print(f"  {p:10s}: {c}")

    scores = signed_pagerank(G)

    EDGES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with EDGES_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["citing_doc_id", "citing_case", "cited_doc_id",
                                                "cited_text", "polarity", "verb"])
        writer.writeheader()
        writer.writerows(edge_rows)

    citing_titles = {row["citing_doc_id"]: row["citing_case"] for row in edge_rows}
    with SCORES_OUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "case_title", "proof_support_score", "in_degree", "out_degree"])
        for node, score in sorted(scores.items(), key=lambda kv: -kv[1]):
            writer.writerow([
                node,
                citing_titles.get(node, ""),
                round(score, 6),
                G.in_degree(node),
                G.out_degree(node),
            ])

    print(f"\nWrote {EDGES_OUT}")
    print(f"Wrote {SCORES_OUT}")
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:5]
    print("\nTop 5 Proof-Support Score (most heavily and positively relied-upon authorities):")
    for node, score in top:
        title = citing_titles.get(node, "(cited-only, not in our citing sample)")
        print(f"  {score:.5f}  {node}  {title[:70]}")


if __name__ == "__main__":
    main()
