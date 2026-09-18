"""Baselines for the Evaluation Plan: the same "IRAC analysis + cited authorities" task done
without the LexGraph pipeline, scored by the same verifier and the same metrics.

  closed_book  Vanilla prompting: the model gets only the case title and year -- the "ask a
               chatbot about case X" setting Dahl et al. (2024) profile. Tests what the model
               produces from parametric memory.
  rag          Standard dense RAG over the judgment's own text: sentence-packed ~1200-char
               chunks, embedded with qwen3-embedding-8b, top-k by cosine for a fixed query,
               then generate from those excerpts only.

Both use the pipeline's generator model (llama4-scout-17b) and one output schema. Every
authority either baseline lists goes through eval_lib.score_authorities -- the exact
verify_authority() the pipeline uses -- and the pipeline's own outputs for the same cases
(irac_batch/, re-scored here) are reported alongside, on the intersection of cases every
system completed, so the three columns are directly comparable.

Usage:
  python3 scripts/run_baselines.py --per-decade 6            # run run_evaluation_batch.py first
  python3 scripts/run_baselines.py --per-decade 6 --resume   # skip cases already done
Outputs:
  Data/processed/baselines/<system>/<doc_id>.json
  Data/processed/baseline_comparison.json
"""

import argparse
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ode_lib import ROOT, get_client
from phase3_verify import load_real_outbound_edges, title_from_filename
from eval_lib import (add_selection_args, select_cases, selection_description, judgment_text,
                      score_authorities, authority_metrics, metrics_by_decade)

OUT_DIR = ROOT / "Data" / "processed" / "baselines"
PIPELINE_DIR = ROOT / "Data" / "processed" / "irac_batch"
COMPARISON_PATH = ROOT / "Data" / "processed" / "baseline_comparison.json"
SYSTEMS = ["closed_book", "rag"]

SYSTEM_PROMPT = "You are a legal research assistant specializing in Indian Supreme Court case law."

TASK_PROMPT = """Analyze the Supreme Court of India judgment "{title}" ({year}) in IRAC form \
(Issue, Rule, Application, Conclusion), and list the prior court decisions this judgment cites \
as authorities.

{context}Return ONLY a JSON object, no commentary:
{{"issue": "...", "rule": "...", "application": "...", "conclusion": "...",
  "authorities": ["<case name as 'A v. B', with reporter citation if known>", ...]}}
List only authorities you are confident this judgment actually cites; omit any you are unsure of."""

# Qwen3-embedding takes an instruction prefix on queries (not on documents).
RETRIEVAL_QUERY = (
    "Instruct: Given a court judgment, retrieve passages that cite prior cases as authority or "
    "state the issue, the rule applied, and the holding\n"
    "Query: precedents cited, relied on, followed, distinguished or overruled; the legal issue; "
    "the rule applied; the Court's holding")

CHUNK_CHARS = 1200
MAX_CHUNK_CHARS = 4000  # a punctuation-free run of text can't grow one chunk unboundedly
EMBED_BATCH = 32


def chunk_text(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", re.sub(r"\s+", " ", text))
    chunks, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) > CHUNK_CHARS:
            chunks.append(current)
            current = ""
        current = f"{current} {s}".strip()[:MAX_CHUNK_CHARS]
    if current:
        chunks.append(current)
    return chunks


def embed(client, model: str, texts: list[str]) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = client.embeddings.create(model=model, input=texts[i:i + EMBED_BATCH])
        vectors.extend(d.embedding for d in resp.data)
    return vectors


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1.0)


def retrieve(client, embed_model: str, text: str, top_k: int) -> list[dict]:
    chunks = chunk_text(text)
    chunk_vecs = embed(client, embed_model, chunks)
    query_vec = embed(client, embed_model, [RETRIEVAL_QUERY])[0]
    ranked = sorted(range(len(chunks)), key=lambda i: -cosine(query_vec, chunk_vecs[i]))[:top_k]
    return [{"chunk_index": i, "text": chunks[i]} for i in sorted(ranked)]  # document order


def parse_response(raw: str) -> tuple[dict, str | None]:
    body = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    start, end = body.find("{"), body.rfind("}")
    try:
        parsed = json.loads(body[start:end + 1]) if start != -1 else None
    except json.JSONDecodeError as e:
        return {}, f"invalid JSON: {e}"
    if not isinstance(parsed, dict):
        return {}, "no JSON object in response"
    return parsed, None


def run_system(system: str, row: dict, text: str, client, args) -> dict:
    context, retrieved = "", None
    if system == "rag":
        retrieved = retrieve(client, args.embed_model, text, args.top_k)
        excerpts = "\n\n".join(f"[{n}] {c['text']}" for n, c in enumerate(retrieved, 1))
        context = f"Answer using only these excerpts from the judgment:\n\n{excerpts}\n\n"
    prompt = TASK_PROMPT.format(title=title_from_filename(row["filename"]), year=row["year"],
                                context=context)
    resp = client.chat.completions.create(
        model=args.model, temperature=0, max_tokens=1500,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
    raw = resp.choices[0].message.content or ""
    parsed, parse_error = parse_response(raw)
    authorities = parsed.get("authorities") or []
    return {
        "doc_id": row["doc_id"], "filename": row["filename"], "year": row["year"], "system": system,
        "irac": {k: parsed.get(k) for k in ("issue", "rule", "application", "conclusion")},
        "authority_results": [{"cited": str(a)} for a in authorities if a],
        "parse_error": parse_error,
        "retrieved_chunks": retrieved,
        "raw_response": raw,
    }


def process_case(row: dict, client, args) -> dict[str, dict]:
    """Run (or load) every baseline for one case; returns {system: scored case record}."""
    text = judgment_text(row)
    if text is None:
        return {}
    real_edges = load_real_outbound_edges(row["doc_id"])
    results = {}
    for system in SYSTEMS:
        out_path = OUT_DIR / system / f"{row['doc_id']}.json"
        if args.resume and out_path.exists():
            case = json.loads(out_path.read_text())
        else:
            try:
                case = run_system(system, row, text, client, args)
            except Exception as e:
                print(f"FAILED {system} {row['filename']}: {e!r} -- re-run with --resume to retry")
                continue
        case["decade"] = row["decade"]
        case["n_real_edges"] = len(real_edges)
        case["authority_results"] = score_authorities(case["authority_results"], row, text, real_edges)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(case, indent=2))
        results[system] = case
    counts = {s: sum(a["verification"]["status"] == "VERIFIED" for a in c["authority_results"])
              for s, c in results.items()}
    print(f"{row['filename'][:55]:55s} verified: {counts}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    add_selection_args(parser)
    parser.add_argument("--model", default="llama4-scout-17b")
    parser.add_argument("--embed-model", default="qwen3-embedding-8b")
    parser.add_argument("--top-k", type=int, default=8, help="chunks retrieved for RAG")
    parser.add_argument("--workers", type=int, default=4, help="cases run concurrently")
    parser.add_argument("--resume", action="store_true", help="reuse existing per-case outputs")
    parser.add_argument("--comparison", type=Path, default=COMPARISON_PATH)
    args = parser.parse_args()

    rows = select_cases(args)
    client = get_client()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        baseline_results = list(pool.map(lambda r: process_case(r, client, args), rows))

    # The pipeline's outputs for the same cases, re-scored with the current verifier.
    by_system = {s: [] for s in SYSTEMS + ["lexgraph"]}
    compared = []
    for row, results in zip(rows, baseline_results):
        pipeline_path = PIPELINE_DIR / f"{row['doc_id']}.json"
        if not pipeline_path.exists() or any(s not in results for s in SYSTEMS):
            continue  # compare only on cases every system completed
        pipeline_case = json.loads(pipeline_path.read_text())
        pipeline_case["decade"] = row["decade"]
        pipeline_case["n_real_edges"] = results[SYSTEMS[0]]["n_real_edges"]
        pipeline_case["authority_results"] = score_authorities(
            pipeline_case["authority_results"], row, judgment_text(row))
        by_system["lexgraph"].append(pipeline_case)
        for s in SYSTEMS:
            by_system[s].append(results[s])
        compared.append(row["doc_id"])

    comparison = {
        "selection": selection_description(args),
        "model": args.model, "embed_model": args.embed_model, "rag_top_k": args.top_k,
        "cases_compared": len(compared),
        "cases_skipped": len(rows) - len(compared),
        "systems": {s: authority_metrics(c) for s, c in by_system.items()},
        "by_decade": {s: metrics_by_decade(c) for s, c in by_system.items()},
        "baseline_parse_errors": {s: sum(bool(c.get("parse_error")) for c in by_system[s]) for s in SYSTEMS},
        "doc_ids": compared,
    }
    args.comparison.write_text(json.dumps(comparison, indent=2))

    print(f"\n=== COMPARISON ({len(compared)} cases every system completed) ===")
    print(f"{'system':12s} {'outputs':>8s} {'case-shaped':>12s} {'verified':>9s} {'rate':>6s} "
          f"{'unver.named':>12s} {'unver.absent':>13s} {'edge recall':>12s}")
    for s, m in comparison["systems"].items():
        print(f"{s:12s} {m['authority_outputs']:8d} {m['case_shaped_citations']:12d} "
              f"{m['status_counts']['VERIFIED']:9d} {str(m['verified_rate']):>6s} "
              f"{m['unverified_named_in_judgment']:12d} {m['unverified_not_in_judgment']:13d} "
              f"{str(m['edge_recall']):>12s}")
    print(f"\nWrote {args.comparison}")


if __name__ == "__main__":
    main()
