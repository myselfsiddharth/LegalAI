"""Stage 2 (§6.1): masked text -> atomic Fact propositions, each tied to a verbatim span.

Every fact carries a `source_span` that indexes the real judgment, because §0.5 makes
traceability a property of the pipeline rather than a reporting afterthought: §11's trace has to
walk from a conclusion back to the characters it rests on, and a fact without an offset breaks
that walk.

The span is recovered from the model's own quote rather than asked for as numbers. Models are
poor at counting characters and good at copying text, so we require the quote, verify it occurs
in the chunk, and compute the offsets ourselves. A fact whose quote cannot be located is
DISCARDED, not kept with a null span -- an unlocatable fact is indistinguishable from an
invented one.

Extraction reads `masked_text`, so no fact can be derived from the court's reasoning or its
order (§5.2). That is what makes these facts usable as predictor inputs rather than as targets.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, field

from pydantic import BaseModel, RootModel

from src import grounding, paths
from src.extract.fact_filters import is_legal_statement, orientation_coverage
from src.llm import client

PROMPT_ID = "extract_facts.v2"
OUT_PATH = paths.INTERIM / "facts.jsonl"

CHUNK_CHARS = 5_000
CHUNK_OVERLAP = 300
MAX_CHUNKS = 8            # a cap on spend per case; masked median is ~11k chars
SENT_BREAK = re.compile(r"(?<=[.;:!?])\s+")

ASSERTED_BY = {"plaintiff", "defendant", "court_narrative", "admitted", "unknown"}
DISPUTED = {"admitted", "contested", "unclear", "inferred"}


class FactJSON(BaseModel):
    text: str
    quote: str = ""
    asserted_by: str = "unknown"
    disputed_status: str = "unclear"
    event_date: str | None = None
    entities: list[str] = []
    evidence_refs: list[str] = []


class FactList(RootModel):
    root: list[FactJSON]


@dataclass
class Fact:
    fact_id: str
    case_id: str
    text: str
    quote: str
    source_span: list[int]          # [start, end] into the ORIGINAL judgment text
    span_basis: str                 # 'masked' -- offsets index masked_text, mapped below
    asserted_by: str
    disputed_status: str
    event_date: str | None
    entities: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    chunk_idx: int = 0


def chunk(text: str) -> list[tuple[int, str]]:
    """(offset, chunk_text), split on sentence boundaries with a small overlap so a fact
    straddling a boundary is still quotable from one chunk."""
    if len(text) <= CHUNK_CHARS:
        return [(0, text)]
    out, pos = [], 0
    while pos < len(text) and len(out) < MAX_CHUNKS:
        end = min(len(text), pos + CHUNK_CHARS)
        if end < len(text):
            window = text[pos:end]
            breaks = list(SENT_BREAK.finditer(window))
            if breaks and breaks[-1].end() > CHUNK_CHARS * 0.5:
                end = pos + breaks[-1].end()
        out.append((pos, text[pos:end]))
        if end >= len(text):
            break
        pos = max(pos + 1, end - CHUNK_OVERLAP)
    return out


def case_header(rec: dict, title: str) -> str:
    """Who the parties are. Without this the model cannot attribute a fact to the plaintiff or
    the defendant, and defaults to `court_narrative` -- which §8.1 treats as unusable."""
    t = re.sub(r"\s+on\s+\d{1,2}\s+\w+\s+\d{4}\s*$", "", title or "").strip()
    m = re.split(r"\s+vs?\.?\s+", t, maxsplit=1, flags=re.I)
    if len(m) == 2:
        return (f"PARTIES — the party who brought this proceeding (treat as 'plaintiff'): "
                f"{m[0].strip()}\nThe opposing party (treat as 'defendant'): {m[1].strip()}\n"
                f"Year: {rec.get('year')}\n")
    return f"PARTIES — not clearly stated in the case title.\nYear: {rec.get('year')}\n"


def extract_case(rec: dict, model: str, title: str = "") -> tuple[list[Fact], Counter]:
    """rec: a masked_text.jsonl record. Offsets are into `masked_text`; `kept_spans` maps them
    back to the original judgment when a caller needs original coordinates."""
    masked = rec["masked_text"]
    stats = Counter()
    facts: list[Fact] = []
    seen_quotes: set[str] = set()

    for ci, (base, ch) in enumerate(chunk(masked)):
        res = client.complete(
            [{"role": "system", "content": client.load_prompt(PROMPT_ID)},
             {"role": "user", "content": case_header(rec, title) + "\nEXCERPT:\n" + ch}],
            model=model, json_schema=FactList, prompt_id=PROMPT_ID, max_tokens=3000)
        if not res.ok:
            stats["chunks_dropped"] += 1          # never got an answer -- NOT "no facts"
            continue
        stats["chunks_ok"] += 1
        if res.parsed is None:
            stats["chunks_unparseable"] += 1
            continue

        items = res.parsed.root
        stats["proposed"] += len(items)
        if not items:
            stats["chunks_empty"] += 1
        for f in items:
            if is_legal_statement(f.text):
                stats["discarded_legal_statement"] += 1
                continue
            span = grounding.locate(f.quote, ch)
            if span is None:
                stats["discarded_quote_not_located"] += 1
                continue
            key = grounding.normalize(f.quote)
            if key in seen_quotes:                 # the chunk overlap re-surfaces facts
                stats["deduped"] += 1
                continue
            seen_quotes.add(key)
            ab = f.asserted_by if f.asserted_by in ASSERTED_BY else "unknown"
            ds = f.disputed_status if f.disputed_status in DISPUTED else "unclear"
            if f.asserted_by not in ASSERTED_BY:
                stats["asserted_by_coerced"] += 1
            facts.append(Fact(
                fact_id=f"{rec['doc_id']}_f{len(facts)+1:03d}",
                case_id=rec["doc_id"], text=f.text.strip(), quote=f.quote,
                source_span=[base + span[0], base + span[1]], span_basis="masked",
                asserted_by=ab, disputed_status=ds,
                event_date=(f.event_date or None),
                entities=[e for e in f.entities if e][:8],
                evidence_refs=[e for e in f.evidence_refs if e][:6],
                chunk_idx=ci))
            stats["kept"] += 1
    return facts, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--split", default=None, help="restrict to one split's cases")
    ap.add_argument("--priority", default="eval-first",
                    choices=["eval-first", "doc-id"],
                    help="'eval-first' extracts TEST and DEV cases of every split before train "
                         "cases; 'doc-id' is the old arbitrary order")
    ap.add_argument("--only-missing", action="store_true",
                    help="re-run cases that produced zero facts or had a dropped chunk")
    ap.add_argument("--model", default=client.DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--min-masked-chars", type=int, default=1000)
    args = ap.parse_args()

    from src.data.label_merge import load_final
    labels = load_final()
    titles = {json.loads(l)["doc_id"]: json.loads(l)["title"]
              for l in open(paths.CASE_REGISTRY)}
    # Same eligibility as splits.py, including the property-domain screen. Without it the
    # spreadsheet's keyword filter drags in tax and criminal cases, and the canonicalisation
    # pilot duly produced labels like `Statute.fee_for_appeal` and a fact about a whisky dealer.
    screen = {json.loads(l)["doc_id"]: json.loads(l)
              for l in open(paths.INTERIM / "case_screen.jsonl")}
    eligible = {d for d, r in labels.items()
                if r["outcome"] in ("WIN", "LOSE")
                and screen.get(d, {}).get("is_property", False)}
    if args.split:
        sp = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
        eligible &= set(sp["train"]) | set(sp["dev"]) | set(sp["test"])

    cases = []
    for line in open(paths.INTERIM / "masked_text.jsonl"):
        r = json.loads(line)
        if r["doc_id"] in eligible and r["n_chars_masked"] >= args.min_masked_chars:
            cases.append(r)

    # Order matters, and the obvious order was wrong.
    #
    # Taking the first N by doc_id is arbitrary with respect to the splits, so an 800-case run
    # covered only 143 of forum_heldout's 902 TEST cases -- and test coverage is what every
    # reported number is limited by. `eval-first` extracts the test and dev cases of every split
    # before touching train, so a partial run still yields a fully-covered evaluation set.
    if args.priority == "eval-first":
        rank = {}
        for sp in sorted(paths.SPLITS.glob("*.json")):
            d = json.loads(sp.read_text())
            for c in d.get("test", []):
                rank[c] = min(rank.get(c, 9), 0)
            for c in d.get("dev", []):
                rank[c] = min(rank.get(c, 9), 1)
            for c in d.get("train", []):
                rank[c] = min(rank.get(c, 9), 2)
        cases.sort(key=lambda r: (rank.get(r["doc_id"], 3), r["doc_id"]))
        by_rank = Counter(rank.get(c["doc_id"], 3) for c in cases)
        print(f"  priority=eval-first: {by_rank[0]} test, {by_rank[1]} dev, {by_rank[2]} train, "
              f"{by_rank[3]} unsplit", flush=True)
    else:
        cases.sort(key=lambda r: r["doc_id"])

    done = Counter()
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            for line in f:
                try:
                    done[json.loads(line)["case_id"]] += 1
                except json.JSONDecodeError:
                    pass
    if args.only_missing:
        # A case absent from the output yielded zero facts, which is either a genuinely
        # fact-free excerpt or a call that never returned. Re-running distinguishes them.
        todo = [c for c in cases if not done.get(c["doc_id"])]
        print(f"  --only-missing: {len(todo)} cases have no facts on disk", flush=True)
    else:
        todo = [c for c in cases if c["doc_id"] not in done]
    if args.n:
        todo = todo[:args.n]
    print(f"{len(cases)} eligible cases, {len(done)} already extracted, {len(todo)} to do "
          f"(model={args.model})", flush=True)

    total = Counter()
    per_case = []
    with OUT_PATH.open("a") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (facts, st) in enumerate(
                pool.map(lambda c: extract_case(c, args.model, titles.get(c["doc_id"], "")),
                         todo), 1):
            for f in facts:
                out.write(json.dumps(asdict(f)) + "\n")
            total.update(st)
            per_case.append(len(facts))
            if i % 10 == 0:
                out.flush()
                print(f"  {i}/{len(todo)}  facts so far {total['kept']}  "
                      f"dropped chunks {total['chunks_dropped']}", flush=True)

    report(total, per_case, len(todo))


def report(total: Counter, per_case: list[int], n_cases: int) -> None:
    import statistics
    print(f"\n=== fact extraction: {n_cases} cases ===")
    if per_case:
        print(f"  facts per case: median {statistics.median(per_case):.0f}, "
              f"mean {statistics.fmean(per_case):.1f}, min {min(per_case)}, max {max(per_case)}")
        print(f"  cases yielding zero facts: {sum(1 for x in per_case if x == 0)}")
    prop = total["proposed"] or 1
    print(f"  proposed {total['proposed']}  ->  kept {total['kept']} "
          f"({100*total['kept']/prop:.1f}%)")
    print(f"    discarded, quote not located in the excerpt: "
          f"{total['discarded_quote_not_located']} "
          f"({100*total['discarded_quote_not_located']/prop:.1f}%)")
    print(f"    discarded as a statement of law or about an authority: "
          f"{total['discarded_legal_statement']} "
          f"({100*total['discarded_legal_statement']/prop:.1f}%)")
    print(f"    deduped across chunk overlap: {total['deduped']}")
    print(f"  chunks: ok {total['chunks_ok']}, empty {total['chunks_empty']}, "
          f"unparseable {total['chunks_unparseable']}, DROPPED {total['chunks_dropped']}")
    if total["chunks_dropped"]:
        print("    NOTE dropped chunks are calls that never returned -- they are NOT "
              "evidence that those excerpts contain no facts.")
    print(f"  usage: {client.usage()}")


if __name__ == "__main__":
    main()
