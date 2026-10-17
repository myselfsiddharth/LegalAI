"""Does each case a closed-book brief cites exist at all? Checked against all 26,688 SC judgments.

The brief's own resolver matches only the land-dispute pool and only cases decided before the
query, so its 'unresolved' mixes fabrications with (a) real SC cases outside the pool and (b) real
cases decided after the held-out query -- anachronistic for the evaluation, fine for a lawyer today.
This check uses every judgment title (from the PDF filenames), any date. Both sides of the cited name
must match (brief.resolves). Duplicate copies of one judgment (same title stem, same year) count
as one match.

  exists                        matches a real SC judgment
  exists_after_query            ... but decided after the held-out case
  cited_as_non_SC               the citation is to a High Court / Privy Council reporter
  not_found                     SC-styled and matches nothing: the fabrication signal, subject to
                                matcher recall (audited by hand before reporting)
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

from src import paths
from product.brief import party_tokens, resolves

P = paths.ROOT / "product"
NON_SC = re.compile(r"\b(?:Mad|Bom|Cal|All|Pat|Ker|Kar|Guj|Raj|P&H|Del|Ori|Lah|Nag|MP|AP|PC)\b\.?\s*\d|"
                    r"\bP\.?C\.?\b|\bPrivy Council\b|\bH\.?C\.?\b")


def main() -> None:
    titles = []
    for _, _, fs in os.walk("Data/raw_pdfs"):
        for f in fs:
            m = re.match(r"(.+)_on_(\d{1,2})_(\w+)_(\d{4})(?:_\d+)?\.pdf$", f, re.I)
            if m:
                titles.append((m.group(1).replace("_", " ") + f" on {m.group(2)} {m.group(3)} {m.group(4)}",
                               int(m.group(4))))
    toks = [(party_tokens(t), t, y) for t, y in titles]
    reg = {json.loads(l)["doc_id"]: json.loads(l) for l in open(paths.CASE_REGISTRY)}
    st, nf = Counter(), []
    for r in map(json.loads, open(P / "data" / "briefs.jsonl")):
        if r["arm"] != "closed" or r["status"] != "ok":
            continue
        yq = reg[r["case_id"]]["year"]
        for iss in r["brief"]["issues"]:
            for p in iss["precedents"]:
                pt = party_tokens(p)
                hits = {(y, frozenset(party_tokens(t))) for tk, t, y in toks if pt & tk and resolves(p, t)}
                if hits:
                    k = "exists" if any(y <= yq for y, _ in hits) else "exists_after_query"
                elif NON_SC.search(p):
                    k = "cited_as_non_SC"
                else:
                    k = "not_found"
                    nf.append(p)
                st[k] += 1
    n = sum(st.values())
    out = {"titles": len(titles), "closed_book_citations": n, "counts": dict(st),
           "shares": {k: round(v / n, 4) for k, v in st.items()}, "not_found_examples": nf[:40]}
    (P / "experiments" / "brief_closed_fullcorpus.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("closed_book_citations", "counts", "shares")}, indent=1))
    print(*nf[:25], sep="\n")


if __name__ == "__main__":
    main()
