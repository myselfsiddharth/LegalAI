"""BM25 retrieval with time-respecting filtering (§9.3).

Lifted from the legacy pre-`src/` pipeline, removed 2026-10-03 (recoverable from git history),
that is not an importable package. Two parts of it are exactly what §9.3 requires and are kept
unchanged:

  `before_year=` on `search()` enforces that a retrieved authority must PREDATE the querying case.
  §9.3 asks for time-respecting retrieval verified by an assertion in code; the filter is here and
  the assertion is in `precedent_retrieve.py`.

  `scrub_citations()` drops every sentence carrying a case citation from the query. A judgment
  names the authorities it relies on in the same sentence that reasons about them, so without this
  the query contains its own answer. It is deliberately blunt -- it costs real factual content,
  which is far cheaper than leaking a label.
"""
import json
import math
import re
from array import array
from collections import Counter

from src import paths

# Inlined from the legacy pipeline (removed; see git history). These two decide what
# counts as a citation-bearing sentence for `scrub_citations`.
CASE_CONNECTOR = re.compile(r"(?:\bv\.?|\bV\.|\b[Vv][Ss]\.?|\bversus)(?=\s|$)")
REPORTER = re.compile(
    r"(?:\b[A-Z]\.\s?){2,}|\b(?:SCC|SCR|AIR|SCALE|JT|ITR|MLJ|SCJ|KB|QB|AC|WLR)\b")

# Standard English stopwords only. Legal boilerplate ("court", "appeal", "learned") is
# deliberately kept: BM25's idf already discounts whatever is common in THIS corpus, and
# hand-removing domain words throws away signal we cannot get back.
STOPWORDS = set("""a about above after again against all am an and any are as at be because
been before being below between both but by can cannot could did do does doing down during
each few for from further had has have having he her here hers herself him himself his how
i if in into is it its itself me more most my myself no nor not of off on once only or
other ought our ours ourselves out over own same she should so some such than that the
their theirs them themselves then there these they this those through to too under until
up very was we were what when where which while who whom why with would you your yours
yourself yourselves""".split())

TOKEN = re.compile(r"[a-z][a-z0-9]{2,}")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


def scrub_citations(text: str) -> str:
    """Drop every sentence carrying a case citation.

    The prediction task is only honest if the query cannot contain its own answer, and a
    judgment names the authorities it relies on in the same sentence that reasons about
    them. Removing the whole sentence is deliberately blunt: it costs some genuine factual
    content, which is far cheaper than leaking a label.
    """
    flat = re.sub(r"\s+", " ", text)
    kept = [s for s in SENTENCE.split(flat)
            if not (CASE_CONNECTOR.search(s) or REPORTER.search(s))]
    return " ".join(kept)


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_ids: list[str] = []
        self.years: list[int] = []
        self.doc_len: list[int] = []
        self.postings: dict[str, tuple[array, array]] = {}
        self.idf: dict[str, float] = {}

    def build(self, records, min_df: int = 2, max_df_ratio: float = 0.35, verbose=True):
        """records: iterable of (doc_id, year, text). Consumed twice, so pass a callable."""
        df = Counter()
        n = 0
        for doc_id, year, text in records():
            df.update(set(tokenize(text)))
            n += 1
            if verbose and n % 1000 == 0:
                print(f"  df pass: {n} docs, {len(df):,} terms", flush=True)
        max_df = int(max_df_ratio * n)
        vocab = {t for t, c in df.items() if min_df <= c <= max_df}
        if verbose:
            print(f"  vocabulary {len(vocab):,} of {len(df):,} terms "
                  f"(df in [{min_df}, {max_df}])", flush=True)

        self.postings = {t: (array("i"), array("i")) for t in vocab}
        for i, (doc_id, year, text) in enumerate(records()):
            counts = Counter(t for t in tokenize(text) if t in vocab)
            self.doc_ids.append(doc_id)
            self.years.append(year)
            self.doc_len.append(sum(counts.values()) or 1)
            for term, tf in counts.items():
                docs, tfs = self.postings[term]
                docs.append(i)
                tfs.append(tf)
            if verbose and (i + 1) % 1000 == 0:
                print(f"  postings pass: {i + 1} docs", flush=True)

        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        for term, (docs, _) in self.postings.items():
            self.idf[term] = math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
        if verbose:
            print(f"  indexed {n} docs, avg length {self.avgdl:,.0f} tokens", flush=True)

    def search(self, query: str, k: int, before_year: int | None = None,
               exclude: set[str] = frozenset(),
               max_query_terms: int = 500) -> list[tuple[str, float]]:
        """Top-k doc_ids. before_year enforces that an authority must predate the case.

        A whole judgment is a ~10k-token query, and scoring every one of those terms means
        walking a posting list per term. Keeping the highest tf*idf terms cuts the work by
        more than an order of magnitude and drops only terms that were contributing least.
        """
        counts = Counter(tokenize(query))
        terms = [(t, qtf) for t, qtf in counts.items() if t in self.postings]
        terms.sort(key=lambda tq: -tq[1] * self.idf[tq[0]])
        scores: dict[int, float] = {}
        for term, qtf in terms[:max_query_terms]:
            idf = self.idf[term]
            docs, tfs = self.postings[term]
            for i, tf in zip(docs, tfs):
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] = scores.get(i, 0.0) + idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        out = []
        for i, s in ranked:
            if self.doc_ids[i] in exclude:
                continue
            if before_year is not None and self.years[i] > before_year:
                continue
            out.append((self.doc_ids[i], s))
            if len(out) >= k:
                break
        return out


