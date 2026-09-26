"""Lexical (BM25) retrieval over the cached judgment corpus, plus citation scrubbing.

This is the haystack side of the inverted pipeline: given the facts of a dispute, find the
prior cases that could serve as authority. BM25 rather than embeddings on purpose -- it
costs no API calls, indexes full text instead of a truncated representation, and gives a
lexical baseline any dense retriever has to beat.

Memory matters: 5,462 judgments average ~60k characters, so postings are built in two
streaming passes (document frequencies first, then the postings themselves into array('i'))
rather than holding tokenized documents in RAM.
"""
import json
import math
import re
from array import array
from collections import Counter

from ode_lib import ROOT, CASE_CONNECTOR, REPORTER

CORPUS_PATH = ROOT / "Data" / "processed" / "corpus_text.jsonl"

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


def load_corpus():
    """Callable returning a fresh generator of (doc_id, year, text) each time."""
    def gen():
        with CORPUS_PATH.open() as f:
            for line in f:
                d = json.loads(line)
                year = int(d["year"]) if str(d.get("year", "")).isdigit() else 0
                yield d["doc_id"], year, d["text"]
    return gen


# ---------------------------------------------------------------- dense retrieval

DENSE_VECS = ROOT / "Data" / "processed" / "dense_index.npy"
DENSE_META = ROOT / "Data" / "processed" / "dense_index_meta.json"

# Qwen3-embedding expects an instruction prefix on queries but not on documents.
QUERY_INSTRUCTION = (
    "Instruct: Given the facts of a property or land dispute, retrieve prior Supreme Court "
    "decisions that could serve as binding authority\nQuery: ")


class DenseIndex:
    """Same search() contract as BM25Index, so both go through one evaluation path.

    Document vectors are built offline by build_dense_index.py and are already
    L2-normalised, so cosine similarity is a single matrix-vector product.
    """

    def __init__(self, client, model: str, years: dict[str, int]):
        import numpy as np
        meta = json.loads(DENSE_META.read_text())
        self.doc_ids: list[str] = meta["doc_ids"]
        self.model = model
        self.client = client
        self.matrix = np.load(DENSE_VECS)
        self.years = [years.get(d, 0) for d in self.doc_ids]
        self._np = np

    def __len__(self):
        return len(self.doc_ids)

    def _embed_query(self, text: str):
        from run_baselines import chunk_text, embed
        chunks = chunk_text(text) or [text]
        vecs = self._np.asarray(
            embed(self.client, self.model, [QUERY_INSTRUCTION + c for c in chunks]),
            dtype="float32")
        v = vecs.mean(axis=0)
        return v / (self._np.linalg.norm(v) or 1.0)

    def search(self, query: str, k: int, before_year: int | None = None,
               exclude: set[str] = frozenset(), **_) -> list[tuple[str, float]]:
        scores = self.matrix @ self._embed_query(query)
        order = self._np.argsort(-scores)
        out = []
        for i in order:
            doc_id = self.doc_ids[i]
            if doc_id in exclude:
                continue
            if before_year is not None and self.years[i] > before_year:
                continue
            out.append((doc_id, float(scores[i])))
            if len(out) >= k:
                break
        return out
