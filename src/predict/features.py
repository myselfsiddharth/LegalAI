"""§10.1 feature groups, built so each can be ablated independently (§10.2).

Groups, and what each is meant to isolate:

  `F` facts      canonical-atom indicators + per-category counts. What the facts alone carry.
  `P` patterns   indicators for stable, BH-significant patterns from §8, kept separate for the
                 plaintiff and defendant views. What structure adds over bare facts.
  `E` elements   per-element satisfied/defeated counts and a net score, weighted by who bears the
                 burden. **Reads `llm_draft` burden metadata** unless `require_human_burden=True`,
                 and any result using it must say so.
  `C` context    year, forum, proceeding type, claim family, party type.
  `Q` shortcut   prior-court disposition, ISOLATED IN ITS OWN GROUP so §10.2 can run with and
                 without it. It is not part of `C`, because burying a shortcut inside a
                 legitimate group is how a shortcut becomes invisible.

**The feature vocabulary is fit on TRAIN ONLY.** Choosing which atoms to keep by their document
frequency across the whole dataset leaks test distribution into the feature space: a rare atom that
happens to be frequent in test would be selected *because* of test. This is a quiet leak that no
accuracy number would reveal, so `fit()` and `transform()` are separate and only `fit()` sees
train.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from src import paths
from src.extract.fact_filters import in_defendant_view, in_plaintiff_view

GROUPS = ("F", "P", "E", "C", "Q")


def _load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path)]


@dataclass
class FeatureSpace:
    """Fit on train, then applied unchanged to dev and test."""
    atom_vocab: list[str] = field(default_factory=list)
    category_vocab: list[str] = field(default_factory=list)
    pattern_vocab: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    element_vocab: list[str] = field(default_factory=list)
    context_vocab: dict = field(default_factory=dict)
    names: list[str] = field(default_factory=list)
    group_of: list[str] = field(default_factory=list)

    def group_mask(self, groups) -> np.ndarray:
        keep = set(groups)
        return np.array([g in keep for g in self.group_of], dtype=bool)


class FeatureBuilder:
    def __init__(self, min_atom_cases: int = 10, use_patterns: bool = True,
                 require_human_burden: bool = False):
        self.min_atom_cases = min_atom_cases
        self.use_patterns = use_patterns
        self.require_human_burden = require_human_burden
        self.space = FeatureSpace()
        self._load_sources()

    # ---------------------------------------------------------------- sources
    def _load_sources(self):
        self.canon = defaultdict(list)
        for r in _load_jsonl(paths.INTERIM / "canonical_facts.jsonl"):
            self.canon[r["case_id"]].append(r)

        fam_path = paths.INTERIM / "claim_families.json"
        self.membership = (json.loads(fam_path.read_text()).get("membership", {})
                           if fam_path.exists() else {})
        merge_path = paths.INTERIM / "family_merge.json"
        self.fam_to_super = (json.loads(merge_path.read_text()).get("family_to_merged", {})
                             if merge_path.exists() else {})

        self.claims = defaultdict(list)
        for r in _load_jsonl(paths.INTERIM / "claims.jsonl"):
            self.claims[r["case_id"]].append(r)

        self.screen = {r["doc_id"]: r for r in _load_jsonl(paths.INTERIM / "case_screen.jsonl")}
        from src.data.label_merge import load_final
        self.labels = load_final()

        # element burden metadata, with provenance respected
        import yaml
        ont = yaml.safe_load((paths.ONTOLOGY / "ontology_v1.yaml").read_text())
        self.elements = {}
        self.burden_provenance = Counter()
        for c in ont["claims"]:
            for e in c["elements"]:
                prov = e.get("burden_provenance")
                self.burden_provenance[str(prov)] += 1
                if self.require_human_burden and prov != "human":
                    continue
                if not e.get("burden_on"):
                    continue
                self.elements[e["element_id"]] = {
                    "slug": e["slug"], "claim_id": c["claim_id"], "name": e["name"],
                    "burden_on": e["burden_on"], "provenance": prov}

        self.patterns = {}
        if self.use_patterns:
            for p in sorted(paths.PATTERNS.glob("patterns_*.json")):
                d = json.loads(p.read_text())
                for fam, r in d.get("families", {}).items():
                    keep = [tuple(x["pattern"]) for x in r["rules"]
                            if x.get("stable") and x.get("discriminative_bh")]
                    if keep:
                        self.patterns[fam] = keep

    # ---------------------------------------------------------------- helpers
    def _atoms(self, case_id):
        rows = self.canon.get(case_id, [])
        p = {r["label"] for r in rows if in_plaintiff_view(r)}
        d = {r["label"] for r in rows if in_defendant_view(r)}
        return p, d, {r["label"] for r in rows}, rows

    def _families(self, case_id):
        fams = self.membership.get(case_id, {}).get("families", [])
        sup = sorted({self.fam_to_super.get(f, f) for f in fams})
        primary = self.membership.get(case_id, {}).get("primary")
        return sup, (self.fam_to_super.get(primary, primary) if primary else None)

    # ---------------------------------------------------------------- fit
    def fit(self, train_ids):
        atom_cases = Counter()
        cats = Counter()
        for cid in train_ids:
            _, _, allatoms, rows = self._atoms(cid)
            for a in allatoms:
                atom_cases[a] += 1
            for r in rows:
                cats[r["category"]] += 1
        self.space.atom_vocab = sorted(a for a, n in atom_cases.items()
                                       if n >= self.min_atom_cases)
        self.space.category_vocab = sorted(cats)

        self.space.pattern_vocab = [(fam, pat) for fam, pats in sorted(self.patterns.items())
                                    for pat in pats]
        self.space.element_vocab = sorted(self.elements)

        forums = Counter(self.screen.get(c, {}).get("primary_forum") for c in train_ids)
        procs = Counter(self.labels.get(c, {}).get("proceeding_type") for c in train_ids)
        sups = Counter(s for c in train_ids for s in self._families(c)[0])
        self.space.context_vocab = {
            "forum": sorted(f for f, n in forums.items() if f and n >= 10),
            "proceeding": sorted(p for p in procs if p),
            "family": sorted(s for s, n in sups.items() if n >= 10),
            "prior": ["below_dismissed", "below_allowed", "below_upheld", "below_set_aside",
                      "below_remanded"],
        }
        self._build_names()
        return self

    def _build_names(self):
        names, groups = [], []

        def add(nm, g):
            names.append(nm); groups.append(g)

        for a in self.space.atom_vocab:
            add(f"F:atom@p={a}", "F")
            add(f"F:atom@d={a}", "F")
        for c in self.space.category_vocab:
            add(f"F:ncat={c}", "F")
        add("F:n_atoms", "F"); add("F:n_facts", "F")
        add("F:frac_contested", "F"); add("F:frac_attributed", "F")

        for fam, pat in self.space.pattern_vocab:
            add(f"P:{fam}:{'&'.join(pat)}", "P")

        for eid in self.space.element_vocab:
            e = self.elements[eid]
            add(f"E:{eid}:{e['slug']}:satisfied", "E")
            add(f"E:{eid}:{e['slug']}:defeated", "E")
            add(f"E:{eid}:{e['slug']}:net_burden_weighted", "E")
        if self.space.element_vocab:
            add("E:all_elements_touched", "E")

        add("C:year", "C"); add("C:decade", "C")
        for f in self.space.context_vocab["forum"]:
            add(f"C:forum={f}", "C")
        for p in self.space.context_vocab["proceeding"]:
            add(f"C:proceeding={p}", "C")
        for s in self.space.context_vocab["family"]:
            add(f"C:family={s}", "C")
        add("C:n_claims", "C"); add("C:n_defences", "C"); add("C:n_families", "C")
        add("C:state_is_party", "C")

        for p in self.space.context_vocab["prior"]:
            add(f"Q:prior={p}", "Q")
        add("Q:prior_stated", "Q")

        self.space.names, self.space.group_of = names, groups

    # ---------------------------------------------------------------- transform
    def transform(self, ids) -> np.ndarray:
        S = self.space
        X = np.zeros((len(ids), len(S.names)), dtype=np.float32)
        for i, cid in enumerate(ids):
            p, d, allatoms, rows = self._atoms(cid)
            j = 0
            for a in S.atom_vocab:
                X[i, j] = 1.0 if a in p else 0.0
                X[i, j + 1] = 1.0 if a in d else 0.0
                j += 2
            catc = Counter(r["category"] for r in rows)
            for c in S.category_vocab:
                X[i, j] = catc.get(c, 0); j += 1
            X[i, j] = len(allatoms); j += 1
            X[i, j] = len(rows); j += 1
            X[i, j] = (sum(1 for r in rows if r["disputed_status"] == "contested")
                       / max(1, len(rows))); j += 1
            X[i, j] = (sum(1 for r in rows if r["asserted_by"] in ("plaintiff", "defendant"))
                       / max(1, len(rows))); j += 1

            combined = {f"{a}@p" for a in p} | {f"{a}@d" for a in d}
            for _fam, pat in S.pattern_vocab:
                X[i, j] = 1.0 if set(pat) <= combined else 0.0
                j += 1

            if S.element_vocab:
                # An element is "touched" by an atom whose subcategory matches its slug. Crude,
                # and honestly so: §8.3's reviewed pattern->element edges are the real mapping and
                # do not exist yet, so this stands in and is reported as a stand-in.
                subs = {r["subcategory"].lower() for r in rows}
                touched = 0
                for eid in S.element_vocab:
                    e = self.elements[eid]
                    hit = 1.0 if e["slug"] in subs else 0.0
                    neg = 1.0 if any(r["properties"].get("polarity") == "neg"
                                     and r["subcategory"].lower() == e["slug"]
                                     for r in rows) else 0.0
                    sat = hit * (1.0 - neg)
                    X[i, j] = sat
                    X[i, j + 1] = hit * neg
                    w = 1.0 if e["burden_on"] == "claimant" else -1.0
                    X[i, j + 2] = w * (sat - hit * neg)
                    touched += int(hit > 0)
                    j += 3
                X[i, j] = touched / max(1, len(S.element_vocab)); j += 1

            lab = self.labels.get(cid, {})
            yr = lab.get("year") or 0
            X[i, j] = yr; j += 1
            X[i, j] = yr // 10 * 10; j += 1
            forum = self.screen.get(cid, {}).get("primary_forum")
            for f in S.context_vocab["forum"]:
                X[i, j] = 1.0 if forum == f else 0.0; j += 1
            proc = lab.get("proceeding_type")
            for pr in S.context_vocab["proceeding"]:
                X[i, j] = 1.0 if proc == pr else 0.0; j += 1
            sups, _primary = self._families(cid)
            for s in S.context_vocab["family"]:
                X[i, j] = 1.0 if s in sups else 0.0; j += 1
            cl = self.claims.get(cid, [])
            X[i, j] = sum(1 for c in cl if c["kind"] == "claim"); j += 1
            X[i, j] = sum(1 for c in cl if c["kind"] == "defence"); j += 1
            X[i, j] = len(sups); j += 1
            X[i, j] = 1.0 if any(w in (self.labels.get(cid, {}).get("order_evidence") or "").lower()
                                 or w in " ".join(c["text"].lower() for c in cl)
                                 for w in ("state of", "union of india", "municipal",
                                           "collector", "government")) else 0.0
            j += 1

            prior = lab.get("prior_court_outcome")
            for pv in S.context_vocab["prior"]:
                X[i, j] = 1.0 if prior == pv else 0.0; j += 1
            X[i, j] = 1.0 if prior else 0.0; j += 1
        return X

    # ---------------------------------------------------------------- reporting
    def coverage(self, ids) -> dict:
        have_facts = sum(1 for c in ids if self.canon.get(c))
        have_fam = sum(1 for c in ids if self.membership.get(c))
        have_claims = sum(1 for c in ids if self.claims.get(c))
        return {"n": len(ids),
                "with_canonical_facts": have_facts,
                "with_claim_family": have_fam,
                "with_claims": have_claims,
                "burden_provenance": dict(self.burden_provenance),
                "n_features": len(self.space.names),
                "features_per_group": dict(Counter(self.space.group_of))}
