"""S4 input side: the 66-element catalogue, and which claims a case plausibly raises.

The catalogue is authored, not mined -- `../ontology/ontology_v1.yaml` carries 15 claims with 66
elements, each with an `element_id`, `name` and `definition`, converted from the domain expert's
document. v1's legacy attempt to MINE elements from judgment text produced 29 with support >= 2 and
had a known weakness (semantic duplicates surviving a string-normalised clustering), so the authored
catalogue is the better starting point: it is complete by construction and its ids are stable.

Evaluating all 66 elements for every case would be both wasteful and misleading -- a case about
adverse possession has nothing to say about easement elements, and asking produces 60 confident
`UNCLEAR`s that dilute every metric. So each case is matched to its most plausible claims first.

Claim selection is by embedding similarity, reusing the fact embeddings already cached by v1:
each claim gets a descriptor (name + issue + element names), and a claim scores against a case by
the MAXIMUM cosine over that case's facts, not the mean. Maximum, because a claim is raised if ANY
fact implicates it; a mean would bury a single decisive fact under sixteen procedural ones, which is
the same error that made mean-pooled embeddings lose to a soft histogram in S2.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import yaml

from src.llm import client
from v2 import paths


@dataclass(frozen=True)
class Element:
    element_id: str
    name: str
    definition: str
    claim_id: str
    claim_name: str


@dataclass(frozen=True)
class Claim:
    claim_id: str
    name: str
    issue: str
    elements: tuple[Element, ...]

    @property
    def descriptor(self) -> str:
        els = "; ".join(f"{e.name}: {e.definition}" for e in self.elements)
        return f"{self.name}. {self.issue} Elements: {els}"


def load_claims() -> list[Claim]:
    d = yaml.safe_load(open(paths.ONTOLOGY))
    out = []
    for c in d["claims"]:
        els = tuple(
            Element(element_id=e["element_id"], name=e["name"],
                    definition=(e.get("definition") or "").strip(),
                    claim_id=c["claim_id"], claim_name=c["name"])
            for e in (c.get("elements") or []))
        if els:
            out.append(Claim(claim_id=c["claim_id"], name=c["name"],
                             issue=(c.get("issue") or "").strip(), elements=els))
    return out


def all_elements(claims: list[Claim]) -> list[Element]:
    return [e for c in claims for e in c.elements]


def select_claims(claims: list[Claim], fact_vecs: np.ndarray, top_n: int = 2) -> list[int]:
    """Indices of the top_n claims for one case, by MAX cosine over its facts."""
    D = claim_embeddings(claims)
    V = fact_vecs / (np.linalg.norm(fact_vecs, axis=1, keepdims=True) + 1e-9)
    sim = (V @ D.T).max(0)                      # max over facts, per claim
    return list(np.argsort(-sim)[:top_n])


_CLAIM_EMB: np.ndarray | None = None


def claim_embeddings(claims: list[Claim]) -> np.ndarray:
    """Embed the 15 claim descriptors once, cached on disk and in process."""
    global _CLAIM_EMB
    if _CLAIM_EMB is not None:
        return _CLAIM_EMB
    cache = paths.INTERIM / "claim_descriptor_emb.npy"
    if cache.exists():
        D = np.load(cache)
    else:
        D = np.asarray(client.embed([c.descriptor for c in claims], verbose=False),
                       dtype=np.float32)
        np.save(cache, D)
    _CLAIM_EMB = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    return _CLAIM_EMB


if __name__ == "__main__":
    cs = load_claims()
    es = all_elements(cs)
    print(f"{len(cs)} claims, {len(es)} elements")
    for c in cs:
        print(f"  {c.claim_id}  {len(c.elements):2d} elements  {c.name}")
    n_def = sum(1 for e in es if e.definition)
    print(f"\nelements with a definition: {n_def}/{len(es)}")
    burden = sum(1 for c in yaml.safe_load(open(paths.ONTOLOGY))["claims"]
                 for e in (c.get("elements") or []) if e.get("burden_on"))
    print(f"elements with burden_on set: {burden}/{len(es)}  (blocker B2 -- S4 runs unweighted)")
