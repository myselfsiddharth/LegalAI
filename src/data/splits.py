"""Stage 1 (§5.3): frozen, hashed splits.

Primary split is temporal, because the deployment question is "given a dispute now, predict
its outcome", and a random split lets a model learn from the future.

§5.3 also asks for a court-held-out split. **That is not available in this corpus**: it is
Supreme Court only, so there is no second court to hold out. The substitute is to hold out by
ORIGINATING High Court (recovered for 78.9% of cases by `screen.py`), which tests whether a
model generalises across the regional legal cultures and drafting styles that feed the
Supreme Court. It is a weaker control than the spec intends and is reported as such.

Two hazards handled here:

* **Appeal chains.** The same dispute reaches the Supreme Court more than once, and if one
  visit trains while another tests, the model has seen the answer. We group cases by a
  normalised party signature and keep a group whole within one split.

* **The era/format confound.** HEADNOTE presence is 79.5% before 2000 and 0.0% after, so a
  temporal split also separates two structurally different document types. Masking removes the
  HEADNOTE but not the confound. We therefore report the format composition of every split
  rather than leaving it implicit, and the forum-held-out split -- which is era-balanced --
  exists to tell format shift apart from genuine temporal drift.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from src import paths

STOP = {"state", "of", "the", "and", "ors", "anr", "others", "another", "union", "india",
        "govt", "government", "vs", "v", "etc", "ltd", "co", "m", "s", "shri", "sri", "smt"}


def party_signature(title: str) -> str:
    """A key that two visits of the same dispute share.

    Built from distinctive surnames on both sides, with institutional parties dropped --
    `State of A.P.` or `Union of India` appear in hundreds of unrelated cases, so keeping them
    would merge the corpus into a handful of giant groups.
    """
    t = re.sub(r"\s+on\s+\d{1,2}\s+\w+\s+\d{4}\s*$", "", title, flags=re.I)
    toks = [w for w in re.findall(r"[A-Za-z]+", t.lower())
            if w not in STOP and len(w) > 3]
    return "|".join(sorted(set(toks))[:6])


@dataclass
class SplitSpec:
    name: str
    train: list[str]
    dev: list[str]
    test: list[str]

    def as_dict(self) -> dict:
        return {"name": self.name, "train": self.train, "dev": self.dev, "test": self.test,
                "n_train": len(self.train), "n_dev": len(self.dev), "n_test": len(self.test)}


def _load():
    # The merged rules+LLM label set (§5.1), not the rules pass alone: merging recovers the
    # ~33% of cases no regex could reach and grows the eligible set by roughly a third.
    from src.data.label_merge import load_final
    labels = load_final()
    screen = {}
    for line in open(paths.INTERIM / "case_screen.jsonl"):
        r = json.loads(line)
        screen[r["doc_id"]] = r
    masked = {}
    for line in open(paths.INTERIM / "masked_text.jsonl"):
        r = json.loads(line)
        masked[r["doc_id"]] = {"n_chars_masked": r["n_chars_masked"], "year": r["year"]}
    titles = {}
    for line in open(paths.CASE_REGISTRY):
        r = json.loads(line)
        titles[r["doc_id"]] = r["title"]
    return labels, screen, masked, titles


def eligible(labels, screen, masked, *, property_only=True, min_masked_chars=500):
    """The cases any outcome experiment may use: a decided binary label, enough masked text
    to be a real input, and (by default) actually a property dispute."""
    out = []
    for doc_id, lab in labels.items():
        if lab["initiator_outcome"] not in ("WIN", "LOSE"):
            continue
        m = masked.get(doc_id)
        if not m or m["n_chars_masked"] < min_masked_chars:
            continue
        if property_only and not screen.get(doc_id, {}).get("is_property", False):
            continue
        out.append(doc_id)
    return out


def _part_for_year(y: int, t1: int, t2: int) -> str:
    return "test" if y > t2 else "dev" if y > t1 else "train"


def temporal_split(doc_ids, labels, titles, t1: int, t2: int) -> tuple[SplitSpec, int]:
    """Group-whole AND time-respecting. A dispute group whose visits fall on both sides of a
    boundary cannot satisfy both rules, so it is DROPPED rather than forced into one side:

      * assigning it by its latest year put 1970 cases in a "test > 2013" split;
      * assigning it by its earliest year put 2025 cases in a "train <= 2005" split, which is
        learning from the future.

    Both were caught by tests/test_stage1.py. The number dropped is reported, not hidden.
    """
    groups = defaultdict(list)
    for d in doc_ids:
        groups[party_signature(titles.get(d, d)) or d].append(d)
    train, dev, test = [], [], []
    dropped = 0
    for _, members in groups.items():
        parts = {_part_for_year(labels[d]["year"], t1, t2) for d in members}
        if len(parts) > 1:
            dropped += len(members)
            continue
        {"train": train, "dev": dev, "test": test}[parts.pop()].extend(members)
    return SplitSpec(f"temporal_{t1}_{t2}", sorted(train), sorted(dev), sorted(test)), dropped


def _unused_temporal_split(doc_ids, labels, titles, t1: int, t2: int) -> SplitSpec:
    groups = defaultdict(list)
    for d in doc_ids:
        groups[party_signature(titles.get(d, d)) or d].append(d)
    train, dev, test = [], [], []
    for _, members in groups.items():
        # A group is assigned by its EARLIEST year.
        #
        # Assigning by the latest year keeps the group whole but breaks time-respect: a group
        # containing a 1970 case and a 2015 case lands entirely in test, putting a 1970 case in
        # a "test > 2013" split (caught by tests/test_stage1.py). Assigning by the earliest
        # year satisfies both invariants at once -- the test set becomes disputes whose ENTIRE
        # litigation history falls after t2, and any dispute with an earlier visit goes to
        # train, where that earlier visit is legitimately past information.
        y = min(labels[d]["year"] for d in members)
        (test if y > t2 else dev if y > t1 else train).extend(members)
    return SplitSpec(f"temporal_{t1}_{t2}", sorted(train), sorted(dev), sorted(test))


def forum_split(doc_ids, labels, screen, titles, holdout: set[str]) -> SplitSpec:
    groups = defaultdict(list)
    for d in doc_ids:
        groups[party_signature(titles.get(d, d)) or d].append(d)
    train, test = [], []
    for _, members in groups.items():
        forums = {screen.get(d, {}).get("primary_forum") for d in members}
        (test if forums & holdout else train).extend(members)
    # Carve dev off train BY GROUP, with a fixed seed. Two earlier attempts were wrong:
    # splitting by case let one party group straddle train and dev, and splitting by year made
    # dev the latest slice of train, which -- given the base-rate drift documented in the
    # stage report -- gave dev a 60.8% WIN rate against the test set's 42.6% and so selected
    # models against the wrong prior. A seeded group shuffle keeps groups whole and leaves dev
    # era-comparable to train.
    import random
    train_groups = sorted({party_signature(titles.get(d, d)) or d for d in train})
    rng = random.Random(573)
    rng.shuffle(train_groups)
    dev_groups = set(train_groups[:max(1, int(len(train_groups) * 0.12))])
    tr, dv = [], []
    for d in train:
        (dv if (party_signature(titles.get(d, d)) or d) in dev_groups else tr).append(d)
    return SplitSpec("forum_heldout", sorted(tr), sorted(dv), sorted(test))


def _dist(doc_ids, labels):
    c = Counter(labels[d]["initiator_outcome"] for d in doc_ids)
    n = max(1, len(doc_ids))
    return f"n={len(doc_ids):5d} WIN={100*c['WIN']/n:4.1f}%"


def main() -> None:
    labels, screen, masked, titles = _load()
    ids = eligible(labels, screen, masked)
    years = sorted(labels[d]["year"] for d in ids)
    print(f"eligible cases (binary label, >=500 masked chars, property only): {len(ids)}")
    print(f"  year range {years[0]}-{years[-1]}")

    # choose T1/T2 so that test is ~17% and dev ~10%
    t2 = years[int(len(years) * 0.83)]
    t1 = years[int(len(years) * 0.73)]
    temporal, n_dropped = temporal_split(ids, labels, titles, t1, t2)

    forum_counts = Counter(screen.get(d, {}).get("primary_forum") for d in ids)
    # hold out a set of mid-sized forums totalling ~15%, so the held-out group is neither
    # one dominant court nor a handful of rarities
    target = 0.15 * len(ids)
    holdout, acc = set(), 0
    for f, c in sorted(((f, c) for f, c in forum_counts.items() if f), key=lambda x: -x[1])[2:]:
        if acc >= target:
            break
        holdout.add(f)
        acc += c
    forum = forum_split(ids, labels, screen, titles, holdout)

    out = {}
    for spec in (temporal, forum):
        d = spec.as_dict()
        blob = json.dumps({k: d[k] for k in ("train", "dev", "test")}, sort_keys=True)
        d["hash"] = hashlib.sha256(blob.encode()).hexdigest()[:16]
        if spec is forum:
            d["holdout_forums"] = sorted(holdout)
        d["t1"], d["t2"] = (t1, t2) if spec is temporal else (None, None)
        if spec is temporal:
            d["n_dropped_straddling_groups"] = n_dropped
        out[spec.name] = d
        path = paths.SPLITS / f"{spec.name}.json"
        path.write_text(json.dumps(d, indent=1))

        print(f"\n  {spec.name}  (hash {d['hash']})")
        if spec is temporal:
            print(f"    train <= {t1}   dev {t1+1}-{t2}   test > {t2}")
            print(f"    dropped {n_dropped} cases in groups straddling a boundary "
                  f"({100*n_dropped/len(ids):.1f}% of eligible)")
        else:
            print(f"    held-out forums: {', '.join(sorted(holdout)[:8])}"
                  f"{' ...' if len(holdout) > 8 else ''}")
        for part in ("train", "dev", "test"):
            print(f"    {part:5s} {_dist(d[part], labels)}")
        # the era/format confound, reported rather than hidden
        for part in ("train", "test"):
            pre = sum(1 for x in d[part] if labels[x]["year"] < 2000)
            print(f"    {part:5s} pre-2000 share: {100*pre/max(1,len(d[part])):.1f}%")

    # leakage check on the split itself: no party group may straddle parts
    for name, d in out.items():
        sig = {}
        bad = 0
        for part in ("train", "dev", "test"):
            for x in d[part]:
                s = party_signature(titles.get(x, x)) or x
                if s in sig and sig[s] != part:
                    bad += 1
                sig[s] = part
        print(f"\n  {name}: party groups straddling splits: {bad} (must be 0)")


if __name__ == "__main__":
    main()
