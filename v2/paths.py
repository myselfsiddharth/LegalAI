"""v2 output locations. Reads v1's inputs, writes only inside v2/."""
from __future__ import annotations

from src import paths as v1

ROOT = v1.ROOT / "v2"
INTERIM = ROOT / "interim"
EXPERIMENTS = ROOT / "experiments"

# v1 inputs, read-only
FACTS = v1.INTERIM / "facts.jsonl"
MASKED = v1.INTERIM / "masked_text.jsonl"
SPLITS = v1.SPLITS
ONTOLOGY = v1.ROOT / "ontology" / "ontology_v1.yaml"

for _d in (INTERIM, EXPERIMENTS):
    _d.mkdir(parents=True, exist_ok=True)
