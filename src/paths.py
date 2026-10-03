"""Canonical paths. One place, so no module guesses at layout.

NOTE on case: the spec writes `data/`, but this repo already has `Data/` and
macOS APFS is case-insensitive -- `data/` and `Data/` resolve to the same inode here.
We standardise on `Data/` so the same code works on a case-SENSITIVE filesystem
(Linux CI, a teammate on Linux) where the two would be different directories.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "Data"
RAW_PDFS = DATA / "raw_pdfs"
DATASET = DATA / "Legal AI Dataset"
LAND_XLSX = DATASET / "filtered_land_disputes.xlsx"
CITATIONS_DIR = DATASET / "citations"

INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
GOLD = DATA / "gold"
PATTERNS = DATA / "patterns"
SPLITS = DATA / "splits"
CACHE = DATA / "cache"

ONTOLOGY = ROOT / "ontology"
PROMPTS = ROOT / "prompts"
EXPERIMENTS = ROOT / "experiments"

# Stage 0 outputs
CASE_REGISTRY = INTERIM / "case_registry.jsonl"
CASE_TEXT = INTERIM / "case_text.jsonl"

for _d in (INTERIM, PROCESSED, GOLD, PATTERNS, SPLITS, CACHE):
    _d.mkdir(parents=True, exist_ok=True)
