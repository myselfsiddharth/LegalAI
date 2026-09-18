"""Parse a saved `gdown.download_folder(..., skip_download=True)` log into a clean
filename -> Google Drive file ID lookup, so individual PDFs can be fetched by ID later
without re-scraping the whole Drive folder (26,689 files) every time.

Usage: python3 scripts/build_drive_index.py path/to/raw_listing.txt
Output: Data/processed/drive_file_index.json  { filename: drive_file_id }
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "Data" / "processed" / "drive_file_index.json"

LINE_PATTERN = re.compile(r"^Processing file (\S+) (.+)$")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 scripts/build_drive_index.py <raw_listing.txt>")

    raw_path = Path(sys.argv[1])
    index = {}
    dupes = 0
    with raw_path.open() as f:
        for line in f:
            m = LINE_PATTERN.match(line.strip())
            if not m:
                continue
            file_id, filename = m.group(1), m.group(2)
            if filename in index and index[filename] != file_id:
                dupes += 1
            index[filename] = file_id

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(index, indent=0))
    print(f"Indexed {len(index)} unique filenames ({dupes} filename collisions with differing IDs)")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
