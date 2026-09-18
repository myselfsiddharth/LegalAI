"""Download judgment PDFs for the sampled cases from the shared Google Drive folder.

Looks up each filename in Data/processed/drive_file_index.json (built by
build_drive_index.py) and fetches it by Drive file ID, so we only pull the specific
files we need instead of the full 26,688-document corpus.

Google Drive throttles anonymous bulk access to a shared folder -- after enough rapid
requests it starts refusing every file with "Cannot retrieve the public link ... many
accesses", regardless of which file is asked for. This isn't a per-file quota, it's a
block on the whole session, and hammering it with immediate retries only makes the
cooldown longer. So: a deliberate delay between files, a *bounded* retry per file for
ordinary errors, and for a detected rate limit specifically -- an exponential backoff
wait-and-retry loop (starting at --rate-limit-wait, doubling up to a 20-minute cap) that
keeps checking whether the block has cleared, up to --max-rate-limit-hours of total
waiting, instead of giving up immediately. Already-downloaded files are skipped, so this
is always safe to re-run or leave running.

Usage:
  python3 scripts/fetch_sample_pdfs.py --limit 5              # smoke test
  python3 scripts/fetch_sample_pdfs.py                        # full sample, waits out rate limits
  python3 scripts/fetch_sample_pdfs.py --sleep 3               # gentler pace between files
  python3 scripts/fetch_sample_pdfs.py --rate-limit-wait 120   # start backoff at 2 min instead of 1
"""

import argparse
import csv
import json
import time
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "Data" / "processed" / "drive_file_index.json"
SAMPLE_PATH = ROOT / "Data" / "processed" / "sample_cases.csv"
OUT_DIR = ROOT / "Data" / "raw_pdfs"

RATE_LIMIT_MARKER = "Cannot retrieve the public link"
RATE_LIMIT_BACKOFF_CAP = 1200  # never wait longer than 20 min between retries


def download_one(file_id: str, dest: Path, retries: int) -> tuple[bool, str]:
    for attempt in range(retries):
        try:
            gdown.download(id=file_id, output=str(dest), quiet=True)
            if dest.exists() and dest.stat().st_size > 0:
                return True, ""
            return False, "downloaded but empty"
        except Exception as e:
            msg = str(e)
            if RATE_LIMIT_MARKER in msg:
                return False, "RATE_LIMITED"
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return False, msg[:150]
    return False, "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only fetch the first N rows")
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds between downloads")
    parser.add_argument("--retries", type=int, default=2, help="retries per file (non-rate-limit errors only)")
    parser.add_argument("--rate-limit-wait", type=float, default=60,
                         help="initial backoff (seconds) after hitting a rate limit; doubles each retry, capped at 20 min")
    parser.add_argument("--max-rate-limit-hours", type=float, default=6,
                         help="give up if the block hasn't cleared after this much total waiting")
    args = parser.parse_args()

    index = json.loads(INDEX_PATH.read_text())
    with SAMPLE_PATH.open() as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    ok, skipped, failed = 0, 0, []
    backoff = args.rate_limit_wait
    total_rate_limit_wait = 0.0
    max_wait_seconds = args.max_rate_limit_hours * 3600

    for i, row in enumerate(rows):
        filename, year = row["filename"], row["year"]
        file_id = index.get(filename)
        if not file_id:
            failed.append((filename, "not in drive index"))
            continue

        year_dir = OUT_DIR / year
        year_dir.mkdir(parents=True, exist_ok=True)
        dest = year_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue

        while True:
            success, reason = download_one(file_id, dest, args.retries)
            if success:
                ok += 1
                backoff = args.rate_limit_wait  # block cleared -- reset backoff for next time
                print(f"[{i+1}/{len(rows)}] OK   {filename}", flush=True)
                break

            if reason == "RATE_LIMITED":
                if total_rate_limit_wait >= max_wait_seconds:
                    print(
                        f"\nStill rate-limited after {total_rate_limit_wait/3600:.1f}h of waiting "
                        f"(--max-rate-limit-hours={args.max_rate_limit_hours}). Giving up for now.\n"
                        f"Progress so far: {ok} new, {skipped} already had, "
                        f"{len(rows) - i} not yet attempted. Re-run this script later to resume.",
                        flush=True,
                    )
                    print(f"\nNew downloads: {ok}, already present: {skipped}, failed: {len(failed)}, total: {len(rows)}")
                    return
                print(
                    f"[{i+1}/{len(rows)}] rate-limited, waiting {backoff:.0f}s before retrying "
                    f"(total waited so far: {total_rate_limit_wait/60:.0f} min)...",
                    flush=True,
                )
                time.sleep(backoff)
                total_rate_limit_wait += backoff
                backoff = min(backoff * 2, RATE_LIMIT_BACKOFF_CAP)
                continue

            failed.append((filename, reason))
            print(f"[{i+1}/{len(rows)}] FAIL {filename}: {reason}", flush=True)
            break

        time.sleep(args.sleep)

    print(f"\nNew downloads: {ok}, already present: {skipped}, failed: {len(failed)}, total: {len(rows)}")
    if failed:
        for name, reason in failed[:10]:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
