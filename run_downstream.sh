#!/usr/bin/env zsh
# Run every stage downstream of fact extraction, in dependency order, unattended.
#
# Each stage is resumable and idempotent, so this can be re-run after an interruption and will
# continue rather than redo. It waits for any in-flight extraction to finish first.
#
# Ordering is forced by the data: canonical facts need facts; transactions need canonical facts AND
# claim families; patterns need transactions; the outcome models need patterns.
set -u
cd "$(dirname "$0")"

# Abort on any stage failure.
#
# This script originally checked only the QA gate's exit code. When `canonicalize` was killed
# mid-run, the script therefore advanced to `claim_families` on INCOMPLETE canonical facts, and a
# later `pkill` on the script orphaned that child to init, where it kept running for 26 minutes
# against stale state while a fresh run did the same work. Two pipelines writing the same files is
# how state becomes quietly inconsistent, so a failed or killed stage now stops the run.
run_stage() {
  local name="$1"; shift
  local logf="$1"; shift
  log "$name"
  if ! "$@" > "$logf" 2>&1; then
    log "STAGE FAILED: $name (see $logf) -- stopping so nothing downstream runs on partial data"
    exit 1
  fi
  log "  -> $logf"
}
PY=.venv/bin/python
mkdir -p logs

# Hold a power assertion for the life of this script.
#
# Without one, macOS idle-sleeps on battery and every stage simply PAUSES -- the work is not lost
# (each stage is resumable and the HTTP client reconnects on wake), but wall-clock time disappears.
# Measured on the first attempt: three sleeps of 78, 194 and 232 minutes, the last confirmed by
# `pmset -g log` as "Wake from Deep Idle ... lid/HID Activity". A 4-hour elapsed time with 25
# minutes of work done is what that looks like from the outside, and it reads as a hung job.
#
# caffeinate re-execs this script once with the assertion held. CAFFEINATED guards the recursion.
# NOTE: this cannot prevent sleep from CLOSING THE LID on Apple Silicon. For an overnight run,
# leave the lid open or keep the machine on AC power.
if [[ -z "${CAFFEINATED:-}" ]]; then
  export CAFFEINATED=1
  log_pre() { print -r -- "[$(date +%H:%M:%S)] $*" }
  log_pre "re-exec under caffeinate (prevents idle sleep; lid-close still sleeps)"
  exec caffeinate -dimsu "$0" "$@"
fi

log() { print -r -- "[$(date +%H:%M:%S)] $*" }

# 0. wait for extraction, if it is still going
while pgrep -f "src.extract.facts" >/dev/null; do sleep 60; done
log "extraction finished"

# 1. recover cases that produced no facts -- absent means either fact-free or never answered
run_stage "recovering missing cases" logs/p_facts_missing.log \
  $PY -u -m src.extract.facts --n 0 --workers 96 --priority eval-first --only-missing

# 2. QA gate. If spans do not index their source, stop: nothing downstream would be trustworthy.
log "fact-set QA"
if ! $PY -u -m src.extract.facts_qa > logs/p_facts_qa.log 2>&1; then
  log "QA FAILED -- stopping. See logs/p_facts_qa.log"
  exit 1
fi
log "  QA passed"

# 3. claims and defences for every case (§7.1)
run_stage "claims + defences" logs/p_claims.log \
  $PY -u -m src.extract.claims --n 0 --workers 64

# 4. canonicalise every fact (§6.2) -- the long one
#
# One call per fact put this stage at ~85,000 calls and 5.3h against extraction's ~12,500. Facts
# are labelled in batches of 16 sharing one candidate menu, which cuts both call count and input
# tokens. A batched call runs ~80s under this concurrency, so the client timeout is raised for
# this stage only -- 90s would abort work that is progressing, and each abort costs four retries.
log "canonicalising facts (batched)"
VOYAGER_TIMEOUT_S=300 $PY -u -m src.extract.canonicalize --n 0 --workers 96 --batch 8 \
    > logs/p_canon.log 2>&1
log "  -> logs/p_canon.log"

# 5. claim families, then the data-driven merge (§7.2-7.3 -> §6.3)
run_stage "claim families" logs/p_families.log $PY -u -m src.cluster.claim_families
run_stage "family merge"   logs/p_merge.log    $PY -u -m src.cluster.family_merge

# 6. vocabulary refinement proposal (§6.2 steps 3-4). Proposal only: §6.2 wants human review
#    before a vocabulary is frozen, so this does not pass --apply.
run_stage "vocabulary refinement proposal" logs/p_vocab_refine.log \
  $PY -u -m src.extract.vocab_refine

# 7. transactions + FP-Growth per family (§8)
run_stage "transactions" logs/p_transactions.log $PY -u -m src.patterns.transactions
log "patterns"
for S in forum_heldout temporal_2004_2013; do
  $PY -u -m src.patterns.mine --split $S > logs/p_mine_$S.log 2>&1
done
log "  -> logs/p_transactions.log logs/p_mine_*.log"

# 8. outcome models + ablations (§10), and the representation ladder that localises any loss
log "outcome models + representation ladder"
for S in forum_heldout temporal_2004_2013; do
  $PY -u -m src.predict.models --split $S              > logs/p_outcome_$S.log 2>&1
  $PY -u -m src.eval.representation_ladder --split $S  > logs/p_ladder_$S.log 2>&1
  $PY -u -m src.eval.leakage_probe --split $S          > logs/p_probe_$S.log 2>&1
done
log "  -> logs/p_outcome_*.log logs/p_ladder_*.log logs/p_probe_*.log"

log "DONE. Summary of headline numbers:"
for f in logs/p_ladder_*.log logs/p_outcome_*.log; do
  print -r -- "--- $f"
  tail -14 "$f"
done
