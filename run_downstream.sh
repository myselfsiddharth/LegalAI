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
PY=.venv/bin/python
mkdir -p logs

log() { print -r -- "[$(date +%H:%M:%S)] $*" }

# 0. wait for extraction, if it is still going
while pgrep -f "src.extract.facts" >/dev/null; do sleep 60; done
log "extraction finished"

# 1. recover cases that produced no facts -- absent means either fact-free or never answered
log "recovering missing cases"
$PY -u -m src.extract.facts --n 0 --workers 96 --priority eval-first --only-missing \
    > logs/p_facts_missing.log 2>&1
log "  -> logs/p_facts_missing.log"

# 2. QA gate. If spans do not index their source, stop: nothing downstream would be trustworthy.
log "fact-set QA"
if ! $PY -u -m src.extract.facts_qa > logs/p_facts_qa.log 2>&1; then
  log "QA FAILED -- stopping. See logs/p_facts_qa.log"
  exit 1
fi
log "  QA passed"

# 3. claims and defences for every case (§7.1)
log "claims + defences"
$PY -u -m src.extract.claims --n 0 --workers 64 > logs/p_claims.log 2>&1
log "  -> logs/p_claims.log"

# 4. canonicalise every fact (§6.2) -- the long one
log "canonicalising facts"
$PY -u -m src.extract.canonicalize --n 0 --workers 96 > logs/p_canon.log 2>&1
log "  -> logs/p_canon.log"

# 5. claim families, then the data-driven merge (§7.2-7.3 -> §6.3)
log "claim families + merge"
$PY -u -m src.cluster.claim_families > logs/p_families.log 2>&1
$PY -u -m src.cluster.family_merge   > logs/p_merge.log 2>&1
log "  -> logs/p_families.log logs/p_merge.log"

# 6. vocabulary refinement proposal (§6.2 steps 3-4). Proposal only: §6.2 wants human review
#    before a vocabulary is frozen, so this does not pass --apply.
log "vocabulary refinement proposal"
$PY -u -m src.extract.vocab_refine > logs/p_vocab_refine.log 2>&1
log "  -> logs/p_vocab_refine.log"

# 7. transactions + FP-Growth per family (§8)
log "transactions + patterns"
$PY -u -m src.patterns.transactions > logs/p_transactions.log 2>&1
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
