# LexGraph pipeline. Every target is idempotent and safe to re-run: stages either skip
# work already on disk or regenerate deterministically from upstream data.
PY := .venv/bin/python

.PHONY: help stage0 stage1 probe clean-cache
help:
	@grep -E '^[a-z0-9-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t22

stage0: ## §4 registry, judgment text, ontology v1  (text extraction ~4 min)
	$(PY) -m src.data.adapter
	$(PY) -m src.data.text
	$(PY) -m src.data.ontology_convert

stage1: ## §5 screen, labels, masking, splits
	$(PY) -m src.data.screen
	$(PY) -m src.data.labels
	$(PY) -m src.data.mask
	$(PY) -m src.data.splits

labels-llm: ## §5.1 LLM label pass + rules-vs-LLM agreement (needs VOYAGER_API_KEY; ~25 min)
	$(PY) -m src.data.label_llm --workers 12

merge: ## §5.1 reconcile rules + LLM labels into the frozen label set
	$(PY) -m src.data.label_merge
	$(PY) -m src.data.splits

llm-probe: ## §5.2 LLM-0 outcome-identification probe (the second leakage probe)
	$(PY) -m src.eval.llm_probe --split forum_heldout --n 350

stage2: ## §6 seed vocabulary, fact extraction, canonicalisation (needs API key; long)
	$(PY) -m src.extract.vocab
	$(PY) -m src.extract.facts --n 0 --workers 96 --priority eval-first
	$(PY) -m src.extract.canonicalize --n 0 --workers 8

facts-qa: ## audit the extracted fact set (span integrity, split coverage, per-decade drift)
	$(PY) -m src.extract.facts_qa

bench-extractors: ## score candidate extraction models on identical cases
	$(PY) -m src.extract.benchmark_extractors --n 20 --models llama4-scout-17b llama4-maverick-17b

stage3: ## §7 claims, defences, and claim-family clustering
	$(PY) -m src.extract.claims --n 400 --workers 10
	$(PY) -m src.cluster.claim_families

probe: ## §5.2 leakage probe on both splits -- the Stage 1 acceptance gate
	$(PY) -m src.eval.leakage_probe --split temporal_2004_2013
	$(PY) -m src.eval.leakage_probe --split forum_heldout

stage4: ## §8 transactions + FP-Growth patterns per claim family
	$(PY) -m src.patterns.transactions
	$(PY) -m src.patterns.mine --split forum_heldout

stage6: ## §10 outcome models + §10.2 ablations
	$(PY) -m src.predict.models --split forum_heldout
	$(PY) -m src.predict.models --split temporal_2004_2013

side-inputs: ## §10.1 S and R groups: predicted statutes + retrieved precedents
	$(PY) -m src.predict.side_inputs --split forum_heldout

statutes: ## §9.2 statute prediction (novel-provision targets)
	$(PY) -m src.authorities.statute_predict --split forum_heldout --rebuild-targets

precedents: ## §9.3 precedent retrieval (add --text-dense for the best system)
	$(PY) -m src.authorities.precedent_retrieve --split forum_heldout --n-queries 150 --text-dense

llm-baselines: ## §3.3 all five LLM arms on the same cases and metrics
	$(PY) -m src.predict.llm_baselines --split forum_heldout --n 300

errors: ## §10.3 error analysis with a base-rate control
	$(PY) -m src.eval.error_analysis --split forum_heldout --n-llm 50

trace: ## §11 build traces and run the deletion test
	$(PY) -m src.trace.build --split forum_heldout --model gbm --n 25
	$(PY) -m src.trace.evaluate --split forum_heldout --model gbm --n 60

gold: ## §6.4 build the 150-case gold annotation set + the B2 burden sheet
	$(PY) -m src.data.gold_sample
	$(PY) -m src.data.burden_sheet

annotate: ## §6.4 open the annotation tool (correct pipeline output; nothing pre-selected)
	.venv/bin/streamlit run src/annotate/app.py

burden: ## fill element burden metadata (gated against a degenerate annotation)
	$(PY) -m src.data.ontology_burden --model glm-5-3

test: ## unit tests, incl. leakage and time-respect asserts
	$(PY) -m pytest tests/ -q

clean-cache: ## drop the LLM cache (forces re-spend; usually you do NOT want this)
	rm -f Data/cache/llm_cache.sqlite
