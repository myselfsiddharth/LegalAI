"""§6.4 annotation tool: correct the pipeline's output rather than author from scratch.

Run with:  .venv/bin/streamlit run src/annotate/app.py

Design decisions that matter for the data this produces:

**Pipeline output is shown, never pre-selected.** Every judgement starts on "(not reviewed)". If the
pipeline's answer were pre-filled as the default, a reviewer clicking through would silently
manufacture agreement, and §6.1's extraction F1 would measure the reviewer's patience rather than
the extractor's accuracy. The pipeline's answer is displayed beside the control so the reviewer can
agree quickly — but agreeing is an action.

**Disagreement is recorded, not overwritten.** Each field keeps the pipeline value and the reviewer
value side by side, so the gold file supports computing agreement per field and per stratum.

**Facts can be added, not only corrected.** §6.1 asks for precision AND recall, and recall needs
facts the extractor missed. The "missed facts" box is where those go; without it only precision
would be measurable.

**Every fact shows its verified quote.** The reviewer judges the extraction against the span it
claims, not against their memory of the judgment.

Writes to `Data/gold/gold_annotations.jsonl`, one record per save, append-only — so a crashed
session loses nothing and a correction history is preserved.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "Data" / "gold" / "gold_tasks.jsonl"
OUT = ROOT / "Data" / "gold" / "gold_annotations.jsonl"

OUTCOMES = ["(not reviewed)", "WIN", "LOSE", "PARTIAL", "REMAND", "OTHER", "UNCLEAR"]
ASSERTED = ["(not reviewed)", "plaintiff", "defendant", "admitted", "court_narrative", "unknown"]
TRISTATE = ["(not reviewed)", "correct", "wrong"]


@st.cache_data
def load_tasks():
    return [json.loads(l) for l in open(TASKS)]


def load_done() -> dict:
    """Latest annotation per case. Append-only file, so the last record for a case wins."""
    done = {}
    if OUT.exists():
        for line in open(OUT):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[(r["case_id"], r.get("annotator", ""))] = r
    return done


def main() -> None:
    st.set_page_config(page_title="LexGraph gold annotation", layout="wide")
    tasks = load_tasks()
    done = load_done()

    st.sidebar.title("Annotation")
    annotator = st.sidebar.text_input("Your name or initials", value="",
                                      help="Recorded with each judgement. Required for §6.4's "
                                           "inter-annotator agreement on the overlap block.")
    if not annotator:
        st.warning("Enter your name in the sidebar to begin. It is stored with each judgement so "
                   "agreement between annotators can be computed.")
        st.stop()

    strata = sorted({t["stratum"] for t in tasks})
    pick_strata = st.sidebar.multiselect("Strata", strata, default=strata)
    only_todo = st.sidebar.checkbox("Hide cases I have already done", value=True)

    pool = [t for t in tasks if t["stratum"] in pick_strata]
    if only_todo:
        pool = [t for t in pool if (t["case_id"], annotator) not in done]
    mine = sum(1 for (c, a) in done if a == annotator)
    st.sidebar.metric("Done by you", f"{mine} / {len(tasks)}")
    st.sidebar.caption(f"{len(pool)} in the current filter")

    if not pool:
        st.success("Nothing left in this filter.")
        st.stop()

    i = st.sidebar.number_input("Case", 0, len(pool) - 1, 0)
    t = pool[i]

    st.title(f"{t['title']}  ({t['year']})")
    c1, c2, c3 = st.columns(3)
    c1.caption(f"stratum **{t['stratum']}**")
    c2.caption(f"scope: {t.get('task_scope', 'full')}")
    c3.markdown(f"[Indian Kanoon]({t['url']})")
    if t.get("second_annotator"):
        st.info("This case is in the **agreement block** — it is meant to be annotated by two "
                "people independently. Do not look at anyone else's answers.")

    rec: dict = {"case_id": t["case_id"], "annotator": annotator,
                 "stratum": t["stratum"],
                 "annotated_at": datetime.now(timezone.utc).isoformat()}

    # ---------------------------------------------------------------- outcome
    st.header("1. Outcome")
    o = t["outcome"]
    st.markdown(f"Pipeline says **{o['pipeline']}** (from `{o['source']}`"
                + (f"; rules said `{o['rules_said']}`, LLM said `{o['llm_said']}`"
                   if o["source"] == "conflict" else "") + ")")
    if o.get("evidence"):
        st.code(o["evidence"], language=None)
    if t.get("prior_court_text"):
        with st.expander("What the court below did (kept out of the predictor's input)"):
            st.write(t["prior_court_text"][:1500])
    rec["outcome"] = {
        "pipeline": o["pipeline"],
        "reviewer": st.radio("From the perspective of the party who brought the proceeding:",
                             OUTCOMES, horizontal=True, key=f"o{t['case_id']}"),
        "note": st.text_input("Note (optional)", key=f"on{t['case_id']}"),
    }

    # ---------------------------------------------------------------- family
    st.header("2. Claim family")
    f = t["claim_family"]
    st.markdown(f"Pipeline primary: **{f['pipeline_primary']}** · all: `{f['pipeline_all']}`")
    rec["claim_family"] = {
        "pipeline": f["pipeline_primary"],
        "reviewer_correct": st.radio("Is the primary family right?", TRISTATE, horizontal=True,
                                     key=f"f{t['case_id']}"),
        "reviewer_family": st.text_input("If wrong, what should it be?",
                                         key=f"ff{t['case_id']}"),
    }

    # ---------------------------------------------------------------- claims
    if t["claims"]:
        st.header("3. Claims and defences")
        st.caption("Mark any that are not actually claims/defences in this case.")
        cl = []
        for n, x in enumerate(t["claims"]):
            with st.container(border=True):
                st.markdown(f"**{x['kind']}** · raised by *{x['raised_by']}* · "
                            f"pipeline family `{x['pipeline_family']}`")
                st.write(x["text"])
                cl.append({"text": x["text"], "kind": x["kind"],
                           "pipeline_family": x["pipeline_family"],
                           "reviewer_correct": st.radio("Correct?", TRISTATE, horizontal=True,
                                                        key=f"c{t['case_id']}{n}"),
                           "reviewer_family": st.text_input("Corrected family",
                                                            key=f"cf{t['case_id']}{n}")})
        rec["claims"] = cl

    # ---------------------------------------------------------------- facts
    if t["facts"]:
        st.header("4. Extracted facts")
        st.caption("For each: is it a fact about this dispute, is the asserting party right, and is "
                   "the canonical label right? The quote is what the extractor relied on.")
        fl = []
        for n, x in enumerate(t["facts"]):
            with st.container(border=True):
                st.markdown(f"**{x['text']}**")
                st.caption(f"quote: “{x['quote'][:300]}”  ·  span {x['source_span']}")
                st.caption(f"pipeline: asserted_by `{x['pipeline_asserted_by']}` · "
                           f"{x['pipeline_disputed_status']} · label `{x['pipeline_label']}`"
                           + ("  (**outside the vocabulary**)" if x["pipeline_label_is_new"] else ""))
                a, b, c = st.columns([1, 1, 2])
                with a:
                    isf = st.radio("A fact?", TRISTATE, horizontal=False,
                                   key=f"x{t['case_id']}{n}")
                with b:
                    ab = st.selectbox("Asserted by", ASSERTED, key=f"xa{t['case_id']}{n}")
                with c:
                    lb = st.text_input("Corrected label (blank = pipeline label is right)",
                                       key=f"xl{t['case_id']}{n}")
                fl.append({"fact_id": x["fact_id"],
                           "pipeline_label": x["pipeline_label"],
                           "pipeline_asserted_by": x["pipeline_asserted_by"],
                           "reviewer_is_a_fact": isf,
                           "reviewer_asserted_by": ab,
                           "reviewer_label": lb})
        rec["facts"] = fl

        st.subheader("Facts the extractor MISSED")
        st.caption("§6.1 measures recall as well as precision, so a missed fact has to be "
                   "recordable. One per line.")
        rec["missed_facts"] = [s.strip() for s in
                               st.text_area("Missed facts", key=f"m{t['case_id']}",
                                            height=120).split("\n") if s.strip()]

    # ---------------------------------------------------------------- statutes
    s = t["statutes"]
    if s["pipeline_all"]:
        st.header("5. Statutes")
        st.markdown(f"Found anywhere: `{s['pipeline_all']}`")
        st.markdown(f"Introduced by the court (absent from the facts/pleadings): "
                    f"`{s['pipeline_novel']}`")
        rec["statutes"] = {
            "pipeline_all": s["pipeline_all"],
            "pipeline_novel": s["pipeline_novel"],
            "reviewer_correct": st.radio("Are these the right provisions?", TRISTATE,
                                         horizontal=True, key=f"s{t['case_id']}"),
            "reviewer_missing": [x.strip() for x in
                                 st.text_input("Provisions missing (comma separated)",
                                               key=f"sm{t['case_id']}").split(",") if x.strip()],
        }

    # ---------------------------------------------------------------- text
    with st.expander("The masked text the predictor sees"):
        st.text(t["masked_text"][:30000])

    st.divider()
    if st.button("Save this case", type="primary"):
        unreviewed = rec["outcome"]["reviewer"] == "(not reviewed)"
        if unreviewed:
            st.error("The outcome is still '(not reviewed)'. That judgement is the one every "
                     "stratum needs, so it cannot be left blank.")
        else:
            with OUT.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            st.success(f"Saved {t['case_id']}. Advance the Case number in the sidebar.")
            load_tasks.clear()


if __name__ == "__main__":
    main()
