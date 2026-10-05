"""Blind element-satisfaction annotation. One case at a time; the model's answer stays hidden.

Run:  .venv/bin/streamlit run v2/annotate_elements.py

Why blind: the measurement wanted is agreement between an independent reviewer and S4. Showing the
model's verdict first turns that into a confirmation exercise -- anchoring is well documented and
would make a high agreement rate meaningless. The model's answers for a case are revealed only after
that case is submitted, which keeps the task honest while still letting a reviewer calibrate.

Why fact NUMBERS rather than typed quotes: one click instead of a transcription, and the quote is
exact by construction, so the same mechanical check that gates S4 also gates the human. A reviewer
who cannot point at a fact has, by the project's own standard, no evidence.

The file is append-only and written after every case, so stopping halfway leaves a scoreable
partial pass. The last record for a (case, annotator) pair wins.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

TASKS = Path("Data/gold/element_gold_tasks.jsonl")
OUT = Path("Data/gold/element_gold_annotations.jsonl")
VERDICTS = ("SATISFIED", "NOT_SATISFIED", "UNCLEAR")


def load_tasks():
    return [json.loads(l) for l in open(TASKS)]


def load_done() -> dict:
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
    st.set_page_config(page_title="Element annotation", layout="wide")
    if not TASKS.exists():
        st.error(f"{TASKS} missing — run `python -m v2.gold_elements` first.")
        return

    tasks = load_tasks()
    with st.sidebar:
        st.header("Element annotation")
        annotator = st.text_input("Your name or initials", value="",
                                  help="Stored with each record so two reviewers can be compared.")
        if not annotator.strip():
            st.warning("Enter a name to begin.")
            st.stop()
        done = load_done()
        mine = {c for (c, a) in done if a == annotator.strip()}
        todo = [t for t in tasks if t["case_id"] not in mine]
        n_items = sum(len(t["items"]) for t in tasks)
        n_mine = sum(len(done[(c, annotator.strip())]["judgements"]) for c in mine)
        st.metric("cases done", f"{len(mine)}/{len(tasks)}")
        st.metric("judgements done", f"{n_mine}/{n_items}")
        st.progress(len(mine) / max(1, len(tasks)))
        st.caption("Written after every case. Stop any time — a partial pass still scores.")
        st.divider()
        st.caption("**SATISFIED** — a fact affirmatively establishes the element.\n\n"
                   "**NOT_SATISFIED** — a fact affirmatively *contradicts* it. Not mere silence.\n\n"
                   "**UNCLEAR** — the facts do not address it either way. This is the right answer "
                   "when nothing is said, including for negatively-framed elements.")

    if not todo:
        st.success("All cases annotated. Score with "
                   "`.venv/bin/python -m v2.score_gold_elements`")
        return

    t = todo[0]
    st.subheader(f"{t.get('title') or t['case_id']}")
    c1, c2, c3 = st.columns(3)
    c1.caption(f"case `{t['case_id']}` · {t.get('year')}")
    c2.caption(f"stratum **{t.get('stratum')}**")
    if t.get("url"):
        c3.caption(f"[judgment]({t['url']})")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Facts")
        st.caption("These are the only evidence. The court's reasoning and order are excluded.")
        for i, f in enumerate(t["facts"], 1):
            st.markdown(f"**{i}.** {f}")

    with right:
        st.markdown(f"#### {len(t['items'])} element judgements")
        st.caption("The model's answers are hidden until you submit.")
        choices = {}
        for n, it in enumerate(t["items"]):
            with st.container(border=True):
                st.markdown(f"**{it['element_name']}** &nbsp; `{it['element_id']}`")
                if it.get("element_definition"):
                    st.caption(it["element_definition"])
                st.caption(f"from claim: *{it['claim_name']}*")
                v = st.radio("verdict", VERDICTS, index=2, horizontal=True,
                             key=f"v_{t['case_id']}_{n}", label_visibility="collapsed")
                q = 0
                if v != "UNCLEAR":
                    q = st.selectbox(
                        "which fact number supports this?",
                        options=list(range(1, len(t["facts"]) + 1)),
                        key=f"q_{t['case_id']}_{n}",
                        help="Pick the fact that establishes or contradicts the element.")
                choices[n] = (v, q)

        if st.button("Submit case and continue", type="primary", use_container_width=True):
            rec = {
                "case_id": t["case_id"], "annotator": annotator.strip(),
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "stratum": t.get("stratum"),
                "judgements": [{
                    "element_id": it["element_id"],
                    "human_verdict": choices[n][0],
                    "human_fact_index": choices[n][1],
                    "human_quote": (t["facts"][choices[n][1] - 1] if choices[n][1] else ""),
                    "model_verdict": it["model_verdict"],
                    "model_quote": it.get("model_quote", ""),
                } for n, it in enumerate(t["items"])]}
            with OUT.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            st.session_state["reveal"] = rec
            st.rerun()

    if st.session_state.get("reveal"):
        rec = st.session_state.pop("reveal")
        agree = sum(1 for j in rec["judgements"] if j["human_verdict"] == j["model_verdict"])
        st.success(f"Saved `{rec['case_id']}` — you agreed with the model on "
                   f"{agree}/{len(rec['judgements'])}.")
        with st.expander("What the model said (revealed after submission)", expanded=True):
            for j in rec["judgements"]:
                mark = "✓" if j["human_verdict"] == j["model_verdict"] else "✗"
                st.markdown(f"{mark} `{j['element_id']}` — you **{j['human_verdict']}**, "
                            f"model **{j['model_verdict']}**")
                if j["model_quote"]:
                    st.caption(f"model's quote: {j['model_quote'][:200]}")


main()
