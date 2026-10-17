"""LexGraph demo. Run: .venv/bin/streamlit run demo/app.py

Reads only demo/pack/, built by demo/build_pack.py. Nothing is trained or recomputed at launch, so
the demo cannot hang or disagree with the report. Every number traces to a file under experiments/.

The order of the tabs is deliberate: the finding first, the working components second, the negative
result last. A demo that opens with architecture asks the audience to care before it has shown them
why.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

PACK = Path(__file__).parent / "pack"


@st.cache_data
def load(name):
    return json.loads((PACK / name).read_text())


@st.cache_resource
def load_bm25():
    with (PACK / "bm25.pkl").open("rb") as f:
        return pickle.load(f)


st.set_page_config(page_title="LexGraph", layout="wide", page_icon="⚖")
F = load("findings.json")
CASES = load("cases.json")
ELS = load("elements.json")

st.title("LexGraph")
st.caption("Leakage, discretisation and authority grounding in Indian Supreme Court "
           "land-dispute judgments · CSE 573")

t1, t2, t3, t4 = st.tabs(["① The finding", "② Explore a case",
                          "③ Try your own facts", "④ Why the symbolic pipeline fails"])

# ───────────────────────────────── ① the finding ──────────────────────────────
with t1:
    st.header("Published accuracy on this task is mostly an artifact")
    st.markdown(
        "A judgment is written **after** the decision, by the person who made it. Deleting the "
        "final order does not delete the knowledge of it — Indian judgments announce the "
        "conclusion one sentence *earlier*:")
    st.info("“For the foregoing reasons we find no merit in the appeal. "
            "**The appeal is dismissed with costs.**”")
    st.markdown("Deleting the bold sentence is correct. The one before it gives the answer away "
                "just as plainly. **How far does that leftover reach?**")

    sweep = F["setback"]["land_forum"]["sweep"]
    masked = F["setback"]["land_forum"]["results"]["masked"]
    offs = [s["offset_words"] for s in sweep]
    i = st.select_slider("Push the reading window back from where the order was deleted "
                         "(words)", options=offs, value=0)
    row = next(s for s in sweep if s["offset_words"] == i)

    c = st.columns(4)
    c[0].metric("Model score (AUROC)", f"{row['auroc']:.3f}",
                f"{row['vs_masked']:+.3f} vs facts only")
    c[1].metric("Facts-only baseline", f"{masked:.3f}")
    c[2].metric("Outcome words still present", f"{row['cue_rate']:.1%}")
    c[3].metric("Still better than facts?",
                "YES" if row["ci_excludes_zero"] else "no")
    if row["ci_excludes_zero"]:
        st.error(f"**Contaminated.** At {i} words back the window still beats facts alone by "
                 f"{row['vs_masked']:+.3f}, 95% CI {row['ci95']}. The outcome is still leaking in.")
    else:
        st.success(f"**Clean.** At {i} words back the advantage is gone — "
                   f"{row['vs_masked']:+.3f}, 95% CI {row['ci95']} spans zero. "
                   f"This window is no better than the facts.")

    st.line_chart(pd.DataFrame({"model score (AUROC)": [s["auroc"] for s in sweep],
                                "outcome words present": [s["cue_rate"] for s in sweep]},
                               index=[str(o) for o in offs]))
    st.caption("Both fall together as the window moves away from the deleted order — the signature "
               "of one shared cause.")

    st.subheader("Found four times independently")
    rows = []
    for k, lbl in (("land_forum", "land · forum-heldout"), ("land_temporal", "land · temporal"),
                   ("corpus_temporal", "all case types · temporal"),
                   ("corpus_random", "all case types · random")):
        d = F["setback"][k]
        rows.append({"split": lbl, "cases": d["n_test"],
                     "facts only": round(d["results"]["masked"], 3),
                     "last 512 words": round(d["results"]["w0"], 3),
                     "set back once": round(d["results"]["w1"], 3),
                     "clean at": f"{d['clean_at']} words"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("**The same ~128-word boundary every time**, across 26,688 judgments and every "
                "case type. Set the window back once and it is indistinguishable from the facts.")

    era = F["setback"]["corpus_random"].get("by_era") or {}
    if era:
        st.subheader("Not a historical quirk — significant in every decade, 1950s to 2020s")
        st.bar_chart(pd.DataFrame(
            {"advantage from the contaminated window":
             [v["w0_minus_masked"] for _, v in sorted(era.items())]},
            index=[f"{k}s" for k, _ in sorted(era.items())]))

    st.subheader("And it holds up against a real model, not just word counts")
    fm, fl = F.get("ft_masked"), F.get("ft_last512")
    if fm and fl:
        cc = st.columns(3)
        cc[0].metric("Fine-tuned · facts only", f"{fm['acc']:.1%}",
                     help=f"InLegalBERT, pretrained on Indian legal text; mean of "
                          f"{len(fm.get('seeds', []))} seeds")
        cc[1].metric("Fine-tuned · contaminated window", f"{fl['acc']:.1%}")
        cc[2].metric("Published state of the art", "78%")
        if fm.get("paired"):
            pm = fm["paired"]["mean_of_seeds_1_2"]
            st.caption(
                f"Facts-only AUROC {fm['auroc']:.3f} (sd {fm['sd']:.3f} over seeds) against "
                f"{fm['bar']['auroc']:.3f} for word counts on the same cases. The gain is small: "
                f"no single seed is significantly better (paired bootstrap), the average of two "
                f"seeds is ({pm['diff']:+.3f}, 95% CI [{pm['ci95'][0]:+.3f}, {pm['ci95'][1]:+.3f}]). "
                f"Facts alone top out near 0.70, whatever reads them.")
        st.warning(
            f"The **same model, same training, same cases**: {fl['acc']:.1%} reading the "
            f"contaminated window, {fm['acc']:.1%} reading the facts. The published 78% sits with "
            f"the contaminated number, not the honest one.\n\n"
            f"**Never compare {fm['acc']:.0%} to 78% directly** — they measure different things, "
            f"and ours is strictly harder. The gap *is* the result.")

# ───────────────────────────── ② explore a case ───────────────────────────────
with t2:
    st.header("What the system produces for a real case")
    st.caption("Facts are extracted verbatim and span-verified. The court's reasoning and its "
               "order are excluded — this is only what existed before the decision.")
    opts = {f"{c['year']} · {c['title'][:72]}": c for c in
            sorted(CASES, key=lambda x: x["year"])}
    case = opts[st.selectbox("Case", list(opts))]

    h = st.columns(4)
    h[0].metric("Year", case["year"])
    h[1].metric("Facts extracted", len(case["facts"]))
    h[2].metric("Actual outcome", case["outcome"] or "—")
    if case.get("url"):
        h[3].link_button("Full judgment", case["url"])

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Facts")
        TYPE_NOTE = {"FINDING": "⚖️ a court's conclusion, not evidence",
                     "LAW": "📘 a legal proposition, not a fact of this case",
                     "ASSERTION": "🗣️ a party's claim", "PROCEDURAL": "🏛️ a step in the litigation"}
        for n, f in enumerate(case["facts"][:18], 1):
            who = f.get("asserted_by") or "unattributed"
            st.markdown(f"**{n}.** {f['text']}")
            note = TYPE_NOTE.get(f.get("type") or "")
            if not note and f.get("prior_court_finding"):
                note = TYPE_NOTE["FINDING"]
            st.caption(f"asserted by: {who} · {f.get('disputed_status') or 'unknown'}"
                       + (f" · **{note}**" if note else ""))
        if len(case["facts"]) > 18:
            st.caption(f"… and {len(case['facts'])-18} more")

    with right:
        st.subheader("Legal requirements the facts bear on")
        st.caption("Every verdict must quote a fact. The quote is checked mechanically; if it is "
                   "not found, the verdict is **thrown away**. That catches invented support "
                   "without needing an answer key.")
        shown = 0
        for e in case["elements"]:
            if e["verdict"] == "UNCLEAR" or not e.get("gated"):
                continue
            meta = ELS.get(e["element_id"], {})
            sat = e["verdict"] == "SATISFIED"
            icon = "✅" if sat else "❌"
            with st.container(border=True):
                st.markdown(f"{icon} **{meta.get('name', e['element_id'])}** — "
                            f"{e['verdict'].replace('_',' ').lower()}"
                            + ("" if sat else "  ·  ⚠️ *low confidence*"))
                st.caption(f"{meta.get('claim','')} · favours {e.get('favours')}"
                           + (" · ~80% agreement with an independent reviewer" if sat else
                              " · 'not satisfied' verdicts agreed with an independent reviewer "
                              "under half the time"))
                if e["quote"]:
                    st.markdown(f"> {e['quote'][:300]}")
            shown += 1
        if not shown:
            st.info("The facts do not settle any element for this case — a common and honest "
                    "outcome, since 64% of all verdicts are UNCLEAR.")

        st.subheader("Statutes predicted from the facts")
        st.caption(f"Works: micro-F1 {F['statutes'].get('masked text', 0):.3f} against "
                   f"{F['statutes'].get('family prior (top-5)', 0):.3f} for the strongest "
                   f"baseline — **1.9×**.")
        st.dataframe(pd.DataFrame(case["statutes"], columns=["provision", "score"]),
                     hide_index=True, width="stretch", height=220)

        st.subheader("Precedents retrieved from the facts")
        st.caption(f"Works: recall@10 {F['precedents']['rrf(bm25+dense_masked_text)']['r10']:.3f} "
                   f"against {F['precedents']['popularity']['r10']:.3f} for a popularity "
                   f"control — **7.4×**.")
        st.dataframe(pd.DataFrame([{"year": p["year"], "case": (p["title"] or p["doc_id"])[:60]}
                                   for p in case["precedents"]]),
                     hide_index=True, width="stretch", height=220)

# ──────────────────────────── ③ your own facts ───────────────────────────────
with t3:
    st.header("Retrieve precedents from facts you type")
    st.caption("Live BM25 over 6,911 land-dispute judgments. This is the part of the pipeline that "
               "genuinely works — the facts predict the *law* well, even though they do not "
               "predict who wins.")
    txt = st.text_area(
        "Describe the dispute in plain terms",
        "The plaintiff has been cultivating the disputed land openly since 1962 without any "
        "objection from the recorded owner. No rent was ever paid and no permission was granted. "
        "The defendant now claims title under a registered sale deed executed in 1998.",
        height=140)
    k = st.slider("How many to retrieve", 5, 25, 10)
    if st.button("Retrieve", type="primary"):
        B = load_bm25()
        hits = B["index"].search(txt, k=k)
        rows = []
        for h in hits:
            d = h[0] if isinstance(h, (tuple, list)) else h
            s = h[1] if isinstance(h, (tuple, list)) and len(h) > 1 else None
            m = B["titles"].get(d, {})
            rows.append({"year": m.get("year"), "case": (m.get("title") or d)[:80],
                         "score": round(s, 3) if isinstance(s, (int, float)) else None})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("In the evaluated pipeline every retrieved authority is additionally asserted "
                   "to pre-date the query judgment. Here there is no query judgment to pre-date, "
                   "so that filter is off.")

# ─────────────────────── ④ why the symbolic pipeline fails ────────────────────
with t4:
    st.header("We built the lawyer-like pipeline. Here is where it loses information.")
    st.markdown("The intended design: **facts → standard categories → patterns → legal "
                "requirements → verdict.** Each arrow was measured separately.")

    if F.get("ladder"):
        st.subheader("Each step toward clean legal structure costs signal")
        ld = [r for r in F["ladder"] if r.get("auroc")]
        st.dataframe(pd.DataFrame([{"representation": r["name"], "AUROC": round(r["auroc"], 3),
                                    "features": r["feats"]} for r in ld]),
                     hide_index=True, width="stretch")

    st.subheader("Two fixes, measured")
    s2, ws = F.get("s2"), F.get("why_s4")
    if s2 and ws:
        a, b = st.columns(2)
        with a:
            st.markdown("**Fix 1 — stop forcing each fact into one box.** Keep it soft.")
            st.metric("soft, 345 dims", f"{s2['best_auroc']:.3f}",
                      f"{s2['best_auroc']-s2['v1_reference']['canonical_atoms']:+.3f} vs forcing "
                      f"one box")
            st.success("**Worked.** 345 numbers do the work of 47,354 — this step loses almost "
                       "nothing when built as designed.")
        with b:
            st.markdown("**Fix 2 — do the same for the legal-requirement layer.**")
            st.metric("soft elements, 66 dims", f"{ws['results']['S4 H2 soft elements']:.3f}",
                      f"{ws['results']['S4 H2 soft elements']-ws['results']['S4 old encoding']:+.3f}"
                      f" vs hard")
            st.error("**Did nothing.** So the problem is not how the elements are encoded.")

    if F.get("clusters"):
        st.subheader("Why: the signal is spread thin, not concentrated")
        st.markdown("Sorting 66 data-driven fact groups by what *kind* of fact they are, then "
                    "scoring each kind alone:")
        wd = F["why_s4"]["results"]
        st.dataframe(pd.DataFrame([
            {"basis": "66 groups the **data** suggests", "AUROC": 0.632},
            {"basis": "66 requirements the **law** prescribes", "AUROC": 0.524},
            {"basis": "legal-substance facts only", "AUROC": 0.606},
            {"basis": "court-procedure facts only", "AUROC": 0.609},
            {"basis": "dates and amounts only", "AUROC": 0.604}]),
            hide_index=True, width="stretch")
        st.info("**Every slice scores about the same.** There is no concentrated signal to find — "
                "it is weak and smeared across everything. A representation that *covers* the "
                "whole space catches it; a checklist of 66 specific legal questions, only ~8 asked "
                "per case, misses it.\n\n**So no better ontology would fix this.** There is no "
                "joint to carve.")
        st.markdown("**But the groups do say something real.** Facts about land-reform "
                    "legislation, proprietary rights vesting in the State and princely-state "
                    "accession all point to **LOSE**. Facts about document execution and contract "
                    "performance point to **WIN**. Post-independence land reform against the "
                    "State loses; private contract disputes win. That is about *what kind of "
                    "fight this is* — not whether a box was ticked.")

    if F.get("s4_layer"):
        st.subheader("The element layer is still the most trustworthy thing here")
        sl = F["s4_layer"]
        m = st.columns(3)
        m[0].metric("Verdicts produced", f"{sum(sl['verdicts'].values()):,}")
        m[1].metric("Invented support", f"{sl['fabrication']:.2%}")
        m[2].metric("Agreement with itself", "96.9%")
        eq = F.get("element_quality")
        if eq:
            g = eq["gold_by_arm"]["tag"]
            q = st.columns(3)
            q[0].metric("Agreement with a blind reviewer", f"{g['agree']/g['n']:.0%}",
                        help=eq["reviewer"])
            q[1].metric("'Satisfied' precision", f"{g['sat_prec'][0]/max(1,g['sat_prec'][1]):.0%}")
            q[2].metric("'Not satisfied' precision", f"{g['not_prec'][0]/max(1,g['not_prec'][1]):.0%}")
            st.caption(f"{eq['n']} judgements on 40 cases, scored against {eq['reviewer']}. "
                       "Positive verdicts are usable; defeat verdicts are not yet — about half rest "
                       "on a court's conclusion or a procedural fact read as a contradiction.")
        st.caption("Useless for predicting the winner, but it must quote its evidence and the "
                   "quote is checked. “Here are the requirements your facts satisfy, with the "
                   "quote for each” is **checkable**. A win/lose prediction never is.")
