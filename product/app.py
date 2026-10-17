"""LexGraph — research briefs for Indian property litigators.

    .venv/bin/streamlit run product/app.py

Paste a client's story, or open a held-out example. Every authority in the brief is one of the
retrieved precedents (certified), every quoted fact is checked against the story, every provision
against the predicted set, and each precedent shows how later courts treated it.

Held-out examples run AS OF their decision year with the case itself excluded, and show which of the
system's precedents the court actually cited -- the same setting as product/experiments/.
"""
from __future__ import annotations

import json

import streamlit as st

from src import paths
from product.pipeline import LexGraph

st.set_page_config(page_title="LexGraph", layout="wide")
P = paths.ROOT / "product"

BADGE = {"GOOD": "🟢 good law", "UNKNOWN": "⚪ no later treatment recorded",
         "DOUBTED": "🟠 doubted", "OVERRULED": "🔴 overruled"}
ASSESS = {"strong": "🟢 strong", "arguable": "🟡 arguable", "weak": "🔴 weak"}


@st.cache_resource(show_spinner="Loading the corpus, indexes and case cards…")
def engine() -> LexGraph:
    return LexGraph(verbose=False)


@st.cache_data
def examples() -> dict:
    nar = {r["case_id"]: r["narrative"] for r in map(json.loads, open(P / "data" / "client_narratives.jsonl"))
           if r["status"] == "ok"}
    reg = {r["doc_id"]: r for r in map(json.loads, open(paths.CASE_REGISTRY))}
    d = json.loads((P / "data" / "candidates_top50.json").read_text())
    return {c: {"narrative": t, "title": reg[c]["title"], "year": reg[c]["year"],
                "court_cited": d["targets"].get(c, [])} for c, t in nar.items()}


st.title("LexGraph")
st.caption("From a client's story to a research brief in which every authority is real and checkable. "
           "Indian Supreme Court land and property law, 1950–2025.")

with st.sidebar:
    mode = st.radio("Input", ["Your client's story", "Held-out example"])
    ex = examples()
    case_id, as_of, exclude = None, None, ()
    if mode == "Held-out example":
        opts = {f"{v['year']} · {v['title'][:60]}": k for k, v in sorted(ex.items(), key=lambda kv: kv[1]["year"])}
        case_id = opts[st.selectbox("Case (retold as a client's story)", list(opts))]
        as_of, exclude = ex[case_id]["year"], (case_id,)
        st.caption(f"Runs as of {as_of}: only earlier judgments, the case itself excluded.")
    st.divider()
    st.caption("Each run makes ~10 model calls on ASU Voyager and takes about a minute.")

default = ex[case_id]["narrative"] if case_id else ""
story = st.text_area("Client's story", value=default, height=220,
                     placeholder="What happened, in the client's words: the property, the documents, "
                                 "what the other side did, what the client wants.")
go = st.button("Prepare brief", type="primary", disabled=not story.strip())

if go:
    with st.spinner("Spotting issues, searching 6,911 judgments, reranking, checking good law, drafting…"):
        st.session_state["out"] = engine().analyse(story, as_of_year=as_of, exclude=exclude)
        st.session_state["case"] = case_id

out = st.session_state.get("out")
if out:
    if out.get("error"):
        st.error(out["error"])
        st.stop()
    prec = {p["id"]: p for p in out["precedents"]}
    cert = out["certification"] or {}
    c = st.columns(4)
    c[0].metric("Precedents certified", f"{cert.get('precedent_ok', 0)}/{cert.get('precedent_ok', 0) + cert.get('precedent_refused', 0)}")
    c[1].metric("Facts quoted verbatim", f"{cert.get('fact_ok', 0)}/{cert.get('fact_ok', 0) + cert.get('fact_refused', 0)}")
    c[2].metric("Provisions from prediction", f"{cert.get('provision_ok', 0)}/{cert.get('provision_ok', 0) + cert.get('provision_unverified', 0)}")
    c[3].metric("Time", f"{sum(out['timings_s'].values()):.0f}s")

    tab_b, tab_p, tab_e, tab_m = st.tabs(["Brief", "Precedents", "Elements", "Issue memo"])
    with tab_b:
        b = out["brief"]
        if not b:
            st.error("The brief could not be drafted for this story.")
        else:
            for n, iss in enumerate(b["issues"], 1):
                with st.container(border=True):
                    st.markdown(f"### {n}. {iss['issue']}")
                    st.markdown(f"**Assessment:** {ASSESS.get(iss['assessment'].strip().lower(), iss['assessment'])}")
                    l, r = st.columns(2)
                    l.markdown("**For the client**\n\n" + iss["client_argument"])
                    r.markdown("**The other side will say**\n\n" + iss["opponent_argument"])
                    if iss.get("precedents_checked"):
                        st.markdown("**Authorities**")
                        for x in iss["precedents_checked"]:
                            p = prec.get(x["ref"].strip())
                            if x["certified"] and p:
                                link = f"[{p['title']}]({p['url']})" if p.get("url") else p["title"]
                                st.markdown(f"✅ {link} — {BADGE[p['good_law']['status']]}")
                            else:
                                st.markdown(f"❌ `{x['ref']}` — refused: not among the retrieved precedents")
                    if iss.get("provisions_checked"):
                        st.markdown("**Provisions:** " + ", ".join(
                            f"`{x['label']}`" + ("" if x["predicted"] else " ⚠️ unverified")
                            for x in iss["provisions_checked"]))
                    for x in iss.get("facts_checked", []):
                        st.caption(("✅ " if x["verbatim"] else "❌ not in the story: ") + f"“{x['quote']}”")
            if b.get("gaps"):
                st.subheader("Before advising, obtain")
                for g in b["gaps"]:
                    st.markdown(f"- {g}")
    with tab_p:
        court = set(ex[st.session_state["case"]]["court_cited"]) if st.session_state.get("case") else set()
        if court:
            hits = sum(1 for p in out["precedents"] if p["doc_id"] in court)
            st.info(f"Held-out check: {hits} of these {len(out['precedents'])} were cited by the court "
                    f"in the real judgment ({len(court)} citable precedents in the corpus).")
        for p in out["precedents"]:
            with st.container(border=True):
                mark = " · ⭐ cited by the court" if p["doc_id"] in court else ""
                st.markdown(f"**{p['id']}. {p['title']}** ({p['year']}) — relevance {p['relevance']}/3 · "
                            f"{BADGE[p['good_law']['status']]}{mark}")
                g = p["good_law"]
                st.caption(f"later courts: followed {g['followed']} · distinguished {g['distinguished']} · "
                           f"doubted {g['doubted']} · overruled {g['overruled']}")
                if p["card"]:
                    st.markdown("**Decided:** " + "; ".join(p["card"]["issues_decided"]))
                    st.markdown("**Held:** " + "; ".join(p["card"]["holdings"]))
                for q in g["negative_quotes"]:
                    st.caption(f"⚠️ “{q}”")
                if p.get("url"):
                    st.link_button("Read the judgment", p["url"])
        st.subheader("Predicted provisions")
        st.markdown(", ".join(f"`{s}`" for s in out["provisions"]))
    with tab_e:
        st.caption("Each verdict quotes the client's own words. 'Not satisfied' means a stated fact contradicts "
                   "the element; silence is 'unclear'.")
        for cl in out["elements"]:
            st.markdown(f"#### {cl['claim']}")
            if not cl["elements"]:
                st.caption("No verdict returned for this claim.")
            for e in cl["elements"]:
                icon = {"SATISFIED": "✅", "NOT_SATISFIED": "❌", "UNCLEAR": "❔"}[e["verdict"]]
                st.markdown(f"{icon} **{e['element']}**" + (f" — “{e['quote']}”" if e["quote"] else ""))
    with tab_m:
        m = out["memo"]
        for k in ("issues", "claims", "provisions", "doctrines"):
            st.markdown(f"**{k.title()}**")
            for x in m[k]:
                st.markdown(f"- {x}")
        st.caption("Provisions named in the memo are the model's suggestions and are NOT verified; only 21% "
                   "matched what courts relied on in evaluation. The brief uses predicted provisions instead.")
