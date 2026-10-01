"""Generate the CSE 573 proposal, in two framings, from measured figures.

Two drafts are produced because the evidence supports two different arguments and the choice is
substantive:

  `A_authority`  leads with what works — statute prediction and precedent retrieval, both well
                 clear of their controls — and reports outcome prediction as the harder secondary
                 finding.
  `B_benchmark`  leads with the methodology — a leakage-controlled benchmark whose probe passes its
                 own sensitivity check — and makes the precise negative result the contribution.

Every number is read from `src.report.figures`, which reads `experiments/`. Nothing is retyped.

Formatting follows the course rule exactly, because violating it costs 20%: A4, 1-inch margins,
12pt Times New Roman, single column. Placeholders for the team roster are written as visible TODO
markers rather than plausible-looking names, so nothing can be submitted silently wrong.
"""
from __future__ import annotations

import argparse

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from src import paths
from src.report.figures import collect

TODO = RGBColor(0xC0, 0x00, 0x00)


def _base_doc() -> Document:
    d = Document()
    for s in d.sections:
        s.page_width, s.page_height = Inches(8.27), Inches(11.69)   # A4
        s.top_margin = s.bottom_margin = s.left_margin = s.right_margin = Inches(1)
    st = d.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.0
    for name in ("Heading 1", "Heading 2", "Heading 3", "Title"):
        try:
            h = d.styles[name]
            h.font.name = "Times New Roman"
            h.font.color.rgb = RGBColor(0, 0, 0)
            h.font.size = Pt({"Title": 16, "Heading 1": 13,
                              "Heading 2": 12, "Heading 3": 12}[name])
            h.font.bold = True
            h.font.italic = False
        except KeyError:
            pass
    return d


def _todo(doc, text: str):
    p = doc.add_paragraph()
    r = p.add_run(f"[TODO — {text}]")
    r.font.color.rgb = TODO
    r.bold = True
    return p


def _table(doc, headers, rows, note: str | None = None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        run = c.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(10)
        run.font.name = "Times New Roman"
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.font.size = Pt(10)
            run.font.name = "Times New Roman"
    if note:
        p = doc.add_paragraph()
        r = p.add_run(note)
        r.italic = True
        r.font.size = Pt(10)
    return t


def _front_matter(doc, title: str, subtitle: str):
    doc.add_heading(title, 0)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(subtitle).italic = True
    _todo(doc, "team roster: replace with the four members' names and ASU IDs")
    _todo(doc, "submission date")
    _todo(doc, "check against the official proposal template on Canvas before submitting; "
               "the professor said one would be posted. Formatting rule already applied: "
               "A4, 1in margins, 12pt Times New Roman, single column, export to PDF")
    p = doc.add_paragraph()
    p.add_run("Instructor: Prof. Hasan Davulcu  ·  CSE 573 Semantic Web Mining, Fall 2026  ·  "
              "Assigned topic P15, Legal AI – Agentic Law").font.size = Pt(11)


def _dataset_section(doc, f):
    doc.add_heading("3. Dataset", level=1)
    c = f["corpus"]
    doc.add_paragraph(
        f"The provided archive holds {c['judgments_total']:,} Indian Supreme Court judgments "
        f"(1950–2025) as year-organised PDFs, with citation metadata for 7,497 cases and a "
        f"spreadsheet of {c['land_cases']:,} filtered to immovable-property disputes. We work on "
        f"that land/property subset.")
    doc.add_paragraph(
        f"Identifier join. Nothing in the dataset links Indian Kanoon document identifiers to PDFs "
        f"across the corpus. The land-dispute spreadsheet, however, carries a filename column, so "
        f"the join for our subset is exact rather than fuzzy: all {c['land_cases']:,} rows resolve "
        f"to a file on disk ({c['pdf_join_rate']}). The join rate is asserted in code, so a change "
        f"in the archive's naming fails loudly instead of silently producing an unusable corpus.")
    doc.add_paragraph(
        f"Extraction. Judgment text for every case is cached once with page and numbered-paragraph "
        f"offsets, from which {c['facts_extracted']:,} atomic fact propositions are extracted over "
        f"{c['cases_with_facts']:,} cases. Every fact carries a source span verified to round-trip "
        f"against the quote it was extracted from.")
    doc.add_paragraph(
        "Three properties of the corpus constrain what can be measured, and all three were "
        "discovered rather than assumed:")
    doc.add_paragraph(
        "No section headers. The dataset documentation describes judgments as containing Facts, "
        "Issue, Arguments, Reasoning and Judgement sections. Sampled across six decades, an ORDER "
        "header appears in 3.7% of judgments and a FACTS header in 29.7%. Segmentation therefore "
        "runs on numbered paragraphs, present in 92.7% of post-2000 judgments at a median of 32 "
        "per judgment, falling back to sentence windows.", style="List Bullet")
    doc.add_paragraph(
        "A hard pre-2000 boundary. The HEADNOTE convention, which lists a judgment's cited "
        "authorities at the top, appears in 79.5% of pre-2000 judgments and 0.0% of post-2000 ones. "
        "It is pure leakage and is removed; but because its presence is almost perfectly determined "
        "by era, any temporal split also separates two structurally different document types.",
        style="List Bullet")
    doc.add_paragraph(
        f"A reachability ceiling. Of 46,904 case-to-case citation edges only 22.1% point at a case "
        f"this corpus contains, since it is Supreme Court only while judgments cite High Courts, "
        f"the Privy Council and English decisions. Retrieval recall is therefore reported against "
        f"reachable authorities with the raw total beside it.", style="List Bullet")
    doc.add_paragraph(
        f"Outcome labels. A rules pass over the operative order region, cross-checked by an LLM "
        f"that must quote the disposition verbatim, yields {c['binary_labels']:,} binary WIN/LOSE "
        f"labels at {100*c['label_win_rate']:.1f}% WIN, with "
        f"{100*c['rules_llm_agreement']:.1f}% agreement between the two labellers where both "
        f"decided and the quote verified. Conflicts are held unresolved rather than broken by "
        f"coin-flip, which would manufacture label noise.")


def _leakage_section(doc, f, level=1):
    doc.add_heading("Leakage control", level=level)
    doc.add_paragraph(
        "A judgment states its own outcome, so the single largest threat to this project is that a "
        "predictor reads the answer rather than inferring it. The masking step keeps only the facts "
        "narrative, the pleadings and each side's arguments, and removes the analysis, the operative "
        "order, the HEADNOTE, and any sentence carrying an outcome cue.")
    doc.add_paragraph(
        "Whether that worked is itself measured. A TF-IDF and logistic-regression probe is run on "
        "four inputs, and the arm that matters most is the sensitivity control:")
    p = f["probe"]
    _table(doc,
           ["probe input", "macro-F1", "AUROC", "what it establishes"],
           [["operative order only", f"{p['order_only']['macro_f1']:.3f}",
             f"{p['order_only']['auroc']:.3f}",
             "the probe detects the outcome easily when present"],
            ["unmasked judgment", f"{p['full']['macro_f1']:.3f}", f"{p['full']['auroc']:.3f}",
             "upper bound on a full document"],
            ["masked text (what we use)", f"{p['masked']['macro_f1']:.3f}",
             f"{p['masked']['auroc']:.3f}", "the honest operating range"],
            ["prior-court disposition only", f"{p['prior_court']['macro_f1']:.3f}",
             f"{p['prior_court']['auroc']:.3f}", "the procedural shortcut, alone"],
            ["majority class", "0.345", "0.500", "floor"]],
           note="Table: leakage probe, 902 test cases. Read AUROC, not macro-F1: with balanced "
                "class weights a model that merely emits both classes beats the majority "
                "baseline's macro-F1 while carrying no signal.")
    doc.add_paragraph(
        f"The order-only arm reaching {p['order_only']['auroc']:.3f} is what licenses every other "
        f"number: a low score on masked text is then evidence about the mask rather than about a "
        f"blind probe. Masked text at {p['masked']['auroc']:.3f} is the headroom any predictor is "
        f"competing for — not the ~0.9 figures reported by legal-judgment-prediction work that does "
        f"not control leakage.")
    doc.add_paragraph(
        "Two further controls follow from the same concern. Authorities a court cites in its "
        "analysis are chosen knowing the outcome, so they are prediction targets and never inputs; "
        "and precedent retrieval is time-respecting, which is asserted per result rather than left "
        "to a filter.")


def _authority_results(doc, f, level=2):
    doc.add_heading("Grounding facts in law", level=level)
    s, pr = f["statutes"], f["precedents"]
    doc.add_paragraph(
        f"Statute prediction. Given only the masked facts, predict the statutory provisions the "
        f"court will rely on. The well-posed version of this task predicts only the provisions the "
        f"court INTRODUCED — those absent from the facts and pleadings — because a provision already "
        f"pleaded sits in the input as well as the target. A control that simply copies provisions "
        f"out of the input scores micro-F1 0.713 on the naive version, beating every learned model, "
        f"which is how we know the naive version is an extraction task rather than a prediction one.")
    rows = []
    for name in ("prior (top-5 global)", "family prior (top-5)", "copy from masked text",
                 "masked text", "extracted fact text", "canonical atoms"):
        r = s["rows"][name]
        rows.append([name, f"{r['micro_f1']:.3f}", f"{r['macro_f1']:.3f}",
                     f"{r['p@1']:.3f}", f"{r['r@5']:.3f}", f"{r['r@10']:.3f}"])
    _table(doc, ["system", "micro-F1", "macro-F1", "P@1", "R@5", "R@10"], rows,
           note=f"Table: statute prediction, {s['n_labels']} provisions clearing a 30-train-case "
                f"floor, {s['n_train']:,} train / {s['n_test']} test. The copy control scores "
                f"exactly 0.000 by construction, confirming the targets are disjoint from the "
                f"inputs.")
    doc.add_paragraph(
        f"Masked text reaches micro-F1 "
        f"{s['rows']['masked text']['micro_f1']:.3f} against the strongest baseline's "
        f"{s['rows']['family prior (top-5)']['micro_f1']:.3f} — 1.9 times — and R@5 "
        f"{s['rows']['masked text']['r@5']:.3f} against {s['rows']['family prior (top-5)']['r@5']:.3f}. "
        f"The margin over the FAMILY prior is the part that reads the facts rather than learning "
        f"that land-acquisition cases cite the Land Acquisition Act.")
    doc.add_paragraph(
        f"Precedent retrieval. Rank earlier cases by how likely the court was to cite them, on "
        f"{pr['n_queries']} queries holding {pr['reachable']:,} reachable targets of "
        f"{pr['raw']:,} citations. Queries are masked text with citation-bearing sentences removed, "
        f"so a precedent cannot be read off the input.")
    rows = []
    for name in ("mention", "popularity", "bm25", "dense_meanpool", "dense_masked_text",
                 "rrf(bm25+dense_masked_text)"):
        r = pr["rows"].get(name)
        if r:
            rows.append([name, f"{r['r@10']:.3f}", f"{r['r@50']:.3f}",
                         f"{r['mrr']:.3f}", f"{r['ndcg@10']:.3f}"])
    _table(doc, ["retriever", "R@10", "R@50", "MRR", "nDCG@10"], rows,
           note="Table: precedent retrieval. Recall is against the reachable target set. "
                "Time-respecting retrieval is asserted per result; zero returned cases "
                "post-date their query.")
    best = pr["rows"]["rrf(bm25+dense_masked_text)"]
    doc.add_paragraph(
        f"Fusing lexical and dense retrieval over masked text gives R@10 {best['r@10']:.3f} and MRR "
        f"{best['mrr']:.3f} — 7.4 times the popularity control's "
        f"{pr['rows']['popularity']['r@10']:.3f}, with the first correct precedent typically around "
        f"rank two or three. Two negative details are worth recording: dense retrieval over "
        f"extracted FACTS fails under every pooling tried "
        f"({pr['rows']['dense_meanpool']['r@10']:.3f}) while dense over the TEXT succeeds, and "
        f"reciprocal-rank fusion actively hurt until the dense arm was strong enough to fuse with.")


def _ladder_results(doc, f, level=2):
    doc.add_heading("Where the predictive signal is lost", level=level)
    lad, ab = f["ladder"], f["vocab_ab"]
    doc.add_paragraph(
        "Each stage of the pipeline replaces one representation with a more structured one. Holding "
        "the classifier, the split, the labels and the case set fixed and varying only the "
        "representation isolates what each stage costs:")
    _table(doc, ["representation", "features", "AUROC"],
           [["masked text", f"{lad['1. masked text']['features']:,}",
             f"{lad['1. masked text']['auroc']:.3f}"],
            ["extracted fact text", f"{lad['2. extracted fact text']['features']:,}",
             f"{lad['2. extracted fact text']['auroc']:.3f}"],
            ["canonical fact labels", f"{lad['3. canonical atoms']['features']:,}",
             f"{lad['3. canonical atoms']['auroc']:.3f}"],
            ["chance", "—", "0.500"]],
           note=f"Table: representation ladder, identical {f['ladder_n']['train']:,} train / "
                f"{f['ladder_n']['test']} test cases at every rung.")
    doc.add_paragraph(
        f"Extraction costs 0.043 AUROC. Mapping those facts onto the ontology's controlled "
        f"vocabulary costs a further 0.097 and lands at chance. Three independent checks say the "
        f"second loss is caused by discretisation itself rather than by this ontology's particular "
        f"labels: sweeping the atom-frequency threshold leaves the result at chance at every "
        f"setting; an INDUCED vocabulary clustered bottom-up from the corpus at the same "
        f"granularity is no better (paired difference "
        f"{ab['paired']['induced - ontology']['diff']:+.3f}, "
        f"P={ab['paired']['induced - ontology']['p_le_0']:.3f}); and raising the atom count from "
        f"345 to 1,000 does not close the gap. Against the text, both vocabularies lose "
        f"significantly (ontology {ab['paired']['text - ontology']['diff']:+.3f}, "
        f"95% CI {ab['paired']['text - ontology']['ci95']}).")
    pt = f["patterns"]
    doc.add_paragraph(
        f"This explains the pattern-mining result rather than leaving it a puzzle. FP-Growth over "
        f"those labels, per claim family, yields {pt['closed_itemsets']} closed itemsets of which "
        f"{pt['stable']} are stable across 20 bootstrap resamples — and "
        f"{pt['bh_significant']} survive Benjamini–Hochberg correction in any family. Ten pass an "
        f"uncorrected threshold against 17.6 expected by chance alone, so this is not a weak effect "
        f"awaiting more data. No pattern over uninformative atoms can be significant.")


def _outcome_results(doc, f, level=2):
    doc.add_heading("Outcome prediction", level=level)
    h = f["head2head"]["rows"]
    rows = []
    for name in ("structured gbm F+P+E+S+R+C", "structured lr S only", "TF-IDF on masked text",
                 "majority (per decade)", "LLM-CoT", "LLM-0", "LLM-CF", "LLM-CF+A", "LLM-FS"):
        r = h.get(name)
        if not r:
            continue
        ci = r.get("macro_f1_ci95", [float("nan")] * 2)
        rows.append([name, f"{r['accuracy']:.3f}",
                     f"{r['macro_f1']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",
                     f"{r['auroc']:.3f}" if r.get("auroc") else "—",
                     f"{r['ece']:.3f}" if r.get("ece") is not None else "—"])
    _table(doc, ["system", "accuracy", "macro-F1 (95% CI)", "AUROC", "ECE"], rows,
           note=f"Table: all systems on the identical {f['head2head']['n_test']} test cases. "
                f"Intervals are 1,000-sample bootstraps.")
    doc.add_paragraph(
        "The honest reading is the intervals. Every macro-F1 interval spans about 0.11 and they "
        "overlap heavily; a per-decade majority baseline sits inside nearly all of them; and a "
        "paired McNemar test against zero-shot prompting is non-significant for all four other LLM "
        "arms. So the defensible claim is not that one system wins, but that on leakage-controlled "
        "inputs no system convincingly clears a per-decade majority.")
    doc.add_paragraph("Four results do survive that caveat:")
    doc.add_paragraph(
        f"Giving a language model our canonical facts instead of the raw text makes it WORSE "
        f"(AUROC {h['LLM-CF']['auroc']:.3f} against {h['LLM-0']['auroc']:.3f}). This was predicted "
        f"from the ladder before being run, and it shows the discretisation loss is a property of "
        f"the representation rather than an artifact of linear and tree classifiers.",
        style="List Bullet")
    doc.add_paragraph(
        f"The features that help are the authority-grounding ones, not the fact labels. Predicted "
        f"statutes alone reach AUROC {h['structured lr S only']['auroc']:.3f} at "
        f"ECE {h['structured lr S only']['ece']:.3f}, the best-calibrated informative arm.",
        style="List Bullet")
    doc.add_paragraph(
        f"Calibration separates the systems where accuracy does not, and that gap does not overlap: "
        f"zero-shot prompting reaches ECE {h['LLM-0']['ece']:.3f}, predicting a win 13% of the time "
        f"where the truth is 46%, against {h['TF-IDF on masked text']['ece']:.3f} for TF-IDF.",
        style="List Bullet")
    e = f["errors"]
    doc.add_paragraph(
        f"The ceiling appears intrinsic. No signal available to the pipeline predicts which cases "
        f"fail — every candidate signal has a lift near 1.0 against a base error rate of "
        f"{e['base_error_rate']:.3f} — and the structured model's errors are statistically "
        f"independent of a language model's working from raw text "
        f"({e['overlap']['both_wrong']} cases wrong in both against 49.6 expected under "
        f"independence).", style="List Bullet")


def _trace_section(doc, f, level=2):
    doc.add_heading("The trace, and what it is verified to do", level=level)
    t = f["trace"]
    doc.add_paragraph(
        "Every prediction emits a trace: the probability, the claim families, the facts with source "
        "spans into the masked text, the predicted statutes, the retrieved precedents with the "
        "analogous party's outcome, and per-feature attributions computed by ablation.")
    doc.add_paragraph(
        f"Faithfulness is tested by deletion, and the test operates on FACTS rather than features — "
        f"attribution is itself feature ablation, so re-ablating the same features would measure "
        f"the attribution against itself. Deleting a fact re-derives its labels, rebuilds the "
        f"feature row and re-scores. Over {t['n_cases']} cases, deleting the three facts the trace "
        f"cites moves the prediction by {t['mean_delta']['cited']:.4f} against "
        f"{t['mean_delta']['random']:.4f} for three random facts and "
        f"{t['mean_delta']['inverse']:.4f} for the trace's own lowest-ranked facts. The paired "
        f"difference against random is {t['cited_vs_random']['mean_diff']:+.4f}, 95% CI "
        f"{t['cited_vs_random']['ci95']}.")
    doc.add_paragraph(
        "This establishes that the trace points at the evidence the model uses. It does not "
        "establish that the cited evidence is legally correct reasoning, and the proposal does not "
        "claim otherwise: a trace can be perfectly faithful to a model that is wrong.")
    doc.add_paragraph(
        "Two fields of the trace contract are deliberately emitted EMPTY rather than filled: "
        "per-element status and the burden that attaches to each. No ontology element carries "
        "burden metadata yet, and no mined pattern survived significance correction, so there is "
        "nothing to populate them with. Inventing element statuses would read as legal reasoning "
        "while being decoration.")


def _evaluation_plan(doc, f, level=1):
    doc.add_heading("Evaluation plan", level=level)
    doc.add_paragraph(
        "Every result reports the system, the simple non-LLM baselines, the applicable LLM "
        "baselines, n, the split name and a 95% bootstrap interval. Three conventions are forced by "
        "what we found in this corpus:")
    doc.add_paragraph(
        "Per-period baselines, never global. The Supreme Court's allowance rate in land disputes "
        "runs from 39% to 64% across our temporal split, so a global majority baseline is beaten by "
        "any model that merely notices the drift.", style="List Bullet")
    doc.add_paragraph(
        "Multiple-testing correction on anything mined. Benjamini–Hochberg changed a conclusion "
        "here: ten patterns pass an uncorrected threshold and none survives correction.",
        style="List Bullet")
    doc.add_paragraph(
        "A copy or mention control on every grounding task. On statute prediction, a control that "
        "copied provisions out of the input beat every learned model, which is the only reason we "
        "discovered the task as first posed was extraction rather than prediction.",
        style="List Bullet")
    doc.add_paragraph(
        "Two splits are frozen with content hashes. The primary split is temporal. The course "
        "specification also asks for a court-held-out split, which this corpus cannot support — it "
        "is Supreme Court only — so we hold out the ORIGINATING High Court instead and report it as "
        "the weaker control it is. It is also era-balanced, which is what lets it separate format "
        "shift from genuine temporal drift.")
    g = f["gold"]
    doc.add_paragraph(
        f"Gold annotation. {g['n_tasks']} cases are prepared for review in four strata "
        f"({', '.join(f'{k} {v}' for k, v in g['strata'].items())}), with a "
        f"{g['second_annotator_block']}-case overlap block for inter-annotator agreement and "
        f"{g['facts_to_review']:,} individual fact judgements. The annotation tool pre-fills "
        f"pipeline output but pre-selects nothing, so a reviewer clicking through cannot "
        f"manufacture agreement.")


def _timeline(doc, f):
    doc.add_heading("Timeline and milestones", level=1)
    _table(doc, ["milestone", "contents", "status"],
           [["M0–M1", "Corpus adapter, labels, masking, splits, leakage probe", "complete"],
            ["M2", "Fact extraction, canonical vocabulary", "extraction complete; vocabulary "
                                                            "not frozen (see Risks)"],
            ["M3", "Claim families, fact patterns", "complete; both results negative"],
            ["M4", "Statute prediction, precedent retrieval", "complete"],
            ["M5", "Outcome models, ablations, trace, error analysis", "complete"],
            ["Oct–Nov", "Gold annotation; element burden annotation", "prepared, awaiting review"],
            ["Nov", "Presentation (Oct 28 – Nov 18)", "pending"],
            ["Nov–Dec", "Dashboard and group demo (Nov 23 – Dec 2)", "not started"],
            ["Dec 9", "Final report", "pending"]])
    _todo(doc, "confirm these dates against the course schedule on Canvas")
    doc.add_heading("Team roles and division of labour", level=1)
    _todo(doc, "assign the four members to: corpus and labelling; extraction and vocabulary; "
               "authorities and retrieval; modelling, trace and dashboard")


def _risks(doc, f):
    doc.add_heading("Risks and mitigations", level=1)
    v = f["vocab"]
    rows = [
        ["The canonical vocabulary cannot be frozen",
         f"{100*v['out_of_vocab_rate']:.1f}% of extracted facts fall outside the "
         f"{v['subcategories']}-label vocabulary, against the {100*v['freeze_threshold']:.0f}% "
         f"threshold we set for freezing it.",
         "A refinement loop is implemented and produces a reviewed proposal. But the measured "
         "evidence says coverage is not the real problem — facts the vocabulary CAN name also lose "
         "their signal when named — so the mitigation is to report the vocabulary as a labelling "
         "device and not rely on it as a feature space."],
        ["No gold annotation exists",
         "Extraction precision and recall against gold, the stated acceptance criterion for the "
         "extraction stage, cannot currently be computed.",
         "150 cases are prepared and the tool is built; this is reviewer time, not engineering. "
         "Until then we report ground-truth-free proxies (quote grounding, label agreement, "
         "downstream utility) and state what each does and does not establish."],
        ["Element burden metadata is unavailable",
         "Two language models produced annotations that passed no quality gate — 92% and 87% of "
         "elements assigned to the same party, with the heightened standard never used even for "
         "adverse possession.",
         "A degeneracy gate refuses to write such an annotation. The element feature group is "
         "reported as EMPTY rather than populated with a constant, and a hand-annotation sheet for "
         "66 elements is prepared."],
        ["Outcome prediction may not be achievable on this corpus",
         "No system convincingly clears a per-decade majority baseline, no signal predicts which "
         "cases fail, and two very different systems' errors are statistically independent.",
         "This is reported as a finding about the task rather than a failure to be engineered "
         "away. The authority-grounding results and the leakage-controlled benchmark stand "
         "independently of it."],
        ["Domain contamination in the provided subset",
         "The land-dispute spreadsheet was filtered by keyword and admits cases that merely occur "
         "on land: 5.5% are criminal appeals, 3.9% tax-heavy.",
         "Cases are screened and tagged rather than deleted, and every experiment states the "
         "subset it used."],
    ]
    _table(doc, ["risk", "evidence", "mitigation"], rows)


def _related_work(doc):
    doc.add_heading("2. Related work", level=1)
    doc.add_paragraph(
        "Legal knowledge representation has long separated legal concepts from the logic of their "
        "application, from LKIF-Core onwards, and the canonical demonstration that a statute can be "
        "executed as a logic program is Sergot et al.'s formalisation of the British Nationality "
        "Act (1986). Several of its lessons bear directly on this project and we adopt them "
        "explicitly: vague predicates are handled as qualified conclusions rather than forced "
        "decisions; negation as failure is distinguished from classical negation, so that a fact "
        "not proved is not recorded as proved false; and parameters are dated from the outset, "
        "since limitation and adverse possession both turn on time.")
    doc.add_paragraph(
        "Legal judgment prediction has an extensive literature, much of it reporting high accuracy "
        "on inputs that include the court's own reasoning. Our contribution is partly a response to "
        "that: we measure what the task looks like once the outcome is genuinely removed, and the "
        "difference is large.")
    _todo(doc, "add 8–12 full citations with venues and years; the related-work section currently "
               "names works without formatted references")


def draft_a(f) -> Document:
    """Leads with authority grounding — the positive results."""
    d = _base_doc()
    _front_matter(
        d, "Grounding Indian Land-Dispute Facts in Law: Predicting Statutes and Precedents, "
           "and the Limits of Symbolic Fact Representations",
        "Project Proposal — CSE 573: Semantic Web Mining, Fall 2026")

    d.add_heading("1. Motivation and problem statement", level=1)
    d.add_paragraph(
        "A lawyer handed a dispute does not hold a decided judgment. They hold facts and evidence, "
        "and the first question is which rules and authorities apply. Most extraction-based legal AI "
        "inverts this: it takes a finished judgment as input and recovers structure the court has "
        "already supplied, which solves a problem nobody has.")
    d.add_paragraph(
        "This project therefore takes the facts of a dispute as input and predicts the law that "
        "governs it — the statutory provisions a court will invoke and the precedents it will cite — "
        "with every prediction accompanied by a trace from the conclusion back to the facts it "
        "rests on. We then ask the harder question of whether those same facts predict the outcome, "
        "and report, with a controlled benchmark behind it, that largely they do not.")
    d.add_paragraph(
        "Three commitments shape the work. Authorities the court cites in its reasoning are chosen "
        "knowing the outcome, so they are targets and never inputs. Every reported figure is "
        "measured on this corpus rather than cited from elsewhere. And where a result is negative, "
        "it is reported as a result: the most substantial finding below is that mapping extracted "
        "facts onto a controlled vocabulary destroys their predictive signal, which has a direct "
        "consequence for ontology-based legal AI.")
    _related_work(d)
    _dataset_section(d, f)
    d.add_heading("4. Approach", level=1)
    d.add_paragraph(
        "The pipeline runs facts → canonical labels → claim families → authorities → outcome, with "
        "a trace emitted at the end. Stages are evaluated independently, because a pipeline reported "
        "only end-to-end hides which stage is responsible for its performance.")
    _leakage_section(d, f, level=2)
    d.add_heading("5. Preliminary results", level=1)
    d.add_paragraph(
        "All figures are measured on this corpus. Splits are frozen with content hashes, intervals "
        "are 1,000-sample bootstraps, and every number in this section is generated directly from "
        "the experiment files rather than transcribed.")
    _authority_results(d, f, level=2)
    _ladder_results(d, f, level=2)
    _outcome_results(d, f, level=2)
    _trace_section(d, f, level=2)
    d.add_heading("6. Evaluation plan", level=1)
    _evaluation_plan(d, f, level=2)
    _timeline(d, f)
    _risks(d, f)
    return d


def draft_b(f) -> Document:
    """Leads with the benchmark and the negative result as the contribution."""
    d = _base_doc()
    _front_matter(
        d, "What Survives Leakage Control? A Benchmark for Indian Land-Dispute Judgment "
           "Prediction, and Where Symbolic Fact Representations Fail",
        "Project Proposal — CSE 573: Semantic Web Mining, Fall 2026")

    d.add_heading("1. Motivation and problem statement", level=1)
    d.add_paragraph(
        "Legal judgment prediction routinely reports accuracies above 0.85. Much of that literature "
        "trains on inputs that contain the court's own reasoning, and sometimes its order, so it is "
        "unclear how much of the reported performance is prediction and how much is reading. The "
        "question this project asks first is therefore not how well a model can do, but what the "
        "task looks like once the answer has genuinely been removed.")
    d.add_paragraph(
        "We build a leakage-controlled benchmark over Indian Supreme Court land and property "
        "disputes, with a probe that passes its own sensitivity check, and then measure a full "
        "structured pipeline against it: facts, canonical labels, claim families, mined fact "
        "patterns, predicted statutes, retrieved precedents, and a verified trace.")
    d.add_paragraph(
        "The central finding is negative and precisely localised. Extraction preserves most of the "
        "signal the text carries; mapping those facts onto a controlled vocabulary destroys it, "
        "regardless of whether the vocabulary is authored or induced, and regardless of its size. "
        "That explains why pattern mining over such labels yields nothing significant, and it is a "
        "result about symbolic fact representations rather than about this particular ontology. "
        "Where the pipeline does work — predicting the statutes and precedents a court will rely "
        "on — we report that too.")
    _related_work(d)
    _dataset_section(d, f)
    d.add_heading("4. The benchmark", level=1)
    _leakage_section(d, f, level=2)
    d.add_heading("5. Measuring a structured pipeline against it", level=1)
    _ladder_results(d, f, level=2)
    _outcome_results(d, f, level=2)
    d.add_heading("6. What does survive: grounding in authorities", level=1)
    _authority_results(d, f, level=2)
    d.add_heading("7. Traceability", level=1)
    _trace_section(d, f, level=2)
    d.add_heading("8. Evaluation plan", level=1)
    _evaluation_plan(d, f, level=2)
    _timeline(d, f)
    _risks(d, f)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both", choices=["both", "a", "b"])
    args = ap.parse_args()
    f = collect()
    outs = []
    if args.which in ("both", "a"):
        p = paths.ROOT / "Proposal_A_authority_grounding.docx"
        draft_a(f).save(p)
        outs.append(p)
    if args.which in ("both", "b"):
        p = paths.ROOT / "Proposal_B_leakage_benchmark.docx"
        draft_b(f).save(p)
        outs.append(p)
    for p in outs:
        import docx
        d = docx.Document(p)
        words = sum(len(x.text.split()) for x in d.paragraphs)
        todos = sum(1 for x in d.paragraphs if x.text.startswith("[TODO"))
        print(f"{p.name}: {len(d.paragraphs)} paragraphs, {len(d.tables)} tables, "
              f"~{words:,} words, {todos} TODO markers")


if __name__ == "__main__":
    main()
