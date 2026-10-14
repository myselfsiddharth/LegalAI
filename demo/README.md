# demo — the group demo (15% of the grade)

```
.venv/bin/python -m demo.build_pack          # once, ~2 min
.venv/bin/streamlit run demo/app.py
```

## Design rule: nothing is computed at launch

The app reads **only** `demo/pack/`. No model is trained, no 517 MB file is opened, no API is
called. A demo that trains on launch is a demo that hangs in front of an audience, and a demo that
recomputes can disagree with the report. Every number traces to a file under `experiments/`,
`v2/experiments/` or `ft/experiments/`.

The one live computation is BM25 retrieval on tab ③, from a 23 MB pickled index over 6,911
judgments — no model, milliseconds.

## Why the tabs are in this order

**① The finding** first. The audience has to see *why this matters* before being asked to care
about an architecture. The centrepiece is the slider: push the reading window back from where the
order was deleted and watch the score collapse from 0.838 to 0.669 while the outcome-word count
falls with it. At 128 words the advantage is gone and the panel turns green. That is the project's
main result, made physical.

Then the four-split table (the same ~128-word boundary found four times), the by-decade bars (every
decade, 1950s–2020s), and the fine-tuned comparison: **85.9% reading the contaminated window versus
66.5% reading the facts, same model and same training.**

**② Explore a case** second — what the system actually produces. Facts, then element verdicts each
carrying the quote it relied on, then predicted statutes (1.9× baseline) and retrieved precedents
(7.4× control). 56 curated cases across eight decades, chosen to prefer ones with a
`NOT_SATISFIED` verdict since that is the interesting half of the element layer.

**③ Try your own facts** third — the audience can type a dispute and get real precedents back. This
is the part of the pipeline that genuinely works.

**④ Why the symbolic pipeline fails** last. The negative result is the least demo-friendly and the
most easily misread, so it goes where there is time to explain it rather than where it competes
with the finding.

## Presenting it

Two things to say out loud, because the slides cannot:

1. **Do not compare 66.5% to the published 78%.** They measure different things and ours is
   strictly harder. The gap *is* the result. Tab ① says this in a warning box for exactly this
   reason.
2. **The element layer is a good explanation layer and a bad feature layer.** It scores 0.521 for
   prediction, which is near chance — but it invents support only 0.62% of the time and agrees with
   itself 96.9%. "Here are the requirements your facts satisfy, with the quote for each" is
   checkable; a win/lose prediction never is.

## Known limits, worth stating if asked

- The 56 cases are curated, not a random sample — they are chosen to have enough facts and at least
  one interesting element verdict. The *numbers* on tab ① are from the full evaluation; the *cases*
  on tab ② are a showcase.
- Tab ③'s retrieval has no date filter, since there is no query judgment to pre-date. The evaluated
  pipeline asserts that constraint per result.
- Element verdicts have not been checked against a human (blocker B6). Reliability and fabrication
  are measured; **correctness is not.**

Verified with Streamlit's `AppTest`: the app runs with no exceptions, and the slider, the case
selector and the retrieval button were each exercised.
