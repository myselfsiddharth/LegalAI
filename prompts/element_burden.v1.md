# element_burden.v1

You are annotating the **burden of proof** for one element of one claim in Indian land and
property law, for a research knowledge base.

You will be given: the claim, its legal issue, the element, the element's definition, and any
burden guidance the source ontology already states for that claim.

Reply with ONLY a JSON object, no prose and no code fences:

```
{
  "burden_on": "claimant" | "respondent" | "authority",
  "burden_standard": "prima_facie" | "balance_of_probabilities" | "statutory_presumption" | "clear_proof_required" | "unknown",
  "burden_shifts_when": "<the triggering fact or event, one short phrase, or null>",
  "reasoning": "<one sentence citing the doctrine or statutory provision you are relying on>",
  "confidence": 0.0-1.0
}
```

## Rules

**Follow the ontology's own guidance where it is given.** If the claim states "Default
burden_on: purchaser", the elements of that claim take that default unless a specific element
plainly reverses it.

**`burden_on` is who must prove the element, not who benefits from it.** In adverse possession the
person *claiming* adverse possession must prove hostility, even though the registered owner
benefits if they fail.

**Use `clear_proof_required` sparingly** — for elements Indian courts treat as demanding more than
the ordinary civil standard: adverse possession, benami, fraud, forgery, and the due execution of
a will where suspicious circumstances are alleged.

**`statutory_presumption`** is for elements where a statute or a registered instrument supplies a
presumption that the other side must rebut — for example the presumption of correctness attaching
to a registered document, or a statutory presumption under the Evidence Act.

**`burden_shifts_when`** names the fact that moves the burden, not the consequence. "Registered
deed produced" or "execution of the agreement admitted", not "the defendant must then prove
fraud".

**Set `confidence` below 0.6 when the allocation is genuinely contested in Indian case law,** or
when it depends on which relief is sought. Do not present a settled-sounding answer to an unsettled
question.
