# extract_claims.v1

You are reading an excerpt from a judgment of the Supreme Court of India in a land or property
dispute. The court's reasoning and its order have been removed; what remains is the facts, the
pleadings, and the parties' arguments.

Identify the **claims** and the **defences** in the dispute.

A **claim** is a legal demand a party makes: what relief they ask for and on what legal footing.
A **defence** is a ground the opposing party raises to defeat it.

Reply with ONLY a JSON object, no prose and no code fences:

```
{
  "claims": [
    {
      "text": "<the claim in one sentence, as a legal demand>",
      "quote": "<verbatim span from the excerpt supporting it>",
      "raised_by": "plaintiff" | "defendant" | "unknown",
      "relief_sought": "<the relief asked for, e.g. declaration of title, possession, partition, specific performance, compensation, injunction, cancellation of deed, redemption>",
      "statutory_hook": "<the Act and section pleaded, e.g. 'Specific Relief Act s.16(c)', or null>",
      "family": "<the best-fitting family from the list below, or NEW:<proposal>>"
    }
  ],
  "defences": [
    {
      "text": "<the defence in one sentence>",
      "quote": "<verbatim span from the excerpt supporting it>",
      "raised_by": "plaintiff" | "defendant" | "unknown",
      "family": "<one of: Title/Tenure, Public Interest/Planning Policy, Statutory/Regulatory Subservience, Consent/Compliance, Procedural/Forum, Equitable, or NEW:<proposal>>"
    }
  ]
}
```

## Claim families

Choose the family the relief chiefly turns on:

- `Title` — declaration of title, ownership, competing title, cloud on title
- `Possession` — protection or restoration of possession, injunction, dispossession, trespass
- `LeaseTenancy` — lease, sub-lease, tenancy, allotment, occupancy, renewal, NOC
- `Redevelopment` — redevelopment or planning schemes, developer appointment, consent, FSI/TDR
- `ConstitutionalDeprivation` — deprivation of property without authority of law, Article 300A
- `UltraVires` — executive or delegated action exceeding statutory power
- `CooperativeSociety` — society governance, member consent, amalgamation, associational autonomy
- `AdversePossession` — long or hostile possession, limitation, delay, laches
- `Partition` — share and separate possession of joint family or co-owned property
- `SpecificPerformance` — enforcement of an agreement to sell
- `Cancellation` — cancellation or rectification of a deed or instrument
- `MortgageRedemption` — redemption, foreclosure, enforcement of security
- `Succession` — inheritance, wills, heirship, mutation on death
- `Benami` — property held in another's name while consideration flowed from the real owner
- `LandAcquisition` — validity of acquisition, notification, possession taking, compensation

Use `NEW:<proposal>` only when none of these fits. A dispute may raise several claims in
different families; list each.

## The quote is the evidence

`quote` must be copied character for character from the excerpt. No paraphrase, no joined
fragments, no ellipsis. A claim whose quote does not appear in the excerpt will be discarded.

Return empty lists if the excerpt states no claim or defence.
