# treatment_direction.v1

A passage from an Indian Supreme Court judgment says that some decision was overruled, disapproved,
doubted or declared wrong. You are told the name of one case, the TARGET.

Answer one question: **is the TARGET the decision that was overruled / disapproved / doubted?**

- `YES` -- the TARGET is the decision that lost its authority ("TARGET was overruled", "the view
  taken in TARGET is not correct", "TARGET ... overruled." in a list of cases).
- `NO` -- the TARGET is the decision that DID the overruling, or is merely mentioned ("X was
  overruled in TARGET", "X has been overruled by TARGET", "in TARGET the Court declared the Act
  unconstitutional"), or the passage is about something else.

Reply with ONLY a JSON object: {"answer": "YES"} or {"answer": "NO"}
