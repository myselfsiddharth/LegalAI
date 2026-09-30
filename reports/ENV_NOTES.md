# ENV_NOTES — Voyager API configuration (§3.1)

Variable names present in `.env` (names only, never values):

| Name | Role |
|---|---|
| `VOYAGER_API_KEY` | API key for the ASU Voyager endpoint |

**Not in `.env`, hard-coded in `scripts/ode_lib.py` today** — must move to config per §3.2:

- Base URL: `https://openai.rc.asu.edu/v1` (OpenAI-compatible; use the `openai` SDK with `base_url`)
- Default generation model: `llama4-scout-17b`
- Embedding model: `qwen3-embedding-8b`, 4096-dim (only `qwen3-embedding-4b` / `-8b` exist on this endpoint)

Verified behaviour (carried over from prior work, re-confirm on first smoke test):

- No ASU VPN required; the host resolves to public Cloudflare IPs.
- `/v1/models` returns no capability metadata, so an embedder can only be identified by
  probing `/v1/embeddings`.
- The endpoint is **token**-throughput-limited, not request-limited: larger batches buy
  almost nothing (0.95 s/doc vs 1.06 s/doc).
- Use a 90 s client timeout. The SDK default (600 s x 3 attempts) has silently frozen a
  batch for 20+ minutes.

**Gap vs §3.2:** the current client has **no cache, no retry, no structured-output
validation, and no call logging**. `grep -n "cache" scripts/ode_lib.py` returns nothing
relevant. This is net-new work and everything else depends on it.
