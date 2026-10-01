"""The single door to the LLM (§3.2). Everything else calls this module.

Why it exists rather than each script holding its own `OpenAI(...)`: the prior iteration
of this project did that, and paid for it three times over.

  * No cache, so every re-scored experiment re-spent the API budget and no number was
    reproducible without another paid run. §0.7 makes caching a rule; this is where it lives.
  * No retries, so a transient `APIConnectionError` inside a per-item loop became a
    *finding*: one script reported "0 elements with support >= 2" when in fact 150 of 253
    sentences were never asked. A dropped call and an empty answer are different events and
    this module keeps them distinguishable -- see `LLMResult.ok`.
  * The SDK default timeout (600s x 3 attempts) once froze a batch for 20+ minutes with no
    output. We use 90s.

Cache key is sha256 over (model, messages, and every parameter that can change the output).
Temperature defaults to 0, so a cache hit is a genuine repeat rather than a resampling.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src import paths

# --- endpoint -----------------------------------------------------------------
# ASU Voyager is OpenAI-compatible. No VPN needed (the host resolves to public
# Cloudflare IPs). `/v1/models` carries no capability metadata, so an embedding model can
# only be confirmed by probing `/v1/embeddings`; of the names that endpoint accepts,
# qwen3-embedding-8b is the strongest available.
BASE_URL = os.environ.get("VOYAGER_BASE_URL", "https://openai.rc.asu.edu/v1")
DEFAULT_MODEL = os.environ.get("VOYAGER_MODEL", "llama4-scout-17b")
DEFAULT_EMBED_MODEL = os.environ.get("VOYAGER_EMBED_MODEL", "qwen3-embedding-8b")
EMBED_DIM = 4096

# NOT the SDK default of 600 (see module docstring), but configurable, because the right value
# depends on the request. A 2,500-token extraction chunk answers in ~20s and 90s is generous. A
# batched canonicalisation call carrying 24 facts answers in ~40s idle and ~80s under 96-way
# concurrency, so 90s would time out work that is progressing fine -- and each timeout costs four
# retries with quadratic backoff, which is far worse than waiting.
TIMEOUT_S = float(os.environ.get("VOYAGER_TIMEOUT_S", "90"))
MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 2.0

CACHE_DB = paths.CACHE / "llm_cache.sqlite"
CALL_LOG = paths.CACHE / "llm_calls.jsonl"


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader. Never logs or prints a value."""
    path = path or paths.ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class LLMResult:
    """`ok=False` means we never got an answer (network, timeout, exhausted retries).
    An empty or unparseable `text` with `ok=True` means the model genuinely said nothing.
    Callers that aggregate over many items MUST report these two separately -- conflating
    them turns an outage into a substantive finding."""
    text: str
    ok: bool = True
    error: str | None = None
    model: str = ""
    cached: bool = False
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: int = 1
    parsed: Any = None            # set when a json_schema/pydantic model was requested

    @property
    def dropped(self) -> bool:
        return not self.ok


# --- cache --------------------------------------------------------------------
class _Cache:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("""CREATE TABLE IF NOT EXISTS calls (
            key TEXT PRIMARY KEY, kind TEXT, model TEXT, response TEXT,
            prompt_tokens INT, completion_tokens INT, created REAL)""")
        self._conn.commit()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response, model, prompt_tokens, completion_tokens FROM calls WHERE key=?",
                (key,)).fetchone()
        if not row:
            return None
        return {"response": row[0], "model": row[1],
                "prompt_tokens": row[2], "completion_tokens": row[3]}

    def put(self, key: str, kind: str, model: str, response: str, pt: int, ct: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?,?)",
                (key, kind, model, response, pt, ct, time.time()))
            self._conn.commit()

    def stats(self) -> dict:
        with self._lock:
            n, pt, ct = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0), "
                "COALESCE(SUM(completion_tokens),0) FROM calls").fetchone()
        return {"entries": n, "prompt_tokens": pt, "completion_tokens": ct}


_cache: _Cache | None = None
_client = None
_counters = {"calls": 0, "cache_hits": 0, "dropped": 0,
             "prompt_tokens": 0, "completion_tokens": 0}


def _get_cache() -> _Cache:
    global _cache
    if _cache is None:
        _cache = _Cache(CACHE_DB)
    return _cache


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        load_dotenv()
        key = os.environ.get("VOYAGER_API_KEY")
        if not key:
            raise RuntimeError(
                "VOYAGER_API_KEY is not set. Create .env at the repo root with "
                "VOYAGER_API_KEY=<key> (get one at voyager.rc.asu.edu -> LLM Access). "
                "Never commit .env.")
        _client = OpenAI(api_key=key, base_url=BASE_URL, timeout=TIMEOUT_S, max_retries=0)
    return _client


def _key(kind: str, model: str, payload: Any, params: dict) -> str:
    blob = json.dumps({"kind": kind, "model": model, "payload": payload,
                       "params": {k: params[k] for k in sorted(params)}},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _log(rec: dict) -> None:
    rec["ts"] = time.time()
    with CALL_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


# --- prompts ------------------------------------------------------------------
def load_prompt(prompt_id: str) -> str:
    """Prompts are versioned files under prompts/, never inline strings (§3.2), so that
    every reported number can name the exact prompt that produced it.
    `prompt_id` is a filename stem, e.g. 'extract_facts.v1'."""
    p = paths.PROMPTS / f"{prompt_id}.md"
    if not p.exists():
        raise FileNotFoundError(f"no prompt file {p} (prompt_id={prompt_id!r})")
    return p.read_text()


# --- completion ---------------------------------------------------------------
_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")


def strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = _FENCE.sub("", t)
        # a trailing fence may survive if the model emitted no trailing newline
        t = re.sub(r"```\s*$", "", t).strip()
    return t


def extract_json(text: str) -> Any:
    """Parse JSON out of a model reply, tolerating fences and surrounding prose."""
    t = strip_fences(text)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = t.find(opener), t.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object/array found in reply")


def complete(messages: Sequence[dict],
             model: str | None = None,
             temperature: float = 0.0,
             max_tokens: int = 2048,
             json_schema: type | None = None,
             prompt_id: str | None = None,
             seed: int | None = 0,
             use_cache: bool = True,
             _retry_note: str | None = None) -> LLMResult:
    """One chat completion, cached and retried.

    `json_schema` is a Pydantic model. When given, the reply is parsed and validated; on a
    validation failure we retry ONCE with the error appended to the conversation (§3.2),
    then give up and return ok=True with parsed=None -- a model that cannot produce the
    schema is a finding, not an outage.
    """
    model = model or DEFAULT_MODEL
    params = {"temperature": temperature, "max_tokens": max_tokens, "seed": seed}
    msgs = list(messages)
    if _retry_note:
        msgs = msgs + [{"role": "user", "content": _retry_note}]

    key = _key("chat", model, msgs, params)
    if use_cache:
        hit = _get_cache().get(key)
        if hit is not None:
            _counters["cache_hits"] += 1
            res = LLMResult(text=hit["response"], model=model, cached=True,
                            prompt_tokens=hit["prompt_tokens"],
                            completion_tokens=hit["completion_tokens"])
            if json_schema is not None:
                res.parsed = _validate(res.text, json_schema)
            return res

    client = _get_client()
    last_err = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        t0 = time.time()
        try:
            kwargs = dict(model=model, messages=msgs, temperature=temperature,
                          max_tokens=max_tokens)
            if seed is not None:
                kwargs["seed"] = seed
            resp = client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "")
            u = getattr(resp, "usage", None)
            pt = getattr(u, "prompt_tokens", 0) or 0
            ct = getattr(u, "completion_tokens", 0) or 0
            latency = time.time() - t0

            _counters["calls"] += 1
            _counters["prompt_tokens"] += pt
            _counters["completion_tokens"] += ct
            if use_cache:
                _get_cache().put(key, "chat", model, text, pt, ct)
            _log({"kind": "chat", "model": model, "prompt_id": prompt_id, "cached": False,
                  "latency_s": round(latency, 2), "prompt_tokens": pt,
                  "completion_tokens": ct, "attempts": attempt, "ok": True})

            res = LLMResult(text=text, model=model, latency_s=latency,
                            prompt_tokens=pt, completion_tokens=ct, attempts=attempt)
            if json_schema is not None:
                res.parsed = _validate(text, json_schema)
                if res.parsed is None and _retry_note is None:
                    note = ("Your previous reply did not parse as the required JSON. "
                            "Reply with ONLY the JSON value, no prose and no code fences.")
                    return complete(messages, model=model, temperature=temperature,
                                    max_tokens=max_tokens, json_schema=json_schema,
                                    prompt_id=prompt_id, seed=seed, use_cache=use_cache,
                                    _retry_note=note)
            return res

        except Exception as e:                                  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_BASE_S * attempt ** 2)      # quadratic backoff

    _counters["dropped"] += 1
    _log({"kind": "chat", "model": model, "prompt_id": prompt_id, "ok": False,
          "error": last_err, "attempts": MAX_ATTEMPTS})
    return LLMResult(text="", ok=False, error=last_err, model=model, attempts=MAX_ATTEMPTS)


def _validate(text: str, schema: type):
    try:
        data = extract_json(text)
    except ValueError:
        return None
    try:
        if hasattr(schema, "model_validate"):
            return schema.model_validate(data)
        return schema(**data)
    except Exception:                                           # noqa: BLE001
        return None


# --- embeddings ---------------------------------------------------------------
def embed(texts: Sequence[str], model: str | None = None, batch_size: int = 32,
          use_cache: bool = True, verbose: bool = False, workers: int = 24):
    """Embed texts, cached per text. Returns an (n, EMBED_DIM) float32 array.

    Batches are issued CONCURRENTLY. They used to run one after another, which made embedding the
    bottleneck of canonicalisation: 65,577 texts at 32 per request is ~2,050 sequential round trips,
    measured at 10.3 texts/s and 77 minutes, while the chat path alongside it was sustaining 96
    concurrent requests. The endpoint queues rather than refuses (see WORKLOG on the concurrency
    measurement), so the fix is the same here as there.

    A row of zeros means that text's embedding was never obtained -- callers must check, not assume.
    """
    import numpy as np

    model = model or DEFAULT_EMBED_MODEL
    out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    pending: list[int] = []

    for i, t in enumerate(texts):
        if use_cache:
            hit = _get_cache().get(_key("embed", model, t, {}))
            if hit is not None:
                _counters["cache_hits"] += 1
                out[i] = np.array(json.loads(hit["response"]), dtype=np.float32)
                continue
        pending.append(i)
    if not pending:
        return out

    client = _get_client()
    batches = [pending[s:s + batch_size] for s in range(0, len(pending), batch_size)]

    def run(idx: list[int]):
        batch = [texts[i] for i in idx]
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                return idx, [d.embedding for d in resp.data]
            except Exception as e:                              # noqa: BLE001
                if attempt == MAX_ATTEMPTS:
                    _log({"kind": "embed", "model": model, "ok": False,
                          "error": f"{type(e).__name__}: {e}", "n": len(idx)})
                    return idx, None
                time.sleep(BACKOFF_BASE_S * attempt ** 2)
        return idx, None

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, vecs in pool.map(run, batches):
            if vecs is None:
                _counters["dropped"] += len(idx)                # rows stay zero, by design
                continue
            for i, v in zip(idx, vecs):
                out[i] = np.array(v, dtype=np.float32)
                if use_cache:
                    _get_cache().put(_key("embed", model, texts[i], {}), "embed", model,
                                     json.dumps(v), 0, 0)
            _counters["calls"] += 1
            done += len(idx)
            # §14 asks every experiment to report token usage. Successful embed batches were not
            # being logged at all -- only failures were -- so embedding cost was invisible in
            # llm_calls.jsonl even though it is a large share of the total.
            _log({"kind": "embed", "model": model, "ok": True, "n": len(idx)})
            if verbose and done % (batch_size * 20) < batch_size:
                print(f"  embedded {done}/{len(pending)}", flush=True)
    return out


# --- accounting ---------------------------------------------------------------
def usage() -> dict:
    """Per-process counters plus the persistent cache totals. §14 asks every experiment to
    report token usage and cache hit rate; call this at the end of a run."""
    total = _counters["calls"] + _counters["cache_hits"]
    return {**_counters,
            "cache_hit_rate": round(_counters["cache_hits"] / total, 4) if total else 0.0,
            "cache": _get_cache().stats()}


def reset_counters() -> None:
    for k in _counters:
        _counters[k] = 0
