# Bounded client scoreboard collector

GLY-245 adds collection to the existing [offline scoreboard](scoreboard.md).
It sends requests to two explicitly configured, already-running loopback services.
It never starts a backend, downloads a model, retries a request or proves device
cleanup. Real workloads still require explicit device/budget approval.

```sh
starlings scoreboard collect matrix.json collection-plan.json collected.json
starlings scoreboard report collected.json
```

The first command creates `collected.json` and `collected.raw.jsonl` beside the
input matrix and prints the regenerated report. Both output files must be new;
existing evidence is never overwritten. Existing artifact paths remain relative
to the same evidence directory. Preserve the input matrix, plan, immutable receipts
and private request input for reproduction. The collector does not copy request
inputs into its outputs. Only a `test-backend` build accepts synthetic inputs with
an additional `--allow-fixtures`; fixture comparisons never become wins.

## Plan and matching

Strict JSON rejects unknown fields and duplicate keys. The collection plan is:

```json
{
  "version": 1,
  "fixture": true,
  "repeats": 5,
  "timeout_ms": 1000,
  "budget_ms": 10000,
  "exo": {"address": "127.0.0.1:9743", "model": "synthetic-exo"},
  "glyd": {"address": "127.0.0.1:9740", "model": "synthetic-glyd"},
  "cells": [{"id": "ttft_1024", "request_path": "private/request.json", "cancel_after_content": false}]
}
```

This is a synthetic configuration illustration, not a runnable native benchmark.
Use cell IDs actually present in the matrix. Each selected cell needs verified
program/build/recipe and matched model/profile/hardware/workload/definition/
parameters receipts, and zero existing observations. Other planned cells and
prior observations remain intact. All matrix artifacts are verified before I/O
and again before final publication. Operate on a stable evidence directory;
these checks do not defend against a malicious same-user filesystem race.

The selected cell's workload artifact is a strict redacted JSON receipt:
`version: 1`, `request_sha256: Digest`, `expected_prompt_tokens: 1..1048576`,
`concurrency: 1..16`. The request file must reside inside the matrix directory,
match that exact-byte SHA-256, and pass the existing strict ChatRequest contract.
Specify temperature, top_p, seed and stream explicitly. The shared body is sent
to both systems, changing only model to the endpoint's explicit alias and adding
`stream_options.include_usage=true` for SSE. Raw records retain each actual wire
body's hash. Same asset/template/precision/sampling semantics and best-working
builds still require review; matching labels or hashes do not establish them.

Optional bearer credentials come only from `STARLINGS_SCOREBOARD_EXO_TOKEN` and
`STARLINGS_SCOREBOARD_GLYD_TOKEN`. Never place credentials in plan/evidence files or
command arguments. Endpoints are numeric loopback addresses with nonzero ports;
proxies and redirects are disabled. POST target is `/v1/chat/completions`.

## Supported boundaries

The definition artifact is strict JSON with `version: 1`, `metric` and `boundary`:

| metric | boundary | Conditions |
| --- | --- | --- |
| ttft_milliseconds | client_first_nonempty_content_or_reasoning_delta | SSE, concurrency 1; request start through parsing first nonempty content/reasoning delta; role-only chunks excluded |
| concurrent_tokens_per_second | client_batch_usage_tokens_over_wall_seconds | Sum of successful final completion usage / batch wall seconds, including client setup, thread launch, connection, transfer and parse time; JSON or SSE |

JSON duration is recorded but never substituted for TTFT. SSE chunks are not token
IDs and never produce token counts or inter-token latency. Client TTFT is not
server prefill time. Server decode/prefill rates, native/device peak memory,
setup, quality, reliability, capacity and multi-node metrics remain uncollected;
keep their explicit `not_measured` cells. The legacy BenchmarkRun contract requires
qualified memory for measured runs; this collector instead uses GLY-270
observations plus its redacted client receipt and does not fabricate that memory.

Success requires one stable response ID/model/object, a supported assistant
message/delta, a stop/length finish and coherent usage matching the expected
prompt count with positive completion count no greater than max_tokens. SSE must
include usage and `[DONE]`. Wrong IDs/models, missing/malformed usage, duplicate
keys, unsupported events, incomplete streams and non-200 statuses are failures.
For the audited Exo source pin `bc412e2230f395e652c23426a5eb88dfa515769b` only,
stream records must carry `object: "chat.completion"`: its
[response type](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/api/types/api.py#L153)
is also used by its [stream adapter](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/api/adapters/chat_completions.py#L203).
The collector selects this one spelling from the matrix's existing system/source
identity. Glyd and other Exo pins still require `chat.completion.chunk` for SSE;
JSON always requires `chat.completion`. There is no auto-detection, new user flag
or broader response fallback. Source hashes are retained in the upstream pin
record. This compatibility rule is source-audited and synthetic-tested, not proof
that the pinned build/model works or is the best-working configuration.
No server error body, prompt, generated content, returned ID, alias or credential
is persisted by the collector. Error codes are static; HTTP status is numeric.

## Trials, limits and failure retention

Every cell runs 5..8 paired repeats, alternating Exo/Glyd then Glyd/Exo. Within a
batch, scoped threads issue the specified concurrency; thread creation is part
of batch timing. There is no synchronized server arrival or warmup. Separate
native preparation must establish warmup/cache/EOS policy and system isolation;
the collector cannot prevent two resident services competing for memory.

Limits: 16 selected cells; 256 total planned requests; existing 512-observation/
artifact and 64-cell matrix limits; 30-second maximum individual request deadline,
shortened by a shared network budget of at most five minutes. File preparation,
local parsing and evidence synchronization add bounded work outside socket
waiting; this is not an OS watchdog or guaranteed wall-clock termination under
host suspension. HTTP headers are limited to 16 KiB, JSON to 1 MiB, SSE lines/events
to 64 KiB and each stream to 16 MiB; existing strict JSON depth/node limits apply.

Any failure stops all later submissions, including the paired other system.
Already-running requests in that batch retain their own bounded outcomes. Remaining
planned slots are `NOT_ATTEMPTED_AFTER_FAILURE`; an expired budget before submission
is `BUDGET_EXHAUSTED`. No success subset can win. HTTP disconnect after the first
content event is an explicit cancellation experiment and records `CANCELLED`,
never cleanup success; arrange separate approved cleanup before another run.
Header timeout is `TIMEOUT`; response-body read failure is `BODY_READ` and is not
claimed to identify a timeout versus another I/O failure.

Each raw JSONL trial records fixture marking, plan/collector-executable hashes,
UTC start label, matched/program bindings, system/repeat/order, batch elapsed and
per-request wire hash, attempted flag, status, static outcome, usage counts,
first-content time and duration. UTC is for correlation; durations use Instant.
Each completed trial is synchronized before proceeding. The final matrix references
the complete raw file hash. A process/storage failure may leave a partial raw file
and empty/unfinalized matrix; preserve them and do not treat them as a complete run.
A valid report command returning zero establishes record consistency, not successful
qualification. Inspect the report's failures and `not_measured` rows.
