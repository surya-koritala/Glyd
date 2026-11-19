# Offline benchmark record inspection

GLY-290 implements the record-validation portion of GLY-178. It uses the existing
`BenchmarkRun` draft, shared strict protocol parser and scoreboard artifact reader.
It reads files only; it starts no client, daemon, backend or model.

```sh
starlings benchmark inspect /path/to/evidence/run.json
```

A successful inspection emits the exact record SHA-256, run ID/status, outcome
counts, failed request IDs/codes and the number of verified artifacts. It always
reports `qualification: "not_assessed"`. A valid failed run remains `failed`; a
successful inspection is not successful inference or permission to run anything.
Notes, record paths and raw artifact contents are never echoed. Errors are static
codes with nonzero exit status.

`latency` groups supplied client `duration_ms` and `ttft_ms` by every outcome,
including timeout, cancellation, OOM and error. Each nonempty distribution contains
`samples`, `minimum`, `median`, `maximum`, `p95` and `p99`; absent distributions
are null, not zero-latency observations. Duration includes all requests of that
outcome; TTFT includes only requests with a supplied TTFT. Missing TTFTs therefore
reduce its sample count. A failed run remains failed, even if some requests succeed.

Percentiles are observed nearest-rank values at index `ceil(p × n) - 1`. p95 is
null below20 samples and p99 below100 samples; these explicit reporting floors
provide at least one observation in the corresponding upper5%/1%, not statistical
confidence or a population-tail guarantee. Median averages the middle pair for
even counts using the standard finite midpoint. These are summaries of the
supplied per-request timings, not server-phase/token-rate measurements, a native
qualification, or a baseline comparison. The existing shared scoreboard summary
uses the same sample/range/median/percentile output without changing its win rule.

The first typed implementation finalizes the previously unimplemented
[`BenchmarkRun` v1 draft](../spec/benchmark-run.schema.json). It now requires an
explicit `fixture` boolean. No existing BenchmarkRun collector/record was found
in this repository. Earlier native evidence uses different formats and remains
unchanged; this command does not silently convert or qualify it. Production CLI
refuses fixtures. A `test-backend` CLI permits `--allow-fixtures`; its receipts
retain `fixture: true` and still confer no qualification.

Schema validation asserts date-time formats, required/unknown fields, UUIDs,
lowercase SHA-256 identities and numeric bounds. Semantic checks reject placeholder
commits/builds/digests, self-baselines, repeated build/request/pool identities,
contradictory measured/failed outcomes, missing failure codes and unsafe paths.
Failure codes are bounded uppercase letters/digits/underscores, not error text.
A `measured` run needs nonempty successful requests and peak pool observations.
A `failed` run can retain zero requests/pools if collection failed before them;
it may also contain successful requests when later cleanup/collection failed.

Timings are finite nonnegative client-monotonic milliseconds. TTFT cannot exceed
request duration and is absent when no output token was observed. An empty
inter-token array means those timings were not supplied, including nonstreaming
responses. If supplied, it requires TTFT and exactly `output_tokens - 1` intervals;
TTFT plus intervals cannot exceed duration beyond accumulated floating-point
rounding (`duration × f64::EPSILON × (interval_count + 2)`). These fields require
actual token timing/count provenance; SSE event counts are not token counts.

Hardware inventory and raw-log paths resolve relative to the resolved record's
directory. Both must be distinct regular artifacts within that directory and
match their supplied hashes. The shared reader rejects absolute/dot/escaping
paths, duplicate files/hashes and mismatches. Limits are the shared 1 MiB
JSON/depth/node caps, 16 MiB per artifact and 64 MiB total artifact budget; the two
referenced files therefore consume at most 32 MiB. This is for stable evidence
directories, not a protected namespace against concurrent hostile filesystem edits.

The inspector does not calculate aggregate throughput: v1 has request durations
but no complete run measurement window or per-request start offsets. Summing
concurrent durations would give a different denominator. It does not compare
baselines, attest raw measurements, validate hardware inventory contents, prove
tokenization/precision, inspect executable builds or qualify memory/drain policy.
Hashes establish supplied file identity, not truthful collection. GLY-238 provider
delivery, approved hardware/workloads, real direct/proxy/replica measurements and
reviewed backend evidence remain M2 gates.
