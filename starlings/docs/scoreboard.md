# Offline scoreboard v1

The [bounded client collector](scoreboard-collector.md) supplies paired client TTFT/
throughput observations for GLY-245. Other collectors and actual native qualification
remain open.

GLY-270 implements the independent software portion of GLY-245. It reads supplied
records; it does not collect measurements, launch programs, qualify a backend or
publish results. M2 provider and native acceptance remain open.

```sh
cargo run --locked --offline -p starlings-cli -- scoreboard report examples/scoreboard-plan.json
```

The example prints all 22 planned cells as `not_measured`. Missing second
same-type machines, pinned working builds and workload approvals are explicit
gaps. A successful command means the input was valid, not that any comparison won.

## Input contract

The strict serde records in `crates/starlings-cli/src/scoreboard.rs` define the v1
JSON contract. Unknown fields and duplicate JSON keys are rejected. Every object
below lists all its fields; nullable values are written as JSON `null`.

| Object | Fields |
| --- | --- |
| Matrix | `version: 1`, `fixture: bool`, `programs: Programs`, `artifacts: Artifact[]`, `cells: Cell[]` |
| Programs | `exo: Program or null`, `glyd: Program or null` |
| Program | `repository: GitHub HTTPS owner/repository URL`, `source_commit: 40 lowercase hex characters`, `build: Digest`, `recipe: Digest` |
| Artifact | `path: relative file path`, `sha256: Digest` |
| Cell | `id: unique label`, `metric: Metric`, `matched: Matched or null`, `unavailable: reason or null`, `observations: Observation[]` |
| Matched | `model`, `profile`, `hardware`, `workload`, `definition`, `parameters`: each a Digest of an evidence file |
| Observation | `fixture: bool`, `id: UUID`, `system: exo or glyd`, `repeat: 1..32`, `program_digest: Digest`, `match_digest: Digest`, `unit: string`, `raw: Digest`, `outcome: Outcome` |
| Outcome | `status: measured, value: finite number` OR `status: failed, code: label` |

Digest is a lowercase SHA-256 hex string. Artifact SHA-256 covers exact file bytes.
Every digest reference must name a verified artifact. Paths are relative to the
matrix directory, resolve inside it and identify regular files. Duplicate paths
or artifact digests are rejected. No paths or commands from evidence are executed.
This is an offline analysis tool for a stable evidence directory, not a hardened
filesystem namespace or an authority that attests the truth of supplied receipts.

`build` names a receipt containing executable and dependency identities, platform,
compiler/runtime and build flags. `recipe` records exact steps, dependency lock
digests, patches and runtime options. No recipe is executed by this command.
Exo's repository must be `https://github.com/exo-explore/exo`.

Matched receipts must record the same checkpoint revision and complete asset
hashes, tokenizer/template, precision, context, sampling, prompt/output counts,
warmup/cache/EOS policy, clocks/measurement boundaries, hardware inventory and
network. Each system may have its own implementation/build receipt. Definition
receipts distinguish client timings from server timings, host from device memory,
and per-request from aggregate throughput. Hash matching proves supplied identity
and integrity; it cannot establish equal semantics, truthful measurements or the
best working configuration. Those require reviewed collection and native evidence.

Bindings use existing RFC 8785 canonical JSON SHA-256:

- `program_digest = hash([system, Program])`.
- `match_digest = hash([Cell.id, Cell.metric, Cell.matched])`.
- Report `matrix_digest = hash(Matrix)` after strict typed parsing.

The report supplies bindings for matched cells and supplied programs. Fill a cell
with verified receipts and zero observations first to obtain its binding without
making up a measurement. Mark a cell unavailable with a reason and no observations
when the workload cannot run. Omit neither a failed attempt nor an unsupported cell.

## Metrics and decision policy

All 18 metric kinds must have at least one planned cell. The example further
separates TTFT at 1,024/8,192/32,768 prompt tokens and concurrency 1/4/16. Its decode
target is 512 prompt/128 output tokens; prefill target is 8,192 tokens. Workload
receipts, not labels alone, establish these conditions before real comparisons.

| Metric | Unit | Better |
| --- | --- | --- |
| decode_tokens_per_second, prefill_tokens_per_second, concurrent_tokens_per_second | tokens_per_second | higher, speed rule |
| ttft_milliseconds, setup_milliseconds, recovery_milliseconds | milliseconds | lower, speed rule |
| scaling_ratio | ratio | higher, speed rule |
| peak_resident_bytes, cancel_retained_bytes | bytes | lower |
| largest_model_parameters, maximum_context_tokens, model_families_count | count | higher |
| quality_kl | kl_divergence | lower |
| quality_top1_fraction, api_pass_fraction | fraction | higher |
| quality_perplexity | perplexity | lower |
| setup_manual_steps | count | lower |
| reliability_successes20 | count | higher |

Policy `gly234-conservative-v1` requires at least five successful paired repeat IDs
from both systems in a cell. Duplicate run IDs within a cell, duplicate system/repeat
pairs, wrong units/bindings, nonfinite values or mismatched fixture markers reject
the record. Missing pairs yield `not_measured`. Any failed observation yields
`not_measured` with its failure ID/code preserved; successful subsets cannot win.

For every metric, the complete observed ranges must be strictly disjoint in the
better direction. Speed metrics also require median improvement of at least 10%:
higher candidates are at least 1.1 times baseline; lower candidates are at most
0.9 times baseline. Other metrics require strictly better disjoint ranges, without
a percentage margin. This deliberately conservative extension does not silently
treat overlap as a win. Medians average the two middle samples for even counts.
These are observed ranges, not statistical confidence intervals.
Each summary also reports nearest-rank p95/p99, left null below20/100 samples
respectively. Those reporting floors are not confidence guarantees and do not
alter the median, range, pairing, failure or win rules.

`won` means this supplied cell meets the rule. `lost` means it does not meet the
rule, including ties and overlap; it does not necessarily mean slower. No overall
winner is inferred. `qualification` is always `not_assessed`. Synthetic matrices
are refused by default; only a `test-backend` build accepts `--allow-fixtures`, and
fixture comparisons have status `fixture`, never `won` or `lost`. Per-observation
fixture flags must agree with the matrix. Relabeling fabricated evidence is not
prevented by hashes; evidence review remains required.

Limits: 1 MiB JSON plus the shared parser's depth/node limits, 64 cells, 512 total
observations, 512 artifacts, 16 MiB per artifact and 64 MiB total artifact bytes.
Large raw runs must use reviewed bounded summaries with immutable full-log
references; this command does not verify external references. Labels are bounded
ASCII; values are between zero and 10^15, counts/bytes integral, fractions at most
one, reliability at most 20, speed values positive and perplexity at least one.
Validation errors exit nonzero and print a static error code without evidence data.

## Exo candidate and collection boundary

The research candidate is commit
`bc412e2230f395e652c23426a5eb88dfa515769b` (2026-10-13). The immutable source
file hashes and links are in `benchmarks/upstream/exo-bc412e2.json`. This candidate
has not been built or run here; there is no approved best-working recipe yet.
Its [package metadata](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/pyproject.toml)
requires Python 3.13 and includes pinned MLX sources/wheels and mlx-lm dependencies;
the [lockfile](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/uv.lock)
must be preserved with any eventual build receipt. The
[README](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/README.md)
lists CUDA installation extras while
[PLATFORMS](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/PLATFORMS.md)
still lists Linux as planned. This documentation discrepancy establishes neither
working support nor lack of support; qualification must test the selected build.

Exo's [benchmark methodology](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/bench/METHODOLOGY.md)
uses a dedicated benchmark endpoint with prefix cache disabled by default,
EOS suppressed for fixed output and raw output. Its server decode rate uses
`(completion_tokens - 1) / (last_token_time - first_token_time)`. Client elapsed
time includes network and other phases. Never substitute client HTTP duration
for that server metric or count SSE chunks as tokens. The existing
`spec/benchmark-run.schema.json` client-monotonic record may be referenced as an
artifact where appropriate; it does not supply missing server phase measurements.

Actual collection still needs pinned working Exo/Glyd builds, approved 2026 model
assets/workloads, matched definitions and reviewed raw logs. Two-node comparisons
need two qualified same-type machines. Mixed hardware, Raspberry Pi and mobile
qualification remain parked. This document includes no copied Exo implementation.
