# Offline planner calibration records

GLY-181's `plan calibrate INPUT_JSON` compares the existing planner's estimates
with supplied observations. It reads files only: no benchmark, model, process,
device, network, reservation or native admission runs. It is separate from
[client benchmark inspection](benchmark-inspection.md) and the
[Exo scoreboard](scoreboard.md); it does not infer missing server measurements
from client durations, token counts or SSE chunks.

The report always has `qualification: not_assessed` and `admission: false`.
Predictions remain estimates even when file hashes and record bindings validate.
Nothing is refitted, approved, admitted or classified as an Exo win. Actual native
collection, review of prediction thresholds and measured calibration remain open.

## Input and identity

The strict internal Rust/JSON records are in `starlings_cli::calibration`:

| Record | Fields |
| --- | --- |
| Input | version1, fixture, placement (existing PreviewInput), candidate_index0..15, source (nullable EvidenceSource), artifacts (existing Artifact list), observations |
| EvidenceSource | execution_plan, producer (both SHA-256 digests of supplied evidence files) |
| Observation | id (invocation UUID), fixture, metric, binding, raw (artifact SHA-256), outcome |
| Outcome | status observed with elapsed_us (canonical decimal u64), or status failed with code (1..64 uppercase letters/digits/underscores) |

The existing strict parser enforces1MiB/depth32/16,384 nodes, duplicate-key and
unknown-field refusal. Existing planner/schema/search limits apply. There are
at most512 observations, unique by `(id, metric)`; an attempt may report several
different metrics. The artifact reader retains its512-file,16MiB-per-file,
64MiB-total limits and relative contained regular-file/hash checks. These are
stable offline evidence-directory checks, not a protected namespace against
concurrent hostile filesystem changes. No paths, record contents or raw failure
messages are echoed; hashes, supplied timings and bounded failure codes are output.

The command recomputes the planner preview, selects its candidate and records:

- `placement_digest = hash(PreviewInput)`, binding complete manifest, full
  ProfileKeys/runtime/workload/device/pool identities, snapshots, links and policy;
- `candidate_digest = hash(Candidate)`, binding selected nodes, layers, profile
  digests, physical pool budgets, edges and predicted durations;
- metric `binding = hash(["glyd-plan-calibration-v1", fixture, placement_digest,
  candidate_digest, source, metric])`;
- `input_digest = hash(Input)`, including supplied artifacts and all observations.

Hashes use the existing RFC8785/SHA-256 canonicalizer. For any observation,
`source` is required and both receipts must be verified artifacts. The execution
plan receipt describes the exact attempted runtime plan/stages and any refusal;
the producer receipt describes the actual Compute/collector/backend code/build,
runtime settings, clocks and instrumentation that produced the observations.
Each observation also names verified raw evidence. Receipt bodies are not parsed
or attested here: matching hashes bind supplied identity, not truthful collection,
correct provider mapping, semantic equivalence or a native qualification.

Start with zero observations to obtain bindings. An empty template may have
source null; adding or changing source receipts changes every binding, so obtain
fresh bindings before attaching observations. Changing runtime, workload, input
clocks, candidate or metric also invalidates old bindings. The fixture flag must
agree throughout. Default CLI refuses fixtures; only a test-backend build accepts
`--allow-fixtures` and still prints fixture true/no qualification.

## Timing definitions and errors

All values below are microseconds at the **exact declared ProfileWorkload**.
They compare existing estimator quantities, not arbitrary HTTP/end-to-end timings:

| Metric | Predictor / required observation boundary |
| --- | --- |
| load_serial | Sum of declared stage load times; matching serial load/preparation scope, without overlapping stages or adding download/install/setup work |
| prefill_path | Sum of stage prefill/copy and directed-link costs for the same declared workload and prompt/chunk/session policy |
| decode_path_per_token | Sum of stage decode/copy and directed-link costs for one sequential token path at the same workload/state policy |
| bottleneck_per_token | Maximum stage decode/copy plus its outgoing-link cost; matching component-cycle boundary, not measured whole-cluster throughput |

Native collectors must establish these boundaries and their provenance before
supplying values. A missing/unsupported phase has no observations, not a zero
measurement. All four metric rows are emitted even if empty. Failed attempts
retain their IDs, raw hashes and codes with null deviation; they are never folded
into success summaries. A workload without an eligible predictor candidate is
refused, not assigned a made-up prediction; its original benchmark failure record
remains separate evidence.

For a supplied observation `a` and estimate `p`, output the exact u64 magnitude
`abs(a-p)` plus whether the observation is above the estimate. The optional finite
floating ratio is `abs(a-p)/p`, with estimate as denominator; it is approximate,
not a confidence bound. If p=0 the ratio is null, including when a=0, while the
exact magnitude remains available. No percentage threshold or pass/fail rule is
chosen before real native data and review. No durations are synthesized for failed
or absent observations, and no old profile is automatically promoted or refreshed.
