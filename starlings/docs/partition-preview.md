# Advisory contiguous partition preview

`starlings plan preview INPUT_JSON` reads one bounded local JSON file and
prints candidate costs and refusal counts. It never contacts a device, loads a
model, reserves memory, approves a backend or produces an executable PlacementPlan.
Default builds reject fixtures. Development builds with `test-backend` accept
`--allow-fixtures` explicitly. Empty eligible inventory returns no candidates and
`NO_ELIGIBLE_NODE`; it does not claim the model is unsupported everywhere.

The JSON contract is `starlings_core::planner::PreviewInput`, version 1. Unknown
fields, duplicate keys, invalid decimals and unsupported versions fail. Reuse the
existing [manifest](../spec/model-manifest.schema.json), NodeCapabilities and
[model-cost profile](model-cost-profiles.md) contracts, with no alternative model
inventory. Input is at most 1 MiB, depth 32 and 16,384 JSON values; these aggregate
limits can bind before the structural maxima. No file paths or executable names
are interpreted inside the record. Output contains identities, numerical costs
and fixed refusal codes, never endpoints, prompts or profile payloads.

| Input | Meaning |
| --- | --- |
| `version`, `fixture`, `manifest` | Version 1 and explicit fixture status; every nested capability/profile/link agrees. |
| `now_ms`, `now_wall_ms` | Caller-captured monotonic milliseconds as decimal u64 and UTC Unix milliseconds as i64. These are declared observations in a file, not authenticated live time. |
| `nodes` | At most 64 distinct node UUIDs in one cluster. Each entry has `capabilities`, `captured_at_ms`, `family`, `backend_id`, `expected_profile_key`, `profile`, `profile_clocks`, `approved_profile_digest`, `load_us`, `copy`. |
| `family` | `nvidia` or `apple`. NVIDIA requires Linux/Windows; Apple requires macOS ARM64. It is a declared family, not GPU detection or qualified backend evidence. Mixed-family candidates are refused. |
| `profile_clocks` | `captured_wall_ms`, `now_wall_ms`, `captured_at_ms`, `now_ms` from ProfileAge; the two current clocks must equal the top-level values. Profile freshness uses its original UTC measurement plus monotonic elapsed; see the profile contract. |
| `expected_profile_key` | Full expected model/execution/boundary/backend/runtime/device/pool/workload identity, including exact context, prompt, output, prefill chunk and sessions. All nodes use the same workload. |
| `approved_profile_digest` | Nullable declaration of whole-profile approval. Native data need a matching digest, measured basis and freshness. A supplied file cannot confer trusted approval; the output always says `qualification=not_assessed`, `admission=false`. |
| `load_us` | Decimal u64 conservative cold-load cost bound for any selected range on this node. No range-specific load measurement is inferred. |
| `copy` | `{prefill_us, decode_us_per_token}` decimal u64 costs for local stage copies at this exact workload. Distinct from directed interstage transfer. Zero is an explicit zero-cost assumption. |
| `links` | At most 128 distinct directed pairs, each with `source`, `destination`, `fixture`, `boundary_profile`, `workload_digest`, `evidence_digest`, `transfer`, `age_at_capture_ms`, `captured_at_ms`. Source/destination exist and differ. Digests are lowercase SHA-256; evidence is nonzero. |
| `workload_digest`, `transfer` | Canonical content hash of ProfileWorkload and positive `{prefill_us, decode_us_per_token}` decimal u64 transfer costs for that exact workload/boundary. They are declared costs; this preview does not extrapolate bandwidth into payload timings or verify the referenced evidence. Reverse edges are never inferred. |
| `objective` | `latency` or `throughput`; both are estimates under the formulas below. |
| `max_stages`, `top_k` | 1–8 stages, 1–16 ranked candidates. |
| `max_states`, `max_cost_rows` | 1–100,000 attempted stage evaluations; 1–1,000,000 pool-cost row visits. Caller-selected bounds, not additional job or retry limits. |

Capabilities must be ready and younger than 5 seconds at the declared observation;
links expire at 30 seconds. Future monotonic captures and arithmetic overflow fail.
The selected backend must support generation, model architecture, exact execution/
boundary profiles, build, context and session limits. Every profiled physical pool
must be associated with that backend's device. Pipeline stages also need declared
layer-sharding support. Reported identities/physical aliasing still require trusted
inventory and native conformance; this command never verifies those facts.

Profiles validate once per node, then use the same checked stage accounting as
GLY-179. Each selected stage includes its layers, its own fixed overhead, and all
first/last-owned components at the appropriate endpoints. Full-attention KV,
recurrent/convolution state and both phase scratch budgets are retained. Each
physical pool must fit independently within `available_bytes - reserved_bytes`;
this is a conservative cold-stage headroom estimate, not warm-cache admission or
an alias-free inventory proof. Pools and unified RAM are never summed together.

First evaluate every eligible single-machine full-model fit. If any fits, return
only single-machine candidates (or no ranked candidate if its timing is unknown): current policy pools only for capacity. Otherwise
search ordered simple paths with nonempty contiguous ranges covering all layers
exactly once. A device is optional and occurs at most once per candidate. Every
edge must be present, fresh and within the same family. The bounded depth-first
search visits nodes in UUID order and cuts in layer order. Work is bounded by the
stage and cost-row caps; a small top-k does not imply an unbounded search.

Let each stage's profile phase totals include its declared local copies. Add each
selected directed edge's transfer to the prefill/decode path. The latency score is
`sum(load) + prefill_path + requested_output_tokens * decode_path` in microseconds.
It is a conservative sequential cold-path ranking formula, not measured client
TTFT or latency. The throughput score is the largest stage decode-plus-outgoing-
transfer cycle in microseconds (smaller ranks first), not measured tokens/second.
It assumes pipeline overlap and does not prove a schedulable concurrent pipeline.
Neither formula models setup, queueing, token-return/client transport, contention,
prefill scheduling or power. Those costs must be separately collected/calibrated
before native admission or performance claims in GLY-181/239/241. The declared
load/copy/link inputs are included in `input_digest`, but a hash does not make an
estimate measured or trustworthy.

Candidates retain node and snapshot IDs, whole-profile digests, half-open layer
ranges, separate pool budgets, directed evidence digests and score components.
Sort ascending by score, then by the sequence of `(node UUID,start,end)`; keep top-k.
Refusal counts describe attempted nodes/edges/stages, not distinct root causes.
`node_refusals` maps supplied node UUIDs to fixed refusal codes and attempt counts
at the existing node preparation, stage and path checks. It identifies which
node's profile, capability, backend, pool budget, timing or incoming path prevented
that attempt. Global search/policy and stale-edge counts remain in `refusals`;
the node map is not a partition of every global count. It has at most64nodes and
ten fixed codes per node; no endpoint, arbitrary backend string, profile body,
prompt or tensor is added. This is an additive preview output field, not an input
or provider contract change. `plan modes` retains its separate aggregate report.

| Node refusal | Next input/evidence to check |
| --- | --- |
| `INVALID_PROFILE`, `UNQUALIFIED_OR_STALE_PROFILE` | Exact profile identity, original measurement/freshness and explicit matching approval. |
| `STALE_OR_UNAVAILABLE_CAPABILITY` | A fresh ready capability observation; reading a report does not refresh it. |
| `FAMILY_MISMATCH`, `INCOMPATIBLE_BACKEND`, `LAYER_SHARDING_UNSUPPORTED` | Actual same-family device/backend/architecture/profile/context/session and layer-execution conformance. |
| `POOL_BINDING_MISMATCH`, `INSUFFICIENT_POOL_MEMORY` | Individual physical-pool binding and range-specific budget against declared available/reserved memory; never sum aliased pools. |
| `MISSING_TIMING` | Measured timing for the exact workload/range; a memory fit can exist without it. |
| `MISSING_DIRECTED_LINK` | A fresh directed edge from the preceding stage to this node; no reverse reachability is inferred. |

A node can have refusal counts and still participate in a selected candidate: a
full-model attempt may fail memory while a smaller stage fits. Counts are not a
global exclusion verdict, a capacity certificate or proof that all alternative
ranges/paths were examined. Existing search-completeness and qualification flags
continue to govern interpretation. This offline output does not explain live
registry admission or authorize any new measurement or execution.

`search_complete=false` means a work cap stopped the search: returned candidates
are feasible under supplied assertions but not guaranteed best. `true` means all
policy-eligible search states were considered, not that any native plan qualified.
No persisted state is read or mutated beyond reading the input file.

`single_node_fits` records a validated memory fit even when its timing or later
cost score is unavailable. A false value in an incomplete search is not proof
that no single device fits. [Mode selection](mode-costs.md) reuses this signal and
scores inside the same search, assessing all configurations' singles first.

Controlled tests compare the implementation against an independent exhaustive
per-layer-assignment oracle on small hybrid fixtures for both objectives. They
also cover single-machine preference, optional slow devices, directionality,
family compatibility, stale/invalid inputs, missing timings, hard caps and checked
overflow. These establish software invariants only. GLY-175/179/181/235/236 native
profiles, calibration, provider delivery and transactional admission remain open.
