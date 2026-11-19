# Offline placement-mode costs

`starlings plan modes INPUT_JSON` compares supplied pinned configurations
using the existing [bounded partition search](partition-preview.md) and
[profile accounting](model-cost-profiles.md). It starts no backend or probe,
reserves no memory and always reports `admission=false` and
`qualification=not_assessed`. Default builds reject fixtures. Development builds
require `--allow-fixtures`; [the example](../examples/mode-preview.json) is entirely
synthetic, including its model, hardware, read rates, labels and costs.

## Input contract

The strict internal Rust/JSON contract is `starlings_core::modes::ModeInput`, v1.
The CLI enforces the existing 1 MiB, depth32, 16,384-value, duplicate-key and
unknown-field limits. No referenced asset, evidence path or executable is opened.
Canonical decimal u64 strings carry bytes, bandwidth, microseconds and monotonic
milliseconds; versions and bounded counters retain their existing typed forms.

| Record | Fields and meaning |
| --- | --- |
| ModeInput | `version`, `fixture`, `family` (`nvidia` or `apple`), 1–16 `variants`, `link_limits` |
| Variant | `configuration` (`resident` or `expert_offload`), existing `placement: PreviewInput`, `costs`, nullable `approved_costs_digest` |
| ModeCosts | `tiers`, `storage`, `links`; all are required arrays, with unknown data explicitly unavailable or absent |
| TierBandwidth | `node`, exact whole `profile_digest`, physical `pool_id`, positive `bytes_per_second`, `observation` |
| StorageRead | `node`, exact whole `profile_digest`, `traffic` |
| StorageTraffic | `{status:not_used}`, `{status:unavailable}`, or `{status:available, bytes_per_second, layer_bytes_per_token, first_stage_bytes_per_token, last_stage_bytes_per_token, per_stage_bytes_per_token, observation}` |
| LinkLatency | directed `source`, `destination`, matching `link_evidence_digest` from PreviewInput, nullable `p50_us`, `p99_us`, `idle_wake_us`, `observation` |
| Observation | `basis` (`estimated` or `measured`), nonzero `evidence_digest`, `captured_at_ms`, `age_at_capture_ms`, `valid_for_ms` (1–86,400,000) |
| LinkLimits | `p50_us`, `p99_us`, `idle_wake_us`; explicit supplied limits, not inferred defaults |

Every variant has the same complete manifest, precision/level, workload, cluster,
family and current clocks. A repeated node has the identical captured capability
view across configurations. Profiles may differ in their pinned runtime/backend
configuration. Existing profile freshness, approval, backend compatibility,
physical pool binding and per-pool free-memory checks all remain mandatory.
No precision change is attempted. The `expert_offload` label is a supplied
configuration classification; this command cannot prove a model is MoE or that
an installed backend supports expert offload. That remains provider conformance.

A tier references a real memory pool from that node's capability record and its
exact profile. SSD traffic is separate; it creates no fictional RAM/GPU pool or
additional memory capacity. Available storage vectors cover every manifest layer
in order and separately account for first/last components and per-stage I/O.
`not_used` is an explicit assertion bound by cost approval, never a default.
Missing/unavailable storage prevents a prediction. Storage capacity, catalog
verification, paging/caching behavior and actual staging allocations still need
qualified native inventory and profiles.

For non-fixture inputs, observations must declare measured basis and the variant
must supply `content_hash([configuration,costs])` as `approved_costs_digest`, in
addition to existing whole-profile approvals. The digest binds identities,
observations and explicit zero-storage assertions, not just the rates. Age is
`age_at_capture_ms + now_ms - captured_at_ms`, checked for future captures,
overflow and expiry. Receiver-owned live clocks/evidence verification are still
required before actual admission. A file containing its own matching hashes is
not proof of trusted approval, adequate samples or real measurement; reports
remain unqualified even when all declared approvals match.

## Ranking and limits

First assess single-device fits across every configuration, including offload.
Any eligible single-device fit suppresses every split, even if its prediction is
unknown. Only when none fits does the resident configuration enter the existing
contiguous, non-repeating-node, same-family split search. Offload configurations
remain single-device alternatives. An unapproved/stale profile cannot establish
an eligible fit. No mean/variance is converted into p99 or idle-wake.

For one sequential decoded token at the exact profile workload:

```
memory_read_us = sum ceil(active_weight_bytes_per_token * 1_000_000 / tier_bytes_per_second)
storage_read_us = sum ceil(selected_storage_bytes_per_token * 1_000_000 / storage_bytes_per_second)
decode_us = memory_read_us + storage_read_us + declared_local_copy_us + sum(directed_p50_rtt_us)
predicted_tokens_per_second = 1_000_000 / decode_us
```

Each storage range sums its selected layers, per-stage I/O and components owned
at its first/last endpoints. Prefix sums keep range queries constant-time. Each
remote stage boundary adds one whole measured RTT, once. Its explicit p50, p99
and idle-wake values must all be known, fresh and within limits. Provided p50
must describe the intended workload/cadence; p99 and idle-wake are refusal guards,
not inferred distributions or a radio power-state simulator. Eight old probe
samples do not establish a population p99 or an unobserved idle condition.

Byte/time arithmetic uses checked u64 results with u128 intermediate division and
rounds up. At least one memory tier per stage has nonzero active bytes; absent
active traffic, missing rates, unknown tails, unapproved costs and overflow get
explicit refusal codes. This is a memory-read model with declared copies/RTT;
it omits compute bottlenecks, allocator/OS effects, load/queue/prefill/client costs,
contention and pipeline overlap. Declared profile timings remain required by the
existing planner but do not replace this ranking formula. Native calibration
must determine whether the approximation is useful for each workload.

Scoring happens inside the bounded search before top-k truncation. Post-filtering
the old timing-ranked top-k would miss valid tier-cost winners. Across all
configurations and both passes the command allows at most100,000 stage attempts
and1,000,000 pool-cost row visits, further narrowed by each PreviewInput's caps.
There are at most64 distinct nodes,512 tier rates,64 storage entries and128 link
latencies per variant. The aggregate JSON budget can bind earlier. Hitting a work
cap reports `SEARCH_INCOMPLETE` with no winner/runner-up. Repeated single-device
assessment during the capacity search also consumes those shared budgets.

Output includes reason/refusal counts, work counters, known single-device fit,
winner and distinct runner-up, exact profile keys/digests, layer ranges and
separate pool budgets. Predicted memory/storage/copy/RTT costs retain evidence
digests. Sorting is cost then variant digest then ordered node/range; equivalent
profile/configuration/range choices do not fill both output slots. Refusal counts
count attempted evaluations, not unique causes. This is no executable placement,
measured throughput, native memory guarantee or Exo result.

GLY-239 remains open for delivered GLY-236 data, approved native calibration,
predicted-versus-measured records including failures, and integration into trusted
transactional admission. Existing M2/provider and milestone/release gates remain.
