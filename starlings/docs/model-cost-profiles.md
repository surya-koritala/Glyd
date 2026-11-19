# Model cost profile v1 (proposed InspectModel result)

GLY-179 defines a strict provider-independent result and pure stage accounting for
GLY-180/181. This is a proposed external InspectModel contract, not an existing
installed Glyd API. Provider negotiation/worker delivery remains GLY-185/235;
measured profiles and calibration remain GLY-175/236. No new backend is launched,
no current daemon placement changes, and no native layer support is qualified.

Use the existing `decode::<ModelCostProfile>(bytes, FixtureMode)` / `encode`
functions. The [schema](../spec/model-cost-profile.schema.json) and Rust records in
`starlings-protocol::costs` enforce strict fields, canonical decimal u64 byte/time
values, bounded workload dimensions, duplicate-aware JSON and fixture isolation.
All embedded references are local; validation fetches no files or network data.
The standard 1 MiB/depth/node budgets apply even to the schema's maximum arrays.
Schema-required nullable confidence and timing fields must be present, including
when a profile is nested in preview, mode or calibration input. Explicit null
retains an unknown value; omitting the field is invalid input, never an implicit
null supplied during typed deserialization.

## Delivered Glyd cost data

The delivered `glyd-costs/1` document is a distinct provider contract, not this
`ModelCostProfile`. Inspect an already supplied local document with:

```
starlings plan inspect-glyd-costs PRIVATE_PROFILE_JSON
```

This file-only command hashes the exact input and projects bounded build/model
labels, workload, size/working-memory basis, per-layer resident/active/KV/combined
state bytes, aggregate allocations and reported timing. It preserves null values,
zero timing samples and estimated timing; none becomes measured or zero memory.
The receipt always reports `admission: unsupported` with typed missing Compute
fields. Output contains the supplied build/model labels; keep it private where
those labels identify a private installation.

The existing duplicate-aware 1 MiB/depth/node parser applies. Projected fields
must have their documented types. Projected nullable fields must be present;
explicit null remains unknown. Layers must be contiguous and workloads fit
context. Other provider fields are deliberately unprojected, not validated by
this command. This is a receipt and assessment boundary, not full provider-schema
validation, raw-evidence approval or an import into placement/admission.

Provider `prefill_chunk_tokens` may exceed the prompt (for example, 2048 against
1024); the receipt retains that configured maximum rather than rewriting it to
fit Compute's `ProfileWorkload`. `mamba` and combined state remain exactly that,
without inventing a recurrent/convolution split. The prose contract does not
specify the JSON type of `basis.kv_and_state`; it remains unprojected with an
explicit unverified-state-basis blocker. Provider source commits and
asset hashes cannot replace the registered Compute build and manifest digests;
host/device labels and transfer links do not establish physical memory pools.

Real conversion still requires registered model/build/runtime/device/pool
identities and component ownership, separate state and phase allocation bounds,
scratch/staging/reserves, measured timings with reviewed confidence, freshness,
raw evidence and explicit local approval. No provider executable, service or
hardware is started; existing planning and native eligibility are unchanged.
See the [software evidence](evidence/GLY-179-glyd-cost-receipt.md).

## Identity and evidence

`ModelCostProfile` contains `kind: ModelCostProfile`, `version: 1`, `fixture`,
`key: ProfileKey` and `costs: ModelCosts`.

| Record | Fields |
| --- | --- |
| ProfileKey | manifest_digest, execution_profile, boundary_profile, backend_build_digest, runtime_config_digest, device_digest, pool_inventory_digest, workload |
| ProfileWorkload | context_tokens, prompt_tokens, output_tokens, prefill_chunk_tokens, sessions |
| CostMeasurement | basis (estimated/measured), evidence_digest, observed_at, valid_for_ms, samples, relative_error_bps |
| ModelCosts unavailable | status: unavailable, code: bounded uppercase error code |
| ModelCosts available | status: available, measurement, pool_ids, layers, components, per_stage, per_stage_timing |
| LayerCost | index, layer_type, state_profile, pools, timing |
| ComponentCost | name, disposition, owner_role, pools, timing |
| PhaseCost | prefill_us, decode_us_per_token (both positive); nullable when unmeasured |

Workload bounds: context/prompt/output/chunk 1..1,048,576; sessions 1..256;
prompt+output must fit context and chunk must fit prompt. A profile applies only
to the exact workload key. No interpolation, batching assumption or fixed
compression multiplier is supplied by this contract.

There are 1..8 distinct pool IDs, 1..4096 layers and at most 128 components, within
the shared JSON cap. Every cost vector explicitly lists every pool in the same
order, including zero values. Layer indices cover the manifest exactly; types and
state profiles agree. Every declared non-layer component is present in manifest
order, including embedding, head, vision and MTP entries. Executed components
need first/last ownership; not_in_scope components need none ownership, no timing
and zero costs. Unsupported components require an unavailable result. Known
full-attention layers require KV costs; linear-attention layers require recurrent
or convolution state. A manifest's nonzero resident weights cannot disappear
into an all-zero profile row. Runtime values are not derived from catalog weights.

The runtime configuration digest binds the exact approved backend arguments,
relevant environment and settings, including offload and threading; it cannot
be replaced with an executable hash or a nominal precision label.

The device/pool inventory digest must identify the actual qualified hardware,
physical pool mapping and relevant software/clock/power settings. Shared/unified
RAM gets one physical ID, not both fictitious host and GPU capacities. Structural
validation cannot attest that an inventory is truthful or alias-free; its local
qualification must establish those facts. Different physical pools remain separate
in every budget; their capacities cannot be combined.

Measured basis requires at least one sample and an explicit error bound in basis
points (0..10000); estimates may have zero samples and unknown confidence. The
bound and raw evidence digest remain visible. This is supplied evidence metadata,
not a statistical guarantee or proof of real measurement. The native qualifier
must verify raw evidence, sample adequacy, workload coverage and conservative
memory/latency bounds before approving the entire profile digest.

## Per-pool accounting

Each PoolCost contains `pool_id` and these canonical decimal byte values:

- resident_weight_bytes: actual owned model weights at this pool;
- resident_overhead_bytes: fixed non-weight allocations, including runtime/context;
- active_weight_bytes_per_token: tier bytes read per decoded token, for later
  calibrated tier-cost prediction; never added to resident memory a second time;
- kv_bytes_per_token: per-session state at the declared context precision/layout;
- recurrent_bytes_per_session and convolution_bytes_per_session: fixed session state;
- prefill_scratch_bytes and decode_scratch_bytes: upper bounds for the entire
  declared workload concurrency, not per-token allocations;
- staging_bytes: owned buffers for copies/transport;
- reserve_bytes: separately justified allocator fragmentation/safety headroom.

These fields apply to each layer, owned component and one per-stage cost vector.
Per-stage fixed allocations are counted on every instantiated stage. First-stage
components are counted only for a range beginning at zero; last-stage components
only for a range ending at layer_count. A single full-range stage counts both.

For each pool, sum the selected rows and its per-stage row, then:

```
state = (kv_bytes_per_token * context_tokens
         + recurrent_bytes_per_session + convolution_bytes_per_session) * sessions
total = resident_weight_bytes + resident_overhead_bytes + state
        + prefill_scratch_bytes + decode_scratch_bytes + staging_bytes + reserve_bytes
```

Both scratch bounds are retained because concurrent sessions can occupy different
phases. Summing layer scratch is deliberately conservative; actual allocation/
overlap evidence is required before tightening it. All sums/products are checked;
overflow refuses a budget. Prefill/decode phase durations sum selected timing rows
and per-stage timing; any missing timing yields null, never zero. These are compute
costs only, not end-to-end latency, link cost, calibrated throughput or a speed win.

## Pure assessment API

`starlings_core::profiles::assess(profile, manifest, expected_key, layer_range,
receiver_clocks, approved_profile_digest, fixture_mode)` validates both records,
exact bindings, coverage and range, returning per-pool breakdowns, nullable timing,
evidence/confidence, profile digest and explicit status. It makes no reservation.

ProfileAge contains receiver-owned `captured_wall_ms`, `now_wall_ms` (Unix UTC ms),
`captured_at_ms`, `now_ms` (same monotonic origin). Derive them from local clocks;
never accept producer-supplied receiver timestamps. Age is the larger of current
wall age since observed_at and original capture age plus monotonic elapsed.
Backward receiver clocks, future measurement time or arithmetic overflow refuse.
Expiry is inclusive at age >= valid_for_ms (maximum one day). Re-observing an old
profile does not renew its fixed measurement timestamp; suspend cannot hide age
behind a stopped monotonic clock. The owner must retain its clock high-water marks
across observations; this pure function is not durable clock/rollback storage.

Statuses are unavailable, stale, fixture_only, estimated, unqualified and qualified.
A fresh measured record becomes qualified only when its ENTIRE canonical profile
hash matches trusted local approval. A digest supplied by the record/peer is not
approval. Even qualified means only an approved cost profile: actual backend
boundary capability, live authenticated resource freshness, exact workload/device
matching, capacity reservation and transactional admission remain separate gates.
GLY-181 owns that integration. Offline budget rows do not activate a route.
