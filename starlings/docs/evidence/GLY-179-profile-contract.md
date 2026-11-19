# GLY-179 independent cost-profile contract

Issue: https://linear.app/getglyd/issue/GLY-179

## Implementation boundary

Add proposed strict ModelCostProfile v1 / InspectModel result records and a local
schema using the existing protocol decoder, canonical scalars, identity validation
and content hashing. A pure core assessment checks manifest/key/range coverage,
returns per-physical-pool memory/timing breakdowns and exposes unsupported,
stale, fixture, estimated, unqualified and explicitly approved profile states.
It does not expose an installed provider operation, change M2 routing, reserve
capacity, launch a backend or qualify native layer execution.

Memory includes weights, fixed non-weight overhead, full-attention KV, recurrent/
convolution state, both phase scratch bounds, staging and reserve. Components
include first/last ownership and explicit excluded vision/MTP entries. Context and
session multiplications and every sum are checked. Different pools never combine
into one capacity. Runtime weights are supplied for the exact profile, without a
nominal compression multiplier or substitution of catalog source sizes.

UTC measurement age and receiver monotonic elapsed can only shorten freshness.
Re-capturing old profiles cannot reset the fixed measurement timestamp; suspend,
rollback/future timestamps and overflow have explicit refusal/stale behavior.
Measured basis is not approval: only a trusted local full-profile digest may
approve cost data, and that still does not admit a pipeline or qualify a backend.
Native evidence/pool mapping, confidence policy and actual integration stay open.

## Review and validation

Correctness/security review traced decode/schema/fixture boundaries, exact
manifest/runtime configuration/workload binding, full ordered layer/component/pool coverage,
first/last extras, state/scratch formulas, checked arithmetic, missing timings,
measurement age and explicit approval. Fixed concurrent scratch accounting to
retain both phases and split fixed overhead from weight memory. No global clock,
TTL, routing, resource policy or quarantine was changed.

Separate Ponytail complexity review reused existing strict parsing, embedded schema,
time parser, hashes and decimal/bounded types. Replaced a redundant per-pool map
with direct vectors because validation already fixes their order. No new dependency,
thread, backend interface implementation, cache, daemon integration or CI job.
Independent milestone review remains deferred, not independently approved.

Pinned Rust/Clippy 1.97.1 (rustup shims first on PATH):

- Formatting and both warning-denied workspace/all-target lint modes pass.
- `RUSTFLAGS='-D warnings' cargo test -p compute-core -p compute-protocol --locked --offline`
  passes core unit1/lifecycle11/profile3/routing3 and protocol contracts7, plus
  empty unit/doc selections. Raw log SHA256
  `5d7409e9450d97bd56583ed454af274c9a0aa8538eddeb24405aa8e1893095cf`.
- Final three profile controls pass after the receiver clock/approval refinements;
  the existing core/protocol suites were not repeated without a new relevant reason.
  All final-source platform CI gates remain required before merge.

Controls use synthetic two-layer full/linear-attention fixtures with two distinct
physical pools and all declared components. They prove exact per-pool arithmetic,
both range endpoints, omitted/mismatched coverage, missing KV/recurrent state,
excluded components, malformed/unknown/duplicate fields, decimal overflow,
invalid workloads, default fixture refusal, explicit fixture/estimate nonapproval,
exact workload identity, full-profile approval invalidation, stale replay,
suspend, backward clocks and missing timing propagation. No simulated profile is
presented as a real machine/model measurement.

Final identity review added the full runtime-configuration digest: a binary pin
alone cannot distinguish threading/offload settings. Its mismatch control passes.
The resulting source requires a fresh exact-source CI gate; older source CI is
retained with its actual outcome, never promoted to final qualification.

Initial lint checks caught unit expressions and a collapsible guard; test compilation
caught obsolete clock-field construction during the receiver-age refinement.
These were corrected without suppressions or relaxed checks.

## Remaining gates

GLY-179 stays open for actual provider InspectModel delivery and native profile
qualification; GLY-175/236 measured inputs and GLY-185/235 worker delivery remain
external gates. GLY-180 search and GLY-181 transactional admission are separate.
M2 remains incomplete; its new failed main8afe54d/CI38844775696 Windows 503 assertion
is preserved separately under GLY-267. This patch does not claim to fix that cause.
No native/model/kernel/GPU/Ubuntu/SSH/download/spend/quarantine reset, license,
repository protection or publication change.
