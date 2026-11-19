# GLY-180 — bounded advisory partition preview

Issue: https://linear.app/getglyd/issue/GLY-180

Implements a pure bounded contiguous partition search and offline `plan preview`
command using merged GLY-179 profiles. It keeps per-pool memory separate, assigns
first/last components once, checks backend/workload/family and directed-edge
compatibility, ranks explicit estimated latency/bottleneck objectives and returns
refusal counts and truncation. Single-machine fit is preferred under the current
pool-for-capacity policy. No reservation, launch or inference is added.

Validation commands (pinned toolchain, no models or native devices):

```sh
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo clippy --locked --workspace --all-targets --features compute-cli/test-backend -- -D warnings
cargo test --locked -p compute-core -p compute-protocol
cargo test --locked -p compute-cli --test cli
cargo test --locked -p compute-cli --features test-backend --test cli
```

The independent small-fixture oracle enumerates every per-layer node assignment,
rejects repeated/noncontiguous devices and computes fixture memory/timing directly.
It compares full top-k ranking for 2–4 layers, three nodes and both objectives.
Separate controls exercise no mutation, snapshots, directionality, same-family
policy, slow-device exclusion, one-machine preference, stale/unknown costs,
compatibility, duplicate identities/edges (including stale duplicates), caps and
overflow. CLI controls cover default fixture refusal, explicit development mode,
unknown/duplicate fields, bounded reads and no output files/state mutations.

Local pinned-toolchain formatting and both warning-denied workspace/all-target
lint modes pass. Core/protocol 29 regression tests pass; both CLI modes pass all
seven cases. After correctness and complexity review, the final seven core
profile/planner controls pass (4 planner + 3 profile), and the final CLI preview
control passes in both modes. Exact source/CI links are recorded in Linear.

Correctness/security review fixed a capacity-policy edge: a memory-fitting single
worker with missing timing prevents pooling. Separate complexity review removed
the duplicate recursive link-evidence stack; completion reads costs/evidence from
the same directed map. Existing profile math and bounded JSON/file reading are
reused. No deadline, workflow/job, dependency or warning allowance changes.
Independent milestone review remains deferred, not approved.
See [contract/formulas/limitations](../partition-preview.md). Input-supplied
approvals, family, clocks and costs are advisory assertions, not trusted registry
facts. Native calibration/provider delivery and real sharding acceptance remain
open. M2 stays incomplete; no new native workload or spend occurred.
