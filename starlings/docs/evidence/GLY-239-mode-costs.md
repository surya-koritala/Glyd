# GLY-239 offline mode-cost implementation

Issue: [GLY-239](https://linear.app/getglyd/issue/GLY-239).
Base: `a0145b7231035893efcc5e3f09190ae265939e37` (PR #95).
Exact source, PR and CI qualification are recorded in Linear after publication.

The [strict ModeInput contract](../mode-costs.md) now drives `plan modes` from
bounded local JSON. It reuses the existing planner's profile assessment, backend/
family/pool checks, stage accounting, directed paths and work counters. A private
scoring hook evaluates tier costs before truncation, so a timing-ranked top-k
cannot hide a memory-tier winner. Every configuration's single-device fit is
checked before any split. Checked read-time arithmetic includes separate SSD
traffic without increasing RAM/GPU capacity. Results show distinct alternatives,
exact profile/configuration identities, per-pool memory and reason/refusal codes.

## Review and controls

Correctness review caught a configuration-order error in the initial local draft:
one resident variant could exhaust a split budget before a later fitting offload
variant was assessed. The final two-pass search fixes the cause; regression
checks both variant orders with a budget sufficient only for single-node checks.
Profile validation also correctly refused an initial overflow test fixture that
assigned nonzero traffic to excluded components. The corrected fixture leaves
excluded rows zero and reaches the intended checked overflow refusal.

Security/correctness review checks common manifest/level/workload/clocks, identical
captured hardware views, whole-profile/rate/storage identity binding, explicit
zero-storage assertions, native-declaration approval hashes, freshness, directional
latency references, missing tails, exact pool headroom and aggregate work caps.
No output confers authority or launches work; all reports remain unqualified.
Separate complexity review keeps direct standard-library maps and one existing
search implementation. No dependency, job, backend API or new runtime is added.
Independent milestone/release review remains deferred, not approved.

## Basic Mac checks

Only ordinary model-free checks ran locally with at most two build jobs. They
open no sockets, inspect no processes and start no model/service/instrumentation.

* Six mode tests pass (0.37s observed): whole/offload alternatives and identities,
  internal-search ranking, capacity-only splits, once-per-edge RTT, global
  single-first ordering, storage/end-component accounting, explicit refusal
  codes, malformed/substituted identity and non-fixture declaration boundaries.
* Four existing planner checks pass (0.37s), including the independent exhaustive
  small assignment oracle. They preserve old ranking/fit/family/limit behavior.
* One actual CLI file-input check passes in each feature mode (0.74s/0.81s):
  production fixture refusal, explicit development allowance with no admission,
  duplicate/unknown fields, input-size bound and no additional output files.
* Both workspace/all-target lint modes pass with warnings denied. Final formatting
  and whitespace checks are required before commit; all platform CI remains required.

```sh
CARGO_BUILD_JOBS=2 cargo test -p compute-core --test modes --test planner --locked --offline
CARGO_BUILD_JOBS=2 cargo test -p compute-cli --test cli offline_modes_rejects_fixtures_by_default_and_keeps_file_inputs_bounded --locked --offline -- --exact
CARGO_BUILD_JOBS=2 cargo test -p compute-cli --all-features --test cli offline_modes_rejects_fixtures_by_default_and_keeps_file_inputs_bounded --locked --offline -- --exact
CARGO_BUILD_JOBS=2 cargo clippy --workspace --all-targets --locked --offline -- -D warnings
CARGO_BUILD_JOBS=2 cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
```

## Initial source CI and build scheduling

Source `2e84b3f9d9a70b4ba5903c81e08a69b2a629cdc0`,
[run 38905353059](https://github.com/Sigbound/glyd-compute/actions/runs/38905353059),
is not qualified: five platform/policy checks passed, Windows backend concluded
Cancelled at the unchanged ten-minute job boundary and its aggregate failed.
Every Windows backend test group passed before final cleanup: relay1, remaining20
library,23 adapter,9 binary,8 CLI and10 catalog lifecycle cases. The successful
three successive compilation phases took22.36s,23.42s and26.74s (72.52s total);
the full production matrix took192.06s and ownership/restart64.81s. This is a job
budget failure, not evidence that an assertion failed or a historical product
failure was explained. Raw job116987414595 log SHA256:
`c0c62843a223b594eef0c700faba5c1af29b20a11224e0a655b340482b35deb0`.

The correction compiles the same five backend test executables in one Cargo
`--no-run` graph before their unchanged disjoint serialized execution groups.
This permits independent compile units to overlap; runtime concurrency, all
assertions, diagnostics, filters, warning-denied lints, jobs and deadlines stay
unchanged. There is no unchanged-source rerun. Actual platform CI must determine
whether scheduling provides sufficient headroom; no speedup is claimed here.

Basic Mac validation: the exact offline compile-only selection succeeds with
two build jobs in15.82s and lists all five executable targets without running
any. All11 existing multiline shell blocks pass `bash -n`; deleting the three
added workflow lines reproduces the original workflow byte for byte. Correctness
review verifies fixture/runtime coverage is untouched; separate complexity review
keeps a single standard Cargo invocation without scripts, dependencies or jobs.

## Remaining acceptance

These are synthetic software controls and supplied-data cost estimates. The
fixture marked non-fixture in the approval-boundary check is still synthetic;
matching supplied hashes proves consistency only, never real measurement.
No 2026 model, GPU, SSD I/O, physical network, native worker or Exo comparison ran.
GLY-236 profiles/provider mapping, approved device/memory/storage/link inventory,
adequate percentile/cadence evidence, predicted-versus-measured calibration and
trusted transactional admission remain open. GLY-239 is not complete merely
because this software path passes. The Mesh license stays undecided.
