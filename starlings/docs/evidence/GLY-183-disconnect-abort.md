# GLY-183: local generation abort on unfinished flow

Issue: https://linear.app/getglyd/issue/GLY-183

Base: `aca7d99e1b334ce875b259f593292eca17f75136`. The PR records the
exact source commit and cross-platform checks separately. Local checks below
use the pinned Rust toolchain on macOS ARM64 and synthetic enrolled loopback
peers, without a model or native device workload.

## Reproduction and change

Previously, truncation or dropping an unfinished receiver poisoned its connection
and released host reassembly, but its admitted generation remained eligible.
The strengthened existing truncation/cancel check failed with `Admitted` versus
`Failed` in 0.17 seconds before the fix. Its first sandbox invocation was refused
at TCP bind before reaching this assertion; that is not behavioral evidence.

Both flow actors now use their immutable local admission to abort the owned
generation. Core session/plan/epoch checks prevent a stale or mismatched template
from selecting another generation. The operation needs no live peer, token or
registry freshness. It never starts execution or releases a lease. Pending work
becomes Cancelling and retains its reservation until trusted local drain. Existing
terminal states and the first cancellation reason remain unchanged.

No new runtime, dependency, wire field, job, retry, relaxed deadline or native
cleanup permission is introduced. Raw frame primitives still require a local
owner; deployed pipelines must use the aggregate flow API and own post-handoff
cancellation/device drain. See [the flow contract](../data-flow.md).

## Checks

| Command / control | Local result |
| --- | --- |
| Strengthened truncation/cancel control before fix | FAIL, generation still Admitted, 0.17s |
| `cargo test -p compute-security --lib -- --nocapture` | 113 passed, one existing ignored, 18.44s |
| `cargo test -p compute-core -- --nocapture` | 23 passed across suites; lifecycle 12 passed in 15.69s |
| `cargo clippy --workspace --all-targets -- -D warnings` | PASS |
| `cargo clippy --workspace --all-targets --features compute-cli/test-backend -- -D warnings` | PASS |
| `cargo fmt --all -- --check`; `git diff --check` | PASS |

The core check uses two independent live leases/generations. Wrong session, plan
and epoch are refused; repeated abort is idempotent, a new StartStep cannot replay
execution, unrelated work survives, and releasing a pending lease cannot reclaim
it until local drain. The existing pending-step final-handoff check also verifies
the real secured-worker release/drain path.

An actual connected, unread receiver has bounded 8 KiB OS buffers; an 8 MiB
TLS write fails within a caller-selected 500 ms transfer deadline. The sender
cannot continue, its generation fails, and driven receive failure aborts the
receiver and frees host reassembly. Core lease reservations remain. Local-drop,
truncation, wrong ordering and duplicate receipt controls cover other abort paths.
Both authority-backed and delegated StageWorker owners are exercised. Separate
registry/control-loss checks confirm local abort needs no reauthorization.

## Raw local log identities

Files remain in the task workspace; these hashes identify the observations,
not a backend qualification or cross-platform result.

| File | SHA-256 |
| --- | --- |
| `m3-disconnect-abort-control-before.log` | `935cfdc41587795c1d0f60ea514aefc25643e194785802adfd506e9613e2602e` |
| `m3-disconnect-abort-security.log` | `de84fb5a4ba506e8b34e34762b8fc10c03b10cada8f9a5d3b2b892001f28871e` |
| `m3-disconnect-abort-core-suite.log` | `08fbea2f548adb834b4e7f8e96c1960c450f9d5d85517867bf92af1aa99c1b14` |

Correctness/security review traced both owner variants, stale identity refusal,
lock ordering, pending execution/drain, error/drop paths and successful handoff.
The separate complexity review retained the existing core terminal transition
and flow poison paths; no further abstraction is needed. Independent milestone
review has not been approved.

## Combined review: expired final handoff

After integrating main `c55c8d3099e78995a31d35c3302bc3eeaeaedf6f`
at signed merge `04961e073a1acb435262d6478348b82cf49f414d`, review found
one more failure path. The final frame callback can move the receive reservation
into an owned result, then fail its post-callback deadline check. Dropping that
result frees host inputs, but the receiver's empty reservation previously skipped
generation abort. An existing-fixture control holds the actual policy lock across
the original one-second handoff deadline. Before correction, it returns Expired
and releases all host bytes while leaving the generation Admitted: FAIL1.22s,
raw SHA256 `3a06429fac2c3e59a3b6cdd53e0e3d935342280c65ff73f4a0217c1e29a92784`.

The receiver now explicitly aborts every final frame error, even when the
reservation has already moved and dropped. Successful handoff, trusted drain,
lease reservation and the original deadline are unchanged. No test-only runtime
hook, helper abstraction, retry or dependency is added. The same control passes
in the full local security suite:118 passed, two explicit ignored,17.79s,
raw SHA256 `6d635f6720bb78dde5d5842bee0d6fa27196d7d17a3848cc06084c180219d8e9`.
Both warning-denied workspace lint modes, formatting and diff checks pass.
Correctness/security review covers callback-result drop and both abort paths;
the separate complexity pass retains the existing abort method and native locks.
Final combined PR source/platform/main qualification is recorded separately.

## Remaining acceptance

GLY-183 stays In Progress. These are library/local-owner checks, not deployed
daemon pipeline integration, an actual native RunChunk cancellation, device
cleanup qualification, physical-link measurement or an Exo comparison. They do
not explain GLY-263/267's Windows registry/routing failures. Their failed CI logs
remain preserved; no unchanged-source rerun is justified by these passing checks.
