# GLY-301 — Successful heartbeat cadence

Continues [GLY-267](https://linear.app/getglyd/issue/GLY-267).
Base: `1dff9d16c95c6517003827ac92381d3b8352fd9b`.

## Defect and correction

The worker fence ages an authenticated exchange from its start. The daemon
previously added the complete randomized 2–3 second idle interval after a
successful exchange finished. Two valid 2.5 second exchanges with a three second
idle place the next response eight seconds after the prior start, beyond the
six second fence lifetime. The idle scheduling itself adds an avoidable gap.

Measure successful cadence from the exchange start instead. The same example
now receives its next response at 5.5 seconds. If the prior exchange already
consumed the interval, proceed directly to the next attempt. Preserve the
existing completion-relative error backoff, stop polling, jitter and every
deadline/authority/freshness check. Late replies still fail. Unbounded scheduler
pauses, failed I/O and wall discontinuities can still legitimately expire authority.

The small deadline function is used by the real daemon loop and checked with
controlled Instants, without sleeps or a network timing simulation. It tests both
jitter endpoints, accounts for an attempt not starting before the previous one
finished, and checks unchanged completion-relative failure backoff. Restoring
the original scheduling expression with the final test gives the expected fence
gap assertion failure; the corrected implementation passes.

## New retained Windows failure

[PR78 source CI](https://github.com/Sigbound/glyd-compute/actions/runs/38839329759)
passed all seven checks before its reviewed merge. The subsequent
[automatic main run](https://github.com/Sigbound/glyd-compute/actions/runs/38840126656)
failed Windows backend job116778061880 and the aggregate, with five other checks
passing. Raw job log SHA256:
`081a3efe97b7eb2f10aeb20aa2a7b21ad15025206bcd23f9dbbe5bb5e8e4af16`.

The short eight-case lifecycle group passed (44.27 seconds). Its earlier caught
poison, invitation and stopped-daemon panics are expected passing controls.
The actual failed production matrix result is a socket read timeout10060 at
daemon.rs965 before partial SSE output (98.30 seconds). The service retained
its pin/quarantine with REGISTRY_LOST/offline and zero reserved bytes. Later
worker membership was online at epoch1; coordinator routing had active0/queued0.
Client Transport messages occur during teardown, not proved initiation.

The scheduling defect is independently demonstrated. The log does not prove
which host operation or scheduling pause caused this particular Windows lapse.
Do not equate the clock-discontinuity control with historical causal attribution,
or reclassify any original failed run after this correction.

## Validation and boundaries

```sh
export PATH="$HOME/.cargo/bin:$PATH"
rustup run 1.97.1 cargo test -p compute-cli --bin glyd-compute --locked --offline
rustup run 1.97.1 cargo test -p compute-cli --all-features --bin glyd-compute --locked --offline
rustup run 1.97.1 cargo test -p compute-cli --all-features --test daemon catalog_service_ --locked --offline -- --nocapture
```

Pinned formatting and both warnings-denied workspace/all-target lint modes pass.
Both binary suites pass (six default and ten development cases, 3.01 seconds
each). All 18 synthetic catalog lifecycle cases pass in 67.45 seconds. The initial
private-module reference compile error is corrected to the existing public
constant and excluded from the causal reproduction. Exact final-source and
post-merge qualification receipts remain in GLY-301.

No protocol/config schema, TTL, exchange budget, request retry, backend memory
ownership, quarantine rule, dependency, CI job/selection or cap changes. The
ignored 63-peer acceptance has its own explicit stress cadence and is unchanged.
No model/native/GPU/Ubuntu/download/spend or old recovery reset occurred.
Independent milestone review remains deferred, not approved.
