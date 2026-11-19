# GLY-252: synthetic release barrier and request budget

PR #57 source `705fae87bc3e465fd05d3157543c385cc8352fca`,
run38498114947/job115638446049 passes startup but fails the two-worker response
assertion with HTTP503 WORKER_LOST. Raw Windows log SHA-256:
`410bc414327d9331af475458f86c5224296b730d2607095a7dfac67797daddb6`.
This is distinct from GLY-251's original idle capability and newer startup
admission failures. No exact archived backend/registry exit is inferred.

The synthetic child had an independent five-second controller-release timeout.
The existing two-worker request budget is ten seconds, while its queue saturation,
capacity refusal, retained-body read-back and queued disconnect checks use several
sequential observation windows plus CLI/process work. The child could disappear
before a controller still operating inside the request budget released it.

The controlled adapter check starts a real synthetic HTTP child, waits for its
generation marker, holds release for 5.5 seconds inside an unchanged ten-second
request budget, then releases it. Against the unchanged fixture source it fails
with Transport in 5.90 seconds. Raw `work/m2-gly252-before.log`, SHA-256
`68461a8cb5d3640f0487e65a39e778649be0508c28423fb947f087874dd673ce`.
The fixture launcher now passes the existing validated request budget internally;
the release barrier reuses it instead of the unrelated constant. It is not an
operator timing option or a production timeout expansion.

After correction the same case completes successfully. A second case never
releases the barrier and proves the child exits within its existing one-second
request budget plus a bounded exit observation; no detached fixture can wait
indefinitely. The combined check passes in 7.31 seconds. Raw
`work/m2-gly252-after.log`, SHA-256
`d8bf63d0d69f44cc1f3a9e8f715c2e66d7d3e8674ec113cf208ede0973951a52`.

Runnable check:
`cargo test -p compute-adapters --all-features --lib release_barrier_uses_the_request_budget_and_expires_without_a_controller --locked --offline -- --nocapture`.
The existing production overload assertions remain intact. Final-response failure
diagnostics report both existing worker service records only when the assertion
fails, permitting actual backend/registry/resource attribution on a new failure.

Correctness/security review verifies the extra environment value is computed from
ServiceConfig's validated budget and used only by the synthetic launcher; it cannot
change native request, registry, admin, lease, output or CI limits. Process stop and
test files are cleaned before result assertions. Ponytail complexity review reuses
the existing launcher, budget, marker and stdlib test harness; no dependency/job or
new runtime contract is added. Exact-source CI and merge identities belong in Linear.

These are model-free software checks. Provider delivery, approved native/current
hardware and matched two-real-worker records remain M2 gates; M3 has not started.
Old failed records and quarantines remain intact. Neither a corrected barrier nor
a later green run proves GLY-251's original Windows heartbeat trigger.
