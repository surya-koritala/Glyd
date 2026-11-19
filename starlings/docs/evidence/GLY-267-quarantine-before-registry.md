# GLY-267 local quarantine before registry wait

Issue: [GLY-267](https://linear.app/getglyd/issue/GLY-267).
Base: main `5a4dfa4b0409cb5c8789c2dde1529506b72d8752` (PR #98).

## Observed failure and bounded correction

PR #98 passed all seven source checks in CI38912042150. Separate
[automatic main CI38913468079](https://github.com/Sigbound/glyd-compute/actions/runs/38913468079)
failed Windows backend and aggregate; the other five checks passed. Backend
library/unit/CLI, eight short catalog cases and the production route case passed.
The final restart case failed at daemon.rs2860 after27.60s: service remained
waiting_for_registry, with offline/Transport observations, rather than reporting
the existing retained quarantine. Raw job117014852620 SHA256
`f1221e672d48f23b3026b0f9a4bf36e4a32e9e71100f16ed7a9f3457f246a477`.
The initiating Windows transport cause is still unknown. No unchanged-source
retry, timeout relaxation or successful native qualification is inferred.

Both warm-up and qualified-worker launches share Owner::start/supervise. Previously
supervise waited up to15s for online registry state before inspecting durable
local catalog ownership. A known local cleanup requirement was therefore hidden
by an unrelated availability condition. The shared startup path now opens and
checks the protected catalog first, holding its existing exclusive ownership
through registry admission and supervision. It never reads remote cleanup claims,
clears quarantine, changes recovered records or starts a second backend.

The registry gate remains mandatory for a clear/recovered catalog. Stop still
refuses launch; the later online/stop checks before pin and spawn are unchanged.
The catalog preflight remains on the service thread, outside admin/registry work.
One private callback seam makes this ordering testable without a registry client,
socket, daemon, monitor, native service or model. It is not a provider API.

## Controls and reviews

The new exact binary-unit test first failed against the original registry-first
ordering: REGISTRY_UNAVAILABLE was returned instead of SERVICE_CLEANUP_REQUIRED
(one failed case,0.01s). Moving that callback after the real Catalog open/ownership
check makes the same test pass in both feature modes. The bounded synthetic files
exercise quarantined, drained, legacy and malformed records, missing/recovered
records, unavailable/online registry decisions and stop. They assert that blocked
ownership never calls registry admission, exact record bytes remain unchanged,
failure releases the lock, successful preflight holds it, and no launch record is
created by preflight. This is an ordering/refusal test, not native cleanup proof.

Basic Mac only: cached offline dependencies, explicit installed SDKROOT and at most
two build jobs; no network activity, instrumentation, process inspection or model
workload. Final exact test passes in both feature modes (0.07s each); both workspace
all-target Clippy modes pass with warnings denied (0.56s/0.54s), formatting and
diff checks pass. Commands: cargo test -p compute-cli [--all-features] --bin
glyd-compute service::tests::durable_ownership_refuses_before_registry_without_mutating_records
--locked --offline -- --exact; cargo clippy --workspace --all-targets
[--all-features] --locked --offline -- -D warnings; cargo fmt --all -- --check.
Existing
bounded all-platform CI retains every case, job, assertion and time cap.

Correctness/security review follows both launch callers, catalog state validation,
stop/refusal paths, exclusive lock lifetime, and the unchanged pin/spawn/registry
checks. Separate complexity review retains one direct private helper for ordering
and the existing catalog parser/lock; no dependency, public schema or workflow is
added. Independent milestone/release review remains deferred, not approved.
GLY-267 remains open for unexplained initiating failures; full M2/provider/device
qualification is not supplied by these synthetic controls.

## Initial source CI remains failed

Signeda6a6f9e / [CI38915196028](https://github.com/Sigbound/glyd-compute/actions/runs/38915196028)
is unqualified: Linux x64/ARM64, macOS and dependency checks PASS; both Windows
partitions and aggregate FAIL. The new ordering unit control passes on Windows.
Windows default's stage-process case fails during initial registry heartbeat
(device line111/parent346, Transport); raw job117020745099 SHA256
`57fec831d46c7001881c4999399ff6f3b897367197e6c2e754f9479932e509ae`.
Windows backend passes eight short controls, then production mode two loses
registry authority before a ready sample (line710, REGISTRY_LOST/offline, later
online). The recovery case is not reached. Cache cleanup also fails removing a
locked executable; neither that OS cause nor an initiating heartbeat cause is
established. Raw job117020744961 SHA256
`9c1991205789fa976804f853bf2e53f2a1e28f3d26e4127f71e4c49f191b2e3c`.

Those child paths did not enable the already implemented fixed registry phase /
numeric-I/O diagnostic. Both development daemon launches and the stage-process
child now explicitly opt in to that existing trace. No new production observer,
test case, traffic, job, retry or deadline is added. Only compile/lint and the
file-only ordering control run locally; the process cases run in existing CI.
No shared cause is inferred from Transport or a later online observation. Keep
this PR open until all seven final-source gates pass; preserve these failed logs.
After the explicit trace opt-ins, exact file-only controls pass default0.07s and
development0.09s; both warning-denied workspace/all-target lints pass1.55s/0.45s,
formatting/diff pass. No process/network case ran on the Mac.
