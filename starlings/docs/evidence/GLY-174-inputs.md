# Directed input and fault acceptance

This change supplies typed facts for later placement without starting M3. Status
maps current authenticated observations to both directed inputs with sample ID,
authority number/incarnation and original age. Conversion returns a redacted
error for invalid identities instead of panicking. Null measurements yield null
inputs; consumers still need the live authority gate, never a saved JSON file.

The explicit `examples/directed-links.json` fixture exercises four-to-one
asymmetry and a slower forward observation. `LinkFixtureReplay` preserves age on
repeats, shortens eligibility for added reported delay, rejects modified/older
sample replay, expires observations and retains withdrawal tombstones. History
and pair capacity are bounded with explicit refusal. It accepts fixtures only.

Real signed-proof/TLS loopback tests reject malformed limits, stale epoch,
wrong sequence identity, corrupt data and incomplete transfers at the expected
error boundary. A proved stall terminates within the existing deadline; revocation
interrupts an idle proved peer. Failed attempts consume cooldown and release
permits without occupying heartbeat capacity. The future-wall-clock test sets an
actual future time rather than assuming the test finishes within one second.

Commands: warning-denied default/development Clippy, core links test, security
probes test and default directed-probe daemon test. Final-head CI and exact
commit/PR identities are linked in Linear. Correctness/security review checks
live publication, UUID/epoch validation, no panic, byte/time bounds and replay;
complexity review retains direct standard-library maps/sets and existing TLS.
No dependency, backend grant, private/Exo code, parked code or timeout increase.

Loopback and synthetic replay do not qualify physical LAN, native inference,
calibrated stage costs or an Exo win. GLY-244's Windows failure remains separate;
M2 remains open for resource/profile/routing/benchmark acceptance. Independent
review is deferred, not independently approved. Spend is zero; no model download,
device workload, license/publication or quarantine reset is performed.

Windows run38354600383 exhausted the existing ten-minute job cap after its default
suite passed (5m01s) and development suite passed (2m24s); catalog lifecycle started
with approximately one minute left. It was not a test failure or a live job to
retry blindly. CI now uses native matrix partitions on two standard Windows
runners: default lint/tests and development lint/tests plus lifecycle. Their ten-
minute caps, test assertions, native scheduling and coverage are unchanged. The
stable `Core (windows-latest)` gate depends on every core matrix partition; any
failure, timeout, cancellation or skipped partition prevents success. Both
underlying Windows checks and the other platform/dependency checks must pass.
Partition-specific cache keys avoid two workers racing to save one new cache key;
existing reviewed cache prefixes remain fallback sources. Final-head evidence is
required before claiming measured speed improvement. Repository settings unchanged.

Split run38356501114 stayed within the cap and passed default Windows plus the
other platforms, but backend lifecycle exposed an executing invitation timeout.
The production-route lifecycle passed; warm-up failed during invite-startup-crash
with CONTROL_WAIT_EXPIRED phase=executing. Invitation diagnostics now record only
static subphase names before private-parent open, clock checkpoint, policy lock,
expired-row deletion, transaction, private-file publication and commit. Writes
are best effort; no path/token/member/credential is logged. This distinguishes
the blocking operation without retrying the mutation, weakening persistence or
changing the three-/five-second deadlines. Its root cause remains GLY-244 work.

Review replaced per-step logging with a nonblocking atomic phase observer. No
diagnostic I/O occurs while the authority/policy lock is held; only the already
timed-out admin handler reports the static subphase. The legacy library API uses
a no-op observer. No phase carries input or credentials, and executing subphases
retain the existing unknown-outcome semantics rather than queued-cancellation.
