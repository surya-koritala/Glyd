# GLY-267 heartbeat response boundary

Issue: [GLY-267](https://linear.app/getglyd/issue/GLY-267).
Base: merged main438e293a16d555849126e45ba60e5f3befa87167 (PR #99).

## Current evidence

PR #99 sourcef61ce29/CI38916457383 passed all seven checks. Its independent
[main CI38917868580](https://github.com/Sigbound/glyd-compute/actions/runs/38917868580)
fails Windows backend/aggregate; five other checks pass. Production mode fails
at daemon.rs972 reading partial output after77.26s. Worker records a timed-out
heartbeat response: REGISTRY_FAILURE phase=client_frame_receive, Transport,
Windows10060, then REGISTRY_LOST/offline; later status is online. Raw
job117029895769 SHA256
`02b135be7d710075d8e3daf428f07334852fb2b95d064e3f540b0173351925d7`.
The client had reached frame receive, narrowing the boundary beyond TLS/send;
it does not identify server scheduling, lock, storage, reconciliation or I/O cause.

Independent PR #100 source6abd235/CI38918028918 also fails Windows backend and
aggregate, with five other checks passing. Its new planner/CLI checks pass;
production token publication returns CONTROL_UNAVAILABLE with an unknown outcome
(daemon.rs1773). Command expiry is executing/invitation_step=none and total command
duration4657ms, not a measured storage subphase. Raw job117030445983 SHA256
`3729d4d7f703df5cf3c1952ac14d2f21af91680422f3d87940213c7f961ea142`.
That PR remains unmerged; no common initiating cause is claimed. Both failures
and all older records remain preserved; no unchanged-source retry.

## Bounded observation

The existing explicit GLYD_TEST_REGISTRY_TRACE opt-in now records at most eight
fixed response-phase entries in debug builds: dispatch_wait, policy_lock,
policy_read, credential_check, registry_update, policy_reconcile, frame_send and
receipt_to_reply_total. Only phase, duration in milliseconds and the existing
redacted Error enum are retained. A failed response or a recorded interval of at
least one second prints the bounded summary. This is a diagnostic threshold,
not an exchange/admission/freshness deadline. There are no peer IDs, addresses,
payloads, database queries/values, paths, credentials or raw OS diagnostic text.

The eight-slot vector is allocated before policy access. Steps only record local
durations/results; output occurs after reply_measured returns and every policy
guard is gone. Diagnostic write failure is ignored without changing the actual
operation result. Disabled tracing collects no times; release builds cannot emit
this output. Existing CI fixtures already opt in; no new job, case selection,
traffic, retry or timeout is added.

policy_lock includes Access::lock's storage/clock checks, not only mutex waiting.
registry_update includes sample validation, durable boot/sample replay counters
and in-memory publication; a slow interval alone is not proof of an fsync cause.
The total begins at server frame receipt and does not include preceding client
connect/TLS time. Dispatch includes the existing event/refresh path. This observer
does not remove required durable replay protection, alter SQLite policy or change
any membership/authorization/freshness/lease/ownership decision.

## Checks and review

One exact in-process test uses synthetic durations, a no-trace single-call action,
fixed redacted error values and an injected BrokenPipe writer. It checks disabled
and fast-success silence, failed/slow summaries, exact serialization, error
propagation and non-panicking diagnostic failure. PASS0.00s. Both workspace/all-
target Clippy modes pass with warnings denied2.68s/1.57s; fmt/diff pass. A compile-
only release check of compute-security passes37.87s, covering disabled debug code.
Only basic Mac checks ran, cached/offline assets/max two build jobs. No sockets,
services, process/host inspection, host instrumentation, models or native workloads.

Correctness/security review traces respond/respond_next, both heartbeat paths,
the policy guard lifetime, error/certificate update order and the unchanged counter
commit before eligibility publication. Separate Ponytail complexity review keeps
one direct optional record, shared step helper and bounded formatter; no dependency,
wire schema or observer framework. Independent milestone/release review remains
deferred, not approved. GLY-267 remains open: this identifies the next causal
boundary to observe, and does not claim the initiating Windows failure is fixed.

## Observed response phases and counter correction

Source c8df80cd8487e2c697c13ad672857dfcc0bc58e2, [CI38919953204](https://github.com/Sigbound/glyd-compute/actions/runs/38919953204),
failed Windows backend/aggregate; the other five checks passed. Production's
two-worker request returned503 at daemon.rs894 after177.15s. Worker Transport and
4194304 retained reservation were recorded; registry later returned online.
The coordinator measured registry_update1138ms/816ms, dispatch449ms/700ms and
totals1630ms/1588ms. Those intervals include multiple operations, not fsync alone;
they do not establish the initiating generation/OS cause. Raw Windows job117037003856
SHA256 `2cf18a82d35ab397d4e9d3ebde1c325370403fcab58db9b4cd483c415fb2ab42`
is preserved, with no unchanged-source retry.

Source inspection confirms a durable sample_counter UPDATE on every increasing
resource report. A controlled in-memory SQLite check fails the old implementation
at report2: one write rather than zero inside a reserved range (0.03s). The fix
reserves the report sequence and up to255 successors before publishing capacity.
Increasing reports inside that range reuse the reservation; the entire range is
rejected for capacity after restart. A matching authenticated floor reply lets a
worker recover using a genuinely new probe snapshot above the range. Exact
duplicates keep their age; lower/mutated live repeats stay rejected. Failed or
zero-row reservation writes cannot publish capacity; exhaustion cannot wrap.
Existing DELETE journaling/EXTRA synchronization, TTLs, deadlines, epoch/boot,
revocation, leases, model admission and service ownership are unchanged.

Three exact `registry::tests::resource_sample_windows_` tests pass0.05s with
257 bounded reports, restart/lost-floor recovery, stale/mutated/lower reports,
injected ABORT/IGNORE writes, exhaustion and strict decimal/unknown-field framing.
Final cached warning-denied Clippy modes pass0.15s/0.11s, fmt/diff pass and release
security compile check passes0.16s. Only synthetic in-process/in-memory checks
ran locally, max two build jobs, offline; no sockets, services, native workloads
or host instrumentation. Existing networked restart coverage is updated for CI,
not executed locally. These checks do not measure Windows latency improvement.

Correctness/security review covers all reply constructors, client reply binding,
durable-before-publication ordering, restart tombstones, fresh-capture filtering,
overflow and denied storage publication. Separate Ponytail review: lean already;
existing table/parser/client/sampler, constant-size metadata, no dependencies or
new jobs. Coordinated upgrade is required; legacy strict clients reject the floor,
and downgrade/rollback is not qualified. Full GLY-267 and all native/milestone
acceptance stay open pending actual evidence and independent review disposition.

## Counter source qualification and Windows build duplication

Source cb1bf2c798b55c8c1ad92d77c3c0ca69afa9b7dd,
[CI38923765960](https://github.com/Sigbound/glyd-compute/actions/runs/38923765960),
passes Linux x64/ARM64, macOS, dependency policy and Windows backend. The Windows
backend's eight short controls, production184.97s and recovery57.12s pass; raw
job117049974698 SHA256
`f260732758bcd92ed28c372746ec26aaa80f1ce1011d88d14a44a531d0174156`.
This is source evidence, not a timing improvement or original-cause proof.

Windows default exceeds the unchanged ten-minute job cap (confirmed by the GitHub
check annotation), is canceled and the aggregate fails. Raw job117049974794 SHA256
`5ada732063498a8185d8f122bb9803166b624669048ad5d37bbf85e976afda0a`
is preserved. Its pinned toolchain installation takes125s; successive test builds
take41.18s,85s and25.73s. The priority and remainder security library executables
have different identities. All recorded completed assertions pass, including the
three resource-window controls; cancellation interrupts the CLI daemon suite
after its first two cases. This is incomplete coverage, never a green gate. The
source is not merged or retried unchanged.

Keep the first priority case and remaining non-CLI tests on the same Cargo package
and target selection; their filters remain disjoint. CLI serialization, doctests,
all test selection, warning checks and ten-minute job cap remain intact. A basic
Mac compile-only control confirms15 identical test executable paths, all fresh on
the second selection (0.12s/0.11s cached); list-only inspection confirms exactly
one matching priority case. No network/process suite is executed locally. The
earlier lib-only selection reused its library harness but still deferred normal
library/integration builds; the final selection includes those targets up front.

Cache only the pinned Windows MSVC toolchain directory/update hash using the
already pinned cache action. Both partitions retain unconditional rustup install
and component checks. Backend is the sole writer after successful installation,
before long tests, avoiding competing writes or lost late cache publication.
Keys include OS, architecture and the toolchain file digest. This addresses the
observed install cost without skipping verification or increasing the cap;
actual cache benefit and complete Windows coverage require new-source CI.
Separate complexity review retains Cargo selections/native cache actions rather
than another test runner. Full correctness/security review preserves test filters,
cache scope, required components, feature modes, failure propagation and coverage.
