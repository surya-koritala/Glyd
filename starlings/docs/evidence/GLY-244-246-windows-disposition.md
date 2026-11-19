# GLY-244 / GLY-246: Windows investigation and unresolved disposition

Base main: `ff8d512cfed989a144abdca0fe8dc90766664ee0`.
This review keeps the invitation and service-startup findings separate. Neither
historical root cause is established by subsequent successful CI.

## Evidence reviewed

| Record | Established fact | Missing information |
| --- | --- | --- |
| GLY-244, corrected PR #41 head `296e67ca`, [run38341767929](https://github.com/Sigbound/glyd-compute/actions/runs/38341767929) | Warm-up's invite-unapproved mutation returns CONTROL_UNAVAILABLE; production matrix passes | No invitation subphase; commit outcome not established by file presence |
| GLY-244, Windows split failure | A later invite-startup-crash reports CONTROL_WAIT_EXPIRED phase=executing | No invitation_step; cannot distinguish parent directory, clock, policy lock, database, private-file publication or commit |
| GLY-246, head `488022ad`, [run38380035001](https://github.com/Sigbound/glyd-compute/actions/runs/38380035001) | Production case exceeds its fixture readiness bound while loading, with pin retained and null startup_error/properties; warm-up passes | No startup subphase; record cannot distinguish validation/hash, spawn, endpoint publication or incomplete probe |
| Main `b73f9c2`, [run38388849046 attempt2](https://github.com/Sigbound/glyd-compute/actions/runs/38388849046/attempts/2) | Production/warm-up cases pass; publication-phase timeout line is from started_admin_timeout_keeps_an_unknown_outcome | That unit deliberately stores phase 16; it is not a real invitation/filesystem-stall reproduction |
| Main `ff8d512`, [run38468224363](https://github.com/Sigbound/glyd-compute/actions/runs/38468224363) | All required checks pass | A passing run supplies no new failing phase and cannot prove either old cause |

Preserved workspace log SHA-256:

- `m2-fixture-windows-corrected-failure.log`: `42126c405f6ef40e20421a35be5c7d488f279b85558aee6deedfd8753709fb80`.
- `m2-probe-windows-split-failure.log`: `4321ed6a372017df029594ecde669bb110fdc3bde2885bb1752b29f26b61835a`.
- `m2-properties-windows-failure.log`: `5a0f301efbdb16381d704e1d5faa43dc3f4390df2b701ceee16f21db9f0057d5`.
- `m2-windows-diagnostic-attempt2.log`: `6ca0a5968c98d54e8e85680a232efe8bd5a0738e47d15da80be4b057e06fcfe3`.

The raw failures remain unchanged; no failed result is relabeled passed.

## Demonstrated diagnostic corrections

The timeout reporter read the shared invitation stage twice. A transition between
reads could combine a phase from one moment with a step from another. One atomic
snapshot now supplies both labels. Queued expiry and started unknown-outcome
semantics are unchanged.

Service startup's generic spawn phase covered configuration/path validation,
executable hashing, process creation and the synthetic child's endpoint report.
The existing observer pattern now exposes static labels for those steps:
configuration_digest, configuration_paths, executable_digest, key_validation,
process_spawn, and fixture_directory/fixture_endpoint for development fixtures.
The supervisor also exposes key_generation. No paths, arguments, keys, model
contents or response bodies reach the observer.

Original validate/spawn callers retain their APIs and use the same implementation
with a no-op observer. Every configuration and executable check remains in place,
including original and transformed fixture configuration validation. The observer
must remain nonblocking and cannot grant readiness or native qualification.
Synchronous filesystem/process calls remain subject to OS scheduling; these labels
do not create a hard execution deadline or preempt a blocked syscall.

The runnable model-free check rejects a wrong approval digest and wrong executable
digest before process creation, verifies the actual synthetic startup phase
sequence, then requires the existing authenticated readiness probe. It introduces
no delay, new dependency, workflow or CI partition.

## Disposition

**Both historical timeouts remain unresolved.** The phase consistency correction
and visibility improvements are implemented; they do not prove the original
Windows cause fixed. Earlier queued-expiry and shared readiness-I/O fixes remain
separately demonstrated by their own regressions.

No mutation is retried. An ambiguous invitation can be inspected with the existing
authorized invite-status command; a negative/error result does not authorize
reissuance. No user quarantines, pins or consumed approval markers are changed.
The three-second control wait, five-second IPC bound, two-second readiness I/O
budget, service/fixture/request bounds and ten-minute CI caps are preserved.

If another genuine Windows failure occurs, capture its exact head/job and first
failed phase, distinguish intentional timeout unit tests from real daemon failure,
and reproduce the identified operation under the existing bound before changing
its behavior. Do not schedule repeated successful CI runs to search blindly, claim
filesystem/antivirus contention without evidence, or enlarge deadlines.

GLY-246's reviewed unresolved-disposition option is satisfied by this report and
linked final-head evidence. The finding itself remains open; GLY-244's identify/fix
item remains unchecked. This is no approval to defer required M2 acceptance.
Native/provider/hardware gates and M3 ordering are unchanged.

## Review

Correctness/security review traced invitation dispatch/expiry, all validate/spawn
callers, development fixture transformation, supervisor status and readiness
publication. Observers carry static labels only, preserve refusal/approval hashes
and cannot bypass authentication or resource ownership. Separate complexity
review reuses the current callback pattern and preserves ordinary APIs; no tracing
framework, thread pool, retry layer or runtime dependency is added.

## Local verification

Pinned Rust 1.97.1, locked offline dependencies and warnings denied:

- Formatting and all-feature workspace/all-target Clippy pass.
- Existing adapter/CLI library, binary and CLI selection: 37 passed, one native
  relay ignored. The new observed-startup check alone took 0.54 s.
- Existing catalog-service daemon selection: nine passed in 37.81 s.
- All seven exact-head CI checks and fresh-main ancestry remain required before
  merge. Final commits/run links are recorded in Linear. Independent review is
  deferred under the current merge authorization, not claimed approved.

No native/model/GPU/PC workload, downloads or spend occurred.


## Controlled cause reproductions (GLY-244 / GLY-246)

Base: `a85e8c2dd8b937bccfa19a3a580846867df8a5cc`. These tests expose real
source defects independently of runner timing. They do not prove either was the
only cause of the archived Windows failures above.

- `paused_invitation_publication_cannot_block_policy_maintenance`: pauses the
  actual private-file publication boundary with a bounded channel, then calls
  real `Authority::maintain(0)`. Before removing the spanning policy guard and
  SQLite transaction, maintenance failed its one-second bound (test failed in
  1.08 s). After the fix, maintenance completes while publication is paused and
  the released invitation commits successfully. The barrier is released and
  threads joined before asserting the result.
- `split_endpoint_report_never_selects_a_valid_partial_port`: writes a real,
  incomplete fixture endpoint report `127.0.0.1:4`, the valid prefix of
  `127.0.0.1:49152`. The old reader selected the wrong port; the test failed
  immediately. The fixture now terminates its report with a newline and the
  parent waits for that terminator before parsing. Complete reports work;
  zero ports, non-loopback addresses and trailing data remain refused.
- `publication_rechecks_drain_and_capacity_before_commit`: changes real policy
  during publication, requiring refusal and an inactive retained file when
  drain begins or all 128 slots are taken before commit.
- `paused_invitation_keeps_status_available_and_stop_remains_latched`: holds
  the daemon's publication worker, checks a second publication is refused
  without starting, runs sibling status through the actual admin dispatcher,
  then releases/joins the job and checks completion and the existing stop latch.

Invitation work uses one standard-library thread with an atomic in-flight slot.
Its shared issuer contains public identity/CA certificates and policy access,
never coordinator/node signing keys. Queued expiry still uses the original CAS;
started work retains unknown-outcome semantics after caller timeout. No mutation
retry, new dependency, executor, CI job or longer deadline was added. Private
filesystem publication is outside the policy lock; other synchronous OS/database
operations are not claimed preemptible. Shutdown retains the owned job until it
returns and denies late commits through shared drain state.

Correctness/security review checked public-only issuer contents, all invitation
callers, transaction ordering, capacity races, drain, partial report framing,
thread ownership and timeout semantics. Ponytail complexity review removed the
redundant test-only Authority forwarding method and reused existing reply/stage
and worker-reaping paths. Independent review remains deferred, not approved.

Run with pinned toolchain, locked offline dependencies and warnings denied:

```sh
cargo test -p compute-security --lib paused_invitation_publication --locked --offline
cargo test -p compute-security --lib publication_rechecks --locked --offline
cargo test -p compute-adapters --lib split_endpoint_report --locked --offline
cargo test -p compute-cli --bin glyd-compute paused_invitation --locked --offline
```

An initial broad local run was blocked by sandbox loopback permissions; its
PermissionDenied results are not application regressions. A later host-permitted
run caught and corrected a test expectation: Reply::error carries REFUSED as its
code and INVITATION_BUSY as its message. Full final validation and exact-head CI
links are recorded in Linear. These model-free checks do not qualify native
inference, provider service hooks or hardware acceptance, and spend is $0.

Final local results: formatting, default and all-feature workspace/all-target
Clippy pass with warnings denied. Default workspace suite passes (including
72 security tests; the explicit 63-peer capacity gate remains ignored). Final
maintenance regression passes in 0.07 s; development adapter/CLI selection has
38 passes and one native relay ignored; nine catalog-service lifecycle cases
pass in 36.20 s. The controlled regressions run in existing CI selections.

## First-head CI and controlled registry-maintenance correction

PR #54 head `397411c02a537a388d582a8f3621afd2a5584cd5`,
[run38474080295 Windows backend job115560297486](https://github.com/Sigbound/glyd-compute/actions/runs/38474080295/job/115560297486)
failed `backend_failure_emits_one_redacted_terminal_error_without_retry` at
`count_prompt`, with `Expired` before its intended backend-crash assertions.
The endpoint-report regression passed. No same-head rerun was used to mask this.

Three relay tests omitted the existing `while_maintaining` helper during fixture
setup and execution. Their generation channels require live six-second registry
observations. A controlled setup pause of TTL + 100 ms reproduced the same
`Expired` error before the fix (one test failed in 7.20 s). Running those cases
inside the existing maintenance helper keeps actual heartbeats/policy maintenance
live; production TTL and request/channel bounds remain unchanged.

The existing beyond-TTL pause and Online assertion were moved from the semantic
maintenance test into the backend-crash case, before client/count admission.
This strengthens that failing case without adding another delay. All crash,
redaction, no-retry, count refusal, prompt/context binding and memory-ownership
assertions remain. All 11 relay-library checks pass locally in 8.36 s, with one
native acceptance still ignored. Review traced every fixture service caller and
reused the existing helper; no runtime change, dependency or CI job is introduced
by this correction.

Preserved workspace raw logs:

- `m2-controlled-windows-failure.log` SHA-256:
  `23c2793fd5ac05a810d4756f94b50b1901e051a75eea646b800fc78107f20e61`.
- `m2-controlled-idle-registry-red.log` SHA-256:
  `3e77caeb7b02ddb16cec5ef0e76a115adde1ba43d80c051d729cd5efa196521f`.

This newly reproduced fixture-maintenance cause is separate from attribution of
GLY-244/246's archived invitation/loading records. Exact latest-head CI remains
required; the failed first-head result is retained as failure evidence.

## Genuine Windows invitation clock phase

Corrected head `77ddaff22a5f228e9c1855318af8053984aff60e`,
[run38474854962 Windows backend job115562910719](https://github.com/Sigbound/glyd-compute/actions/runs/38474854962/job/115562910719)
passes the endpoint-report and relay checks, then its real warm-up daemon
invitation fails the original three-second control wait at `invitation_step=clock`.
This is not the intentionally injected phase-16 unit test. The production route
lifecycle case passes; the failing operation is invitation clock processing.
The log does not identify a particular slow filesystem/SQL syscall or its OS cause.

Issuance previously forced a durable clock checkpoint before publication and
again afterward, each outside the eventual invitation-row commit. It now samples
and verifies wall time through the existing policy guard, then commits the maximum
observed clock floor with the token row in one immediate transaction. No valid
token can commit without its rollback floor. This removes separate synchronous
checkpoint transactions and redundant guarded filesystem/database checks, while
preserving clock rollback refusal, expiry, capacity and drain validation.
Hard transaction storage failures reuse the existing active-grant/connection
fail-closed path; Busy is a refusal, never an automatic mutation retry.

`failed_clock_floor_update_rolls_back_the_invitation_row` injects a real SQLite
clock-update failure. The published file remains for inspection, but its token
row is rolled back and cannot be active. This verifies the new atomic failure
boundary; it does not simulate or claim a specific Windows disk-delay cause.
Local security selection has 73 passes and one explicit capacity gate ignored;
nine daemon lifecycle cases pass in 36.99 s. Final-head CI is still required.

Preserved `m2-controlled-windows-clock-failure.log` SHA-256:
`110425f3b2f8f2d94fe395f6d67732374e179dd13e293aa7a50db7b718c59687`.
The failed second head remains failure evidence. No deadline widening, assertion
removal, same-head rerun, native/model workload, new dependency or CI job occurs.
