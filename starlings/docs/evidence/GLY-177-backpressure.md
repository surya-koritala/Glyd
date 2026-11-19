# GLY-177: integrated production backpressure acceptance

Base: `a8beafcce3b48fae5466df16e688a3d08af64b51`. The existing failure matrix
covered backpressure at shared primitives but lacked a sustained slow reader
through production daemon HTTP admission, coordinator dispatch, enrolled worker
and supervised service. This check is synthetic software acceptance, not a
native workload, performance benchmark or provider qualification.

## Check

`catalog_service_slow_reader_cancels_without_replay_or_releasing_ownership`
reuses the existing production fixture setup and failure/ownership assertions.
Windows runs the scenario in its existing serialized production matrix; other
platforms run one separate test in the existing catalog-service selection.

The feature-gated child sends at most 512 nonterminal SSE events, each with a
16 KiB synthetic content segment. Individual events remain below 64 KiB, total
output remains below 16 MiB and channel bytes/records remain below their bounds.
It emits no successful finish and then stalls. The connected client's receive
buffer is reduced on the unconnected socket with the existing socket2 dependency;
after visible partial
output/session headers it stops reading but keeps the socket alive.

Within the existing eight-second failure observation bound the worker must reach
failed_pins_retained with nonzero reserved bytes. Only after that observation does
the test close the client, so client disconnect cannot satisfy the assertion.
The new scenario's fixture request timeout and terminal stall are thirty seconds;
backend timeout cannot satisfy the earlier observation bound. Production idle,
request, control, readiness and all existing scenario deadlines are unchanged.

Both worker/coordinator admin remain responsive. The worker registry must still
be Online, so expired heartbeat eligibility cannot satisfy the pressure check.
The coordinator has one retained
active assignment/request and nonzero retained body bytes, with no queued work.
Existing common assertions require one backend invocation, incomplete trusted
drain and a retained service launch quarantine. No retry, release or successful
native cleanup is inferred from transport cancellation.

## Review and verification

Correctness/security review traced fixture event framing, event/stream/channel
limits, output authorization, idle backpressure, cancellation and ownership
retention. The held client and longer synthetic timeout distinguish the target
failure from disconnect and ordinary backend expiry. No native route is granted.
Ponytail review reuses current fixtures, existing failure checks and platform
scheduling; no new dependency, executor or CI job is added.

Pinned Rust 1.97.1, locked offline dependencies and warnings denied:

```sh
cargo test -p compute-cli --all-features --test daemon catalog_service_slow_reader --locked --offline
```

The focused local check passes in 13.89 s. Required validation uses the existing
formatting, default/all-feature lint and affected service/daemon selections;
all seven exact-head CI checks and current-main ancestry remain required before
merge. Final PR/head/merge/run evidence is recorded in Linear. Independent review
is deferred under current merge authorization, not claimed approved.

Current README, daemon and adapter guides now describe implemented CPU request
routing separately from provider/native/hardware gates. Old immutable evidence
snapshots remain historical. GLY-244/246's software fixes are complete; their
unverified original timing attribution is not a newly discovered code defect.
M2 stays open for actual package/API delivery and separately approved native and
same-type hardware acceptance. M3 has not started; no model/GPU/PC workload,
download, spending, license/settings change or quarantine reset occurred.

## Linux preparation correction

Head `f5563bcd3556df63e11a6c28dfdaffebf8be2f07`, run38482686828,
Linux x64 job115589076179 and ARM job115589076650 failed the slow-reader case:
worker state remained serving beyond the existing observation bound. The other
nine lifecycle cases passed. These failures are retained, not relabeled passed.

The first preparation changed SO_RCVBUF after connect. The
[Linux TCP manual](https://man7.org/linux/man-pages/man7/tcp.7.html) requires socket
buffer sizing before connect/listen for the connection behavior to take effect.
The corrected test creates an unconnected socket through the existing socket2
API, sizes its receive buffer, then connects and converts it into TcpStream.
No platform-specific kernel option, production change or larger payload is added.
All observation, request, idle and channel deadlines and Online/no-replay/ownership
assertions remain unchanged. The corrected local focused check passes in 14.30 s;
all-feature warnings-as-errors Clippy passes. Latest-head platform CI is still
required before merge; no unchanged-head rerun is used to mask either failure.

Preserved workspace log SHA-256:

- Linux x64: `86286eb6fd9fa922f41d4a22a126fee340d75f5b927b884d9ab59bac0417176a`.
- Linux ARM: `1f106dcbf7866635a12f7c8bfd711ee1a379b78ac8d07ad3bb1cfcc5a70cfa26`.

## Output bound and permission outcome follow-up

Pre-connect receive sizing alone did not resolve ARM job115592076086 in
run38483677700: service state remained serving within the eight-second observation
bound. Production client send queues were still host-dependent, so a small reader
window did not ensure pressure was reached before the bound. Accepted client HTTP
sockets now request a 64 KiB send buffer before admission through existing socket2.
This is a runtime resource bound, not an increased payload or relaxed timer. OS
buffer accounting can add overhead; no native memory benchmark is claimed.

The same run's Windows backend job115592076218 exposed a permissions mutation
whose reply expired while executing. This is genuine fresh control evidence,
superseding the earlier statement that GLY-244 had historical questions only.
The three-second reply wait intentionally permits an unknown started outcome.
The API now exposes coordinator-only `cluster permissions-status NODE`: a bounded
read of the current committed model/action allowlist, with membership/clock/private
state checks. Unknown/revoked nodes are refused; no token/permission is mutated,
secret exposed or expiry extended. Library tests assert empty/unknown and full-list
behavior plus unchanged SQLite mutation counts for a read.

Production tests issue each permission mutation once, accept only success or the
specific started unknown-outcome response, then make one read-only observation
and require the exact expected allowlist before proceeding. They neither resend a
mutation nor accept absent/mismatching policy. A negative query is not evidence
that a pending write cannot finish and never authorizes a retry. No control, IPC,
fixture or failure observation deadline is widened.

All 73 security checks pass locally (one explicit capacity gate remains ignored),
all-feature lint is warning-free, and ten lifecycle cases pass in 34.88 s. Reviews
trace the output bound, all permission callers, coordinator privilege, data limits,
read-only policy behavior and unknown-outcome handling. Native/provider/hardware
qualification and original OS-delay attribution remain open. Final source-head CI
must pass before merge; no failed prior head is relabeled successful.
