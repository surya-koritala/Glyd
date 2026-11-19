# GLY-182 — node-local stage admission

Issue: https://linear.app/getglyd/issue/GLY-182

The original ten-control/96-library revision is preserved below. The latest
socket-wait correction, eleven-control/97-library results and Windows failure
disposition are recorded in the final section; earlier results do not qualify
the changed source.

The remote data path now uses a coordinator-issued immutable client/plan/lease
owner, authenticated plan assignment, short-lived route renewal, enrolled stage
TLS and the shared GCP1 parser/admission/delivery path. Device runtimes need no
authority database, CA signing key or client token. See
[the protocol and limits](../stage-delegation.md).

Ten controlled tests cover request-start expiry, no revival, immutable issuer/
client ownership, any-stage revocation, challenge/scope/size failures, malformed
assignment, local release/expiry/owner loss, exact local-lease isolation, data
cancellation and retention of actual in-flight logical reservations until trusted
drain. Two child processes each use real registry, pinned assignment, lifecycle,
grant, TLS and data APIs; all four unchanged golden payload dtypes and receipt
frames pass without changing execution state. This is model-free process/API
evidence, not native execution, OS sandbox isolation or hardware qualification.

Local checks on pinned Rust 1.97.1:

```sh
cargo fmt --all -- --check
RUSTFLAGS='-D warnings' cargo clippy --workspace --all-targets --locked --offline -- -D warnings
RUSTFLAGS='-D warnings' cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
RUSTFLAGS='-D warnings' cargo test -p compute-protocol -p compute-core -p compute-security --locked --offline
RUSTFLAGS='-D warnings' cargo test -p compute-cli -p compute-adapters --all-features --lib --bins --test cli --locked --offline
RUSTFLAGS='-D warnings' cargo test -p compute-cli --all-features --test daemon catalog_service_ --locked --offline -- --nocapture
RUSTFLAGS='-D warnings' cargo test -p compute-security --lib authorization::stage --locked --offline -- --nocapture
```

All checks pass. Security: 96 library tests and 11 catalog tests; the existing
manual capacity check remains ignored. Core/protocol and backend/CLI regressions
pass; all 18 catalog daemon cases pass in 71.63 seconds. The final ten stage
controls pass in 3.83 seconds after moving grant issuance after both process
workers prepare. No request cap, precision check, deadline, test or platform was
removed or relaxed. No dependency, CI job or native workload was added.

Raw local log SHA256:

| Check | SHA256 |
| --- | --- |
| Core/protocol/security | 12066252e46cab20447bcf65a032377584522672ec33778579a2f235b8274068 |
| Backend/CLI | a95f58e5033fac88ce1de77cc618b0329e709fcf9b3516ec59d0623a630c8eb7 |
| Catalog daemon | 96c085922e77e8dc17941bd5573dbb008cca7841321b5838c47cd342167a42ce |
| Final default lint | 2715c494a9a7582460b05963c17a44e3b7e861f26214cabe80c87028c244a344 |
| Final all-feature lint | 4cf86278c1e6d659943a46a3d7e3da429ac15be90f0f17742e24ef85228e7d3e |
| Final stage controls | 821c79422f2d469d75690ece2964f52a8322ccb565cfedb7dfe7544a01cabf2c |

A controlled release test first demonstrated that the draft permission could
outlive its local lease; the runtime/lease/owner checks fix that cause. Other
initial failures were test setup/compiler errors: a missing issuer constructor
argument, a nonexistent listener-permit method, an admitted-versus-queued
assertion and a four-token golden vector paired with a two-token prompt. Tests
now use the actual pre-delivery state and a four-token synthetic prompt while
preserving the golden bytes and existing strict input checks. No historic M2
failure is attributed from these passing results. Failed local records remain
preserved alongside the verified records.

Correctness/security review covered client substitution, full epoch/incarnation,
lease/route/peer ownership, expiry before/after network waits, allocation and final
delivery, lock order, poison/replay behavior, and retention until local drain.
Assignment errors close control. A response cannot renew a different runtime
owner. Native execution operations remain refused by the new actor.

Separate complexity review kept two concrete policies in the existing peer/data
stream, without a new transport trait, parser, cryptography, dependency, executor
or retry framework. Existing registry, TLS, core lifecycle, strict JSON and golden
vectors are reused. The receive path and public whole-request behavior remain
shared; targeted tests check authority-only methods refuse delegated streams.

GLY-182 stays In Progress pending the complete deployed pipeline integration
record. GLY-183 flow/reassembly/ordering and GLY-181/185/186/187/provider235/236
transactional/native stage preparation and daemon activation remain open. No
production stage listener, native allocation/drain, link performance or milestone
qualification is claimed. Exact-source CI, reviewed merge and automatic main
verification must be recorded in Linear before claiming this revision integrated.
Independent milestone review remains deferred, not approved.

## Review correction — isolate input errors from authority withdrawal

A strengthened local-lease control reproduced that a rejected input for another
lease also withdrew the original valid lease (failed raw SHA256
`c1cb35b66f530d85770aeadbe7b4d7296580416bdf980da48e22d49eb681142a`).
The shared runtime gate now distinguishes authorization failure from callback
input failure. Only authority loss closes the grant and withdraws its lease;
invalid data still fails/poisons its stream. The control verifies valid work
remains admitted after input rejection and an unrelated lease remains admitted
even after actual grant expiry cancels the original lease.

Both warning-denied workspace/all-target lint modes and all security regressions
pass after this correction: 96 library tests (one existing manual capacity check
ignored) and 11 catalog tests. Raw security SHA256
`e3f42993b2f8c8042bdce42372809d63c29858d00f03e425aa46c4c2f966a236`;
default lint `488d47cca2a7c77c96093645a1b2cac004a1d2e8006b1f57ecacc30b64bda4b8`;
all-feature lint `f957b3b0df661a37f338dc310c3b94a69fe1f0f371acaf61cc2a9d098e6f6082`.
Earlier results above remain preserved. The changed source requires fresh CI;
the initial PR source run cannot qualify this correction.

## Review correction — renewed authority during blocked data I/O

A live renewal test reproduced that a waiting read still expired at the old
grant deadline after successful coordinator renewal (failed raw SHA256
`6abae85cbb4722aef9fd9c2e3c561fd68fa904c2130cad9c2d427e20d8440234`).
The existing deadline socket now rechecks current grant authority at raw TCP
timeouts and continues only inside the original absolute record deadline. TLS
state and already consumed bytes stay intact; no application request, frame or
backend operation is retried. Unrenewed/closed authority still fails immediately
on the next gate. The control also proves a short absolute record deadline is
not extended by a freshly renewed grant. TLS handshake bounds stay unchanged.

Final full local core/protocol/security checks PASS, including all eleven stage
cases, 97 security library tests and 11 catalog tests. Both warning-denied lint
modes, backend17/CLI21+10+7 and all18catalog lifecycle cases PASS (70.70 seconds).
The existing manual/native ignores remain unchanged. Raw SHA256:

| Check | SHA256 |
| --- | --- |
| Full core/protocol/security | dd58c29eba757606327925e7abb7d6390c10df83de38c74064871629d24e0790 |
| Backend/CLI | 7b3fecf4f2aedacde34655c11e57b945783d9b1152ed0ed357f3f4e282d706b3 |
| Full catalog daemon | 317bf49649f7878db5e7606b7aa1518eaba46bc4ddefa60c6ae139080664fbdc |
| Final default lint | af020795905b9f724fd28dbbdf35f21ad0795bc942eeff098397e747898c4e7c |
| Final all-feature lint | 8319a227ecb0b8556c02df25fef77e72c0a218aab7623299941fb75813566199 |

Superseded source efaf95b/CI38861433000 FAILED Windows default job116844509437:
the existing multi-listener topology test reached its unchanged nine-second
initial-online limit with registry=offline/epoch=1. Windows default daemon9/10
passed; all stage controls there passed. Five other checks passed; the Windows
aggregate failed with default. Raw failed job SHA256
`3f78d2ff24d2659799dc60a9adca6355eae9ad46c24d12b0533ef24ccdd0f5e7`.
No initiating socket/clock/SQLite cause is established; GLY-267 remains open.

The test harness previously compiled panic context only with test-backend, so
default CI lost worker/coordinator logs and route outcomes before cleanup. The
same failure-only helper now runs in both modes and includes topology. Its
existing control additionally checks the topology response and all three refused
queries after shutdown. Local default topology PASS6.13s (raw
`91919eda022c758338aa92ed0f3bca3eaf2bbc79ad4ee3bdcc6ba391a1dabc48`);
controlled observation/unwind PASS4.18s (raw
`92d3afe15967a8541cb1088e4949502c8addd8c9a7dc7b4ee962cf8c460df7a5`).
Its initial two-versus-three response assertion failure is preserved, then the
assertion was extended for the added topology result without removing checks.
Passing local controls do not reproduce or explain the Windows initiating cause.
No timeout, readiness rule, retry, platform, test or warning policy is relaxed.
The changed source still requires fresh exact-head CI and review before merge.
