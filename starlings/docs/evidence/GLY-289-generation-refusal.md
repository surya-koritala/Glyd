# GLY-289 — Controlled generation refusals and partial initialization

Parent investigations: [GLY-267](https://linear.app/getglyd/issue/GLY-267)
and [GLY-263](https://linear.app/getglyd/issue/GLY-263). Baseline main:
`324ff02d626043e9ae0757dc46c3bab478c9f6c4`, whose automatic
[CI 38674893208](https://github.com/Sigbound/glyd-compute/actions/runs/38674893208)
passed all seven checks.

The archived `Expired` generation errors do not distinguish a local lifetime,
clock/certificate check, worker fence or coordinator registry/policy refusal.
The controlled check injects four worker guard states on loopback sockets and
changes the bound epoch on a real enrolled coordinator TLS pair. It requires
exactly five static first-refusal records with opt-in, empty stderr without it,
and permanent channel closure even after restoring local limits. It neither
changes the OS clock nor sleeps until a real credential expires.

With the new test and unchanged baseline production code, the enabled assertion
fails: expected five records, actual stderr empty (0.45 seconds). Raw log SHA-256:
`b586dba340760376ac2cf9fc3e70fd51a778cf9b11feb92b86303d16cfa9e1eb`.
After adding phase labels to the existing checks and atomically claiming the
first close, the identical test passes (0.45 seconds). Raw log SHA-256:
`2bcff1b9af0a2e370598e6072904645b234a4fbe63f8ee3216ff7ed0536999a5`.
These establish the diagnostic gap and its correction, not an archived Windows
OS/queue cause or a performance improvement. The test is one normal security
unit case; it adds no CI job, dependency, filter or deadline.

The existing initialization collision case now also starts with an exclusively
created `authority.json`. Refusal must preserve its original bytes, release
process ownership without unlinking the lock, publish no configuration,
coordinator identity or ready marker, and retain only the empty database created
before the authority-file refusal. Existing lock/database/cache collision checks,
cache-obstruction recovery, full success phase order and untraced silence remain.
This is a later partial-publication boundary, not reproduction of the archived
empty-output initialization timeout. Partial state is never deleted or retried.

Reproduction uses the pinned toolchain, locked offline dependencies and warnings
as errors:

```sh
cargo test -p compute-security --locked --offline generation_first_refusal_diagnostics -- --nocapture
cargo test -p compute-cli --test daemon catalog_service_initialization --locked --offline -- --test-threads=1 --nocapture
cargo test -p compute-cli --all-features --test daemon catalog_service_ --locked --offline -- --nocapture
```

The existing synthetic daemon fixture opts into the new trace, so failure-only
actor context can retain its first gate refusal. Records contain no node/model
identity, paths, credentials, prompt, output or tensor data. The redacted public
error, authorization order, TTLs, deadline comparisons, socket shutdown and
reservation/drain rules remain unchanged. Explicit cancellation and I/O failures
can close a channel before a gate fails; the trace makes no claim for those cases.

Local formatting, both warning-denied workspace lint modes, the security suite,
development relay/library/binary/CLI checks and all 18 catalog lifecycle cases
pass (catalog: 59.16 seconds). The two default initialization cases pass in
0.76 seconds; the changed security library also compiles with debug assertions
disabled and warnings denied. Raw final default/disabled-build log SHA-256:
`6a6a76fe2f247782717f1fc28ebd99c84550bb288e5de7f1c1f7c1a1db4e4c16`.
This is a diagnostics-disabled library build, not optimized release qualification.

Correctness/security review checked all close/check callers, unchanged predicate
order, atomic first-close publication, permanent closure, redacted values and
lock release before diagnostics. Separate complexity review retained direct
static phase labels and reused the existing guard, fixture and collision case;
no new abstraction, dependency, background worker or OS query was added. All seven
exact-source CI checks and the current-main gate are still required before merge.
Independent milestone review is deferred,
not approved. GLY-263/267 remain open until matching initiating evidence is proved.
No native/model/GPU/Ubuntu workload, download, spending, old-state recovery,
quarantine reset, license or repository setting change occurs here. GLY-238 still
requires the provider contract; M2 is incomplete.
