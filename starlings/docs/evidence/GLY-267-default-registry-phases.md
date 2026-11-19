# GLY-267 Windows default registry failures

PR109 source da80e37 / CI38954793897 passed all seven checks before merge
d893ae2aea4e8b96b5518c7c596ee36b80e7e497. Its separate automatic main
CI38955946224 fails Windows default and aggregate; the other five partitions pass.
Raw work/m6-pr109-main-windows-default.log SHA256
74b7a29663654826fd26c6ccfdc764e0d064cca0b7cf88e361e272e57938e7df.

Two registry tests return Transport at registry.rs1556/2235, with corresponding
server-thread Transport failures at1548/2231. The reconnect case fails its first
heartbeat, before creating any pressure sockets. Its pressure stage is therefore
not the demonstrated cause. The membership helper's failing exchange ordinal is
not recorded. No initiating syscall, phase or OS/security-software cause is proved.
The rest of the default security harness reports120PASS/2FAIL/2ignored/1filtered.
The earlier publication-reconciliation panic belongs to a passing caught control.

The failure's non-CLI Windows command did not opt into the existing registry
diagnostics. Enable GLYD_TEST_REGISTRY_TRACE for that command only, leaving CLI
and other platforms unchanged. Reuse the existing numeric I/O, static failure
labels and bounded reply-phase diagnostics, which omit identities, addresses,
credentials and payloads. Add the missing client_tls error label around the shared
RegistryClient TLS call, matching its existing connect/send/receive labels.
inspect_err preserves the exact original result and adds no retry or fallback.
Release builds keep diagnostic output disabled.

No error mapping, protocol, native deadline, clock/freshness policy, connection
budget, test assertion, coverage, harness scheduling, job count or cap changes.
Default non-CLI tests retain normal Rust scheduling; backend/CLI tests retain
their existing serialization. README now describes that distinction accurately.
This repairs an evidence gap, not the unproved Transport cause. Keep GLY-267 and
native parents open even if a successor is green. GLY311's separately qualified
logical-lock defect remains Done; its controls also pass in this failed main run.

The earlier documentation-only PR110 source c6c9154 / CI38956086027 all7PASS is
retained and unmerged. Corrected source needs its own complete matrix, fresh-main,
clean/exact-head/review gate and separate automatic-main qualification. No unchanged
failure retry or local socket/daemon/pressure/model/instrumentation test is used.
Separate complexity review reuses existing opt-in diagnostics and one shared
call-site label; correctness review keeps result/privacy and all runtime/test
boundaries unchanged. Independent milestone/security review remains deferred,
not approved; this document supplies no native or release acceptance.

Basic Mac validation: the existing injected-error/reply-rendering test passes
0.00s after3.38s compilation, including disabled/healthy silence, static error
phases, exact error preservation and broken-output handling. No network I/O runs.
Both warning-denied workspace/all-target lint modes pass3.25s/1.52s; both format
checks pass. File-only comparison confirms identical Cargo test commands,
harness-scheduling flags, matrix and caps; all12 multiline shell blocks parse,
and52 relative documentation links exist. Final diff whitespace check passes.

Prior c6c9154 all-seven-success metadata is preserved privately in
work/m6-pr110-c6-before-correction.json, SHA256
34ea52889fa013dbf023096164476947085a739156cbbf2acdf3a94ce5c9cd4e;
its passing Windows-default raw log SHA256
c946c240f3fe2a57f6d465e57418e54b44e07e28aea2de30e2814b05cecc0394.
Neither proves the unrecorded initiating cause of the separate failed main run.
