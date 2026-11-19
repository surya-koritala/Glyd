# Memory collection and registry contact

[GLY-268](https://linear.app/getglyd/issue/GLY-268) tracks a controlled defect
separately from [GLY-267](https://linear.app/getglyd/issue/GLY-267)'s archived
Windows failures. On main `17470da7c5160af0385bad1d5e1ab9b6513d15ab`, a worker
calls the host-memory API synchronously before `RegistryClient::heartbeat`.

The regression uses real enrolled coordinator/worker binaries and loopback TLS,
with no service, model or generation. A development-backend-only file barrier
pauses the existing memory call before it returns a capture. Against the existing
synchronous ordering, the worker remains offline at epoch zero while both daemons'
protected admin endpoints respond. The unchanged online observation fails after
11.11 seconds. Raw local log SHA-256:
`0edb1644dacf15762462edff70b3330c1c63a29188f9ef1e77654d07b7f1c8ca`.
This is direct evidence of collection blocking registry contact, not evidence
that a particular Windows host API caused any archived intermittent failure.

The correction uses one owned sampler thread per worker and one snapshot slot.
Collection runs outside its mutex. Heartbeats read the retained snapshot without
waiting for collection, using the original capture ID and age. A missing/stale
capture produces no sample. No resource, backend qualification or extra memory
reservation is inferred. The sampler runs at most once per second and joins via
the existing owned-worker shutdown path; it cannot run model code.

The same regression requires online contact beyond the unchanged six-second
registry fence while the collector is paused, with null coordinator resources.
After explicit release it requires fresh telemetry with no backend capability.
Another pause must let the cached observation expire while registry contact
remains online. The development pause has a 30-second cap and an expiry marker;
the test rejects a silently expired pause. All files belong to its new TempDir.
Default builds have no pause environment lookup or marker creation.

The initial corrected contact/release control passes in 12.32 seconds. The full
18-case local catalog lifecycle, including the extended stale-cache regression,
passes in 55.02 seconds; raw SHA-256
`7534fdf945d5d75f919acd523d9dbffd3bfd0a403e4275d4bff979911a283361`.
The existing default-build real-process registry restart, partition, drain and
revocation case passes in 26.94 seconds; raw SHA-256
`69b00d9becf78d99acb234270c178b05802003c23dc0287d722f4c429f2a4957`.
Commands use the pinned toolchain, locked/offline dependencies and warnings as
errors: `cargo test -p compute-cli --all-features --test daemon catalog_service_`
and `cargo test -p compute-cli --test daemon registry_process_restart_partition_drain_and_revocation`.
Formatting and both workspace/all-target Clippy feature modes pass. Correctness/
security review checks timestamp preservation, absent-capability refusal, lock
scope and owned shutdown; separate complexity review reuses stdlib Arc/Mutex and
the existing JoinHandle list. Final seven-check CI/merge evidence is recorded
with the PR and in Linear. No old failed source is rerun or relabeled.

The sampler cannot cancel an OS API that never returns. Admission/output drain
before waiting for owned threads, but final process exit may still wait for that
API. This change does not make OS scheduling hard real-time or prove the original
GLY-251/253/263/267 causes. It changes no fence, authorization, lease, watchdog,
request retry, cleanup receipt or quarantine policy. Independent milestone review
remains deferred, not approved. GLY-238 and native hardware/two-worker benchmark
qualification still gate M2; M3 has not started. No native/model/GPU/PC/Ubuntu
workload, downloads, spending or old state recovery is performed.
