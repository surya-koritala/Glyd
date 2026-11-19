# GLY-172 storage pressure and cost acceptance

Starting main: `fcf786d90cd46dc37d174e778418a3dce5620e0b`. Native service recovery
is already recorded in [the recovery evidence](GLY-172-service-recovery.md).
This change completes the remaining software storage/cost acceptance without
starting a model, resetting an old quarantine or requiring a new dependency.

Local file seeks, staging hash reads, directory enumeration/deletion/sync and
contained path/open errors previously fell through a network Transport mapping.
They now report Storage. An injected StorageFull read error exercises both the
ordinary hash and end-of-file read paths and fails on the old mapping. Network
Hub reads and import source interruptions retain their separate transport behavior.

An actual POSIX namespace/write refusal is introduced after the protected staging
file opens. Publication fails as Storage, the candidate never becomes ready, and
the already verified pinned manifest remains usable across catalog restart and
orphan GC. After restoring the test directory, explicit import succeeds; only
the new unpinned object is collectible. This is not physical ENOSPC/power-cut
measurement. Existing quota/resume/corruption/legacy/recovery tests remain required.

The CLI test holds a real catalog owner while a second process attempts import,
requires refusal, then releases the lock and checks duplicate import leaves the
same storage usage. Pin/GC/readiness assertions continue in the same user flow.
It uses synthetic bytes, not a native model or a mocked lock.

Storage cost keeps unique whole-shard/selected storage separate from declared
execution residency and a 64 KiB transfer buffer. The added largest-asset staging
reserve reuses the same two-copy bound as import admission; existing cache usage
and verified-object reuse remain separate. Backend peak host bytes remain null
without an exact qualified backend/workload profile; GLY-175 owns that qualification.
No sampled smoke RSS or weight size is silently converted into a general budget.

Validation: formatting and default/all-feature Clippy deny warnings; all 11
catalog tests and the cross-process CLI catalog test pass. Default suite: 123
passed, one native ignored; development: 26 passed, one native relay ignored;
six filtered daemon checks pass in 37.16 s. These are model-free checks, followed by all five
latest-head CI checks and current-main verification before authorized merge.
Exact commit/PR/check outcomes are recorded in Linear. Review covers publication,
checkpoint suppression on storage error, pin/GC locking, recovery ordering and
checked quota/cost arithmetic. Complexity review retains existing helpers and
types; no filesystem framework, global hook, retry or new dependency.

GLY-172's software gate does not certify every filesystem's power-loss behavior
or a general native memory envelope. Those limitations stay explicit. M2 still
needs resource policy/profiles, directed links, real multi-worker routing/benchmarks
and integrated review before M3. Independent review deferred, not approved.
