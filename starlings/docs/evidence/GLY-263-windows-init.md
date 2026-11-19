# Current Windows authority-creation failure

PR86 source ae867347c7dd85d9bea99bcf6166ab9147355da5's
[CI38873796648](https://github.com/Sigbound/glyd-compute/actions/runs/38873796648)
fails Windows backend/aggregate; default Windows and the four other checks pass.
The initialization collision case exceeds its unchanged12-second command watchdog
after entry/state_lock/cache/authority_create. Its short suite is7PASS/1FAIL82.05s;
raw job116883896568 SHA256
5dcd2960a74a63119b67e68d381028360c8b0e74bb094f296b4493e84b0987d9.
The exact collision/recovery branch and inner operation are not recorded.
The separate default lifecycle case passes all four labeled scenarios14.36s,
remaining security106PASS/one existing ignore and daemon10/10 in82.21s;
raw default job116883896359 SHA256
3803138166ae3b0dd63db9635e5b680d5444fd26c92fd691bbcdd5775af0edaa.
These passes do not explain the earlier lost-panic failure.

Reuse the existing invitation observer pattern for Authority::create_observed:
sixteen fixed attempted-operation names delimit directory open, exclusive database
file creation/sync, keys/certificates, private record publication, SQLite setup,
database close and authorization open. No path, key, record or runtime value is
passed to the observer. Normal Authority::create delegates with a no-op; the CLI
prints names only under its existing debug/test-backend opt-in. The enrollment
file still closes immediately after sync, before crypto/SQLite. No error,
durability, exclusivity or cleanup policy changes.

The existing collision test emits static case labels, including cache_recovery,
and requires exact expanded refusal prefixes; the success helper requires all
sixteen inner names in order between the original startup/exit phases.
Untraced stderr remains empty. Local default and development controls each pass
two tests in0.88/0.89s; raw SHA256
0cb94c24c144f588f7b3f41488fa9a11c0062b02edfaf7e6f56f0d352f3bdb8b /
34ff2b1c9e3ac857a831072de035e9f7d82fa39b7f5242690fff739d22c33dbd.
Both workspace/all-target warning-denied lints, formatting and diff checks pass.
A default CLI built with debug assertions disabled contains none of the three
marker families and initializes successfully with empty stderr even with opt-in.
This is compile-gate verification, not optimized release qualification.

Correctness/security review preserves all callers, protected state, error returns
and ownership; the separate complexity review reuses the existing callback
pattern without a new logger, hook framework or dependency. No job, timeout,
test filter or assertion is relaxed. This fixes missing diagnostic detail,
not the still-unknown Windows stall. GLY263 and GLY267 remain open; PR86 is
unmerged until changed-source CI and review gates pass. No model/device workload,
download, spending, settings or old-state reset occurs.

# Cache-boundary refusal and ownership release

Merged main d47b08ac5bbbf7790eea1cf032f6f2f1738c0ced's automatic
[CI38627119567](https://github.com/Sigbound/glyd-compute/actions/runs/38627119567)
passes all seven checks. Windows default10cases pass64.13s, raw job116068978229
log SHA256822683ed74c551f4b17a1ab5600dd231cd3a21aa8cd44147d58f44733840ee41;
backend9cases pass224.38s, job116068978246 log
SHA256d359343f280631d1074ad42b006c618de78b699e6905e167d427116a62815775.
No rerun was requested. This qualifies that merged source, not the unknown cause
of earlier failed runs.

The existing collision test now also places a protected ordinary file at the
configured cache directory path. The real CLI must refuse with exactly the
entry/state_lock/cache attempted-phase prefix and PRIVATE_STATE_REQUIRED, without
stdout, exit-completion markers, private paths or fixture content. It preserves
the obstruction bytes and creates no enrollment.sqlite/config/authority/identity.
Both cache and existing-authority refusals must leave the persistent daemon.lock
file present and permit immediate ownership acquisition after the child exits.
Removing only the isolated cache obstruction then permits a fresh explicit init,
including the strict full eight-marker success sequence, directory/config
publication and subsequent ownership acquisition. No production retry is added.

Reuse the existing child runner,12-second watchdog and test selection; the test
now covers three conditions in one case, with no new fixture hook, production
code, dependency, CI job, test filter, deadline or private-state policy change.
The test preserves the actual persistent lock file; it never unlinks it to obtain
ownership. Only new temporary test state is removed, never prior ownership or
quarantine records.

Local locked/offline warning-denied checks:

- Default collision case passes0.67s; log SHA256301580a159921aa4fae27a9a68aa1b2d3d1c0e3b1b5c460d7764effe425ec833.
- All-features collision case passes0.66s; log SHA2562034294f61c0f63dc42231b1668f45ac02c1805027ebe915e355b14b65136019.
- Formatting and both workspace/all-target Clippy feature modes pass.

Correctness/security review follows init through Paths::cache/PrivateDir and
StateLock, preserves exclusive authority creation and checks refusal cleanup.
Separate Ponytail review extends one existing case with stdlib operations, no
new helper/framework. Independent milestone review remains deferred, not approved.
Final-source seven-check CI and reviewed merge remain required. The original
Windows empty-output stall is still unreproduced and unproved; GLY-263 remains
open. M2 backend-contract/hardware/two-real-worker/benchmark gates remain open;
M3 unstarted. No native inference/model/kernel/GPU/Ubuntu/PC workload or spending.

# Previous controlled initialization investigation

The [GLY-263](https://linear.app/getglyd/issue/GLY-263) primary Windows stall remains
unreproduced and unproved. Main5c9e629's automatic
[CI38620755167](https://github.com/Sigbound/glyd-compute/actions/runs/38620755167)
passes all seven checks, in addition to the reviewed PR64 source. Green runs do not
identify the initiating cause of the retained0913e84/6550a96 failures below.

Found and corrected a diagnostic gap: previous markers existed only with
`test-backend`, so the default debug partition could still fail without a phase.
Reuse the same explicit GLYD_TEST_INIT_EXIT_PHASE opt-in in debug builds as well.
Untraced commands remain silent. The default CLI with debug assertions disabled
compiles out both marker families; no new flag, delay, dependency, private-state
policy, process-exit bypass or CI schedule change is introduced.

Two deterministic real CLI boundaries use existing protected state, not injected
sleep or a new fault framework:

| Controlled condition | Last attempted phase and terminal result | State invariant |
| --- | --- | --- |
| Parent holds the actual StateLock | entry/state_lock, PROFILE_IN_USE | No config/authority/identity published |
| Protected enrollment.sqlite already exists | entry/state_lock/cache/authority_create, SECURITY refusal | Original fixture bytes retained; no config/authority/identity published |

Both commands run under the existing12-second child watchdog. Refusals have no
INITIALIZED output or exit-completion markers, private paths or fixture content.
A third path in the two-test selection checks successful initialization with opt-in
explicitly removed: normal stderr stays empty. The shared child runner retains the
same timeout, kill/wait, stdio files and Windows unit-lock recovery. Successful
traced clients continue requiring the full exact eight-marker sequence. Marker
names describe attempted operations; absence of entry cannot distinguish process
startup from a blocked first stderr write. These checks do not recreate the
archived empty-output timeout or rule out other Windows/host-pressure conditions.

Local model-free checks:

- Default focused boundaries/silence:2tests pass0.62s; log SHA256a9a1f65953d6ecd59e9f34e8e893121e5faeaf952892f3737894dfd48664b389 (before catalog-service test-name prefix only).
- All-features focused boundaries/silence:2tests pass0.64s; log SHA256e50443d1ee0c352fb93cb1dd50e422f64aa1cc697df9e440c307fa2947574f5a.
- Both new tests use the catalog_service_ prefix so existing Windows backend selection runs them as well as the complete default suite; no new job or filter is added.
- Existing default enrollment/drain/restart passes7.96s with complete strict marker ordering and silent non-init commands; log SHA2565dc8de4bac6498a91e6e16e9a1e97c8f4f48ae71802595355fb2aa80dbcb10d0.
- Formatting and both warning-denied workspace/all-target Clippy modes pass.
- [Diagnostics-disabled check](GLY-263-diagnostics-disabled.json): default features, final CLI crate built using cargo rustc -- -C debug-assertions=no; both marker strings absent from binary and explicit opt-in still yields silent successful init. This checks the compiled gate, not fully optimized release qualification.

Correctness/security review traces main/init/shared command callers, opt-in and
compiled exclusion, full marker ordering, protected-state collisions, no sensitive
values, silent commands and bounded child cleanup. Separate Ponytail review reuses
stdlib/existing hooks and one process runner for traced/untraced checks; no generic
fault injector, sampling loop, parser or extra OS diagnostic framework. Independent
milestone review remains deferred, not approved. Exact-head seven-check CI and
reviewed green merge are required. GLY-263 stays open for genuine attribution;
M2 provider/current hardware/two-real-worker gates remain open, M3 unstarted.
No model, kernel, GPU, Ubuntu, native comparison, download, spend or old-state reset.

---

## Earlier immutable failure and correction records

# GLY-263: isolate pre-success initialization and test-lock cascades

PR64/source0913e848057043cefe6c71ec7f001cf9646bb19a failed in
[run38615665182/job116029988781](https://github.com/Sigbound/glyd-compute/actions/runs/38615665182/job/116029988781).
The production catalog case's first cluster init exceeded the unchanged12s watchdog
with stdout/stderr empty. No INITIALIZED or exit marker was emitted; this does not
identify whether the Windows process startup or initial state/authority work stalled.
The later survival case did not run its own command: the prior panic poisoned the
unit-valued CLI_PROCESS test mutex, and its expect caused a second failure.
Raw job log is retained locally, SHA256611164fb2137969a9a0c4eaa3d6f9fa9d85a6cbb33ea41493be04c71c53f1bad.
The earlier GLY253 post-success exit case remains distinct and unidentified.

Recover only the test Mutex<()> with the standard PoisonError::into_inner; it
protects serialization and contains no state to corrupt. Keep exclusive guard
ownership, the12s watchdog, child kill/wait and original timeout assertion. A
controlled Windows test deliberately panics under that mutex, then requires the
following real CLI status client to execute and return its own DAEMON_UNAVAILABLE
result. This rejects the old expect behavior without waiting for an accidental
initialization stall. No production mutex, result, filesystem policy or cleanup
path is bypassed.

Reuse the existing development-backend opt-in GLYD_TEST_INIT_EXIT_PHASE flag.
Emit fixed entry/state_lock/cache/authority_create/config_publish phases before
existing work, then the existing authority_drop/state_lock_drop/resources_closed
phases. Successful traced init clients require the complete exact ordered sequence;
default/untraced commands stay silent. No paths, credentials, keys, extra OS query,
new config, timeout or process-exit bypass is introduced. These markers provide
localization, not proof that the initiating Windows cause is fixed.

Model-free local validation: formatting and both workspace/all-target warning-denied
Clippy modes pass. Existing catalog_service_observation_paths_match_and_record_paired_latency
passes3.83s with strict startup/exit marker ordering and unchanged stable status
fields; log SHA256d66f2988d093545985e4ab569d88a7d9b17fd4c64ba7199d0757e29b83e675f6.
The existing default enrollment/role/drain/restart case passes8.07s with normal
silence; log SHA2569dc16f2aa4124f7f479070ce184c782691b6067699b9365639fe3530aa4e87ac.
The forced-poison regression is Windows-only and must pass in final-source CI;
Mac tests are not Windows qualification. Preserve the original failed run; no
unchanged-head rerun or deadline expansion.

Correctness/security review covers every command caller, serialized ownership,
controlled failure vs real command result, strict marker order, feature/opt-in gates,
redaction and unchanged child cleanup. Separate complexity review uses stdlib
recovery and the existing diagnostic flag with no framework/dependency/job addition.
Independent milestone review remains deferred, not approved.

The [approved native comparison](native/2026-10-14-m1max-current-parity/README.md)
still PASSes20 matching direct/proxy JSON/SSE responses in158.352s/$0. No native
comparison/runtime/model restart is performed for this CI diagnosis. M2 remains
open for installed backend delivery and real same-type worker acceptance; M3 unstarted.

## Controlled Windows result and default schedule correction

Revised source6550a96/run38617695573's Windows backend passes all7 catalog cases
in217.76s, including forced poison followed by a real CLI result. Raw backend job
log125a0898abe5c4fd89bd8cffbbb115679867d98bd29bc384bba44171111914ca.
The default partition fails four independent concurrent fixture initializations
with empty stdout/stderr at12s; no subsequent PoisonError masks later results.
Default job116036921966 raw log290cd2f12c535866b93295e7ff69848b908366ef94a8dff547cdc66c724a9bad.
All four non-Windows/dependency checks pass; Windows aggregate correctly fails.

The default compute-cli package now uses the same serial test-case schedule already
used by Windows backend lifecycle tests. Other workspace packages stay concurrent,
and every normal CLI target/doc test still runs exactly once. Daemons, sockets,
requests and concurrency inside each case remain exercised; only independent CLI
fixtures stop competing. Unix is unchanged; seven checks, coverage, warning gates,
12s client watchdog and10-minute jobs remain intact. Valid Bash and both platform
branches are checked without running models. This is a changed-source scheduling
validation, not proof of the initiating Windows OS/VFS cause from different runner
jobs. Original failures remain failures; no unchanged-head timing hunt.
