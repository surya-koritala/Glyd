# Bounded GCP1 parser mutation corpus

Partial software delivery for [GLY184](https://linear.app/getglyd/issue/GLY-184).
The existing four tensor and five control golden records seed2,160 deterministic
cases: each prefix bit, three nonzero XOR masks at32 evenly spaced header bytes,
and16 matched-length truncations per seed. No payload buffer is allocated by the
corpus. Accepted prefixes must satisfy header/payload/type bounds and reproduce
their bytes. Accepted decoded metadata must survive a semantic encode/decode round
trip with the same kind and payload length. Invalid inputs return errors without
panicking. A legal changed identifier can remain valid metadata; parsing alone
never proves authenticity or authorizes execution.

Run the existing suite, without additional tools or dependencies:

    cargo test -p compute-protocol --test data --locked --offline -- --nocapture

Local five-test suite passes0.06s; the corpus records97 accepted and2,063 refused
mutations, raw SHA256
167280f76d3f6b4ab07cbf28dd0303ccbea1b41497e00571cf109e099cb38189.
All-target protocol warning-denied lint, formatting and diff checks pass.
Correctness/security review retains independent golden input and checked semantic
round trips. Separate complexity review uses one test/helper and existing parser,
no RNG crate, new runtime, CI job or timing assertion.

This is a bounded mutation corpus, not coverage-guided fuzzing, a proof of all
possible corruptions, TLS integrity testing, allocation/copy instrumentation,
reassembly/state-machine fuzzing or a performance measurement. GLY184 stays open:
reassembly and resource/copy evidence, analytical8KiB/8MiB/two-tensor workloads,
approved wired-link runs and separated TLS/host-copy/synchronization/scheduling
costs remain required. No link/model speed, native inference or memory qualification
is claimed. GLY183 deployed ownership/cleanup and GLY175 hardware/profile gates
remain open. No model/native/device workload, download, spending or old-state reset.

## Source CI failure and preserved registry errors

Initial signed97df8ee source/PR87
[CI38877174705](https://github.com/Sigbound/glyd-compute/actions/runs/38877174705)
fails Windows backend in the existing
inference::tests::backend_failure_emits_one_redacted_terminal_error_without_retry.
The fixture discards its heartbeat error code, its registry request thread later
returns Transport, and scope teardown panics. CLI library20PASS/1FAIL72.11s;
catalog lifecycle checks do not run. Raw job116894906877 SHA256
9b5dec42376f923c45fd803654a9f28d4c716770281b77c621c3726495e692c2.
This does not attribute the initiating error or implicate the parser corpus.

Keep the original assertions, cadence, deadlines and shutdown ownership. The
heartbeat assertion now includes its bounded Error enum; standard Result::expect
adds fixed request/response context. The existing development test command uses
nocapture so the first panic is visible immediately, matching default/catalog
diagnostics. No test/filter/job/platform or policy changes. The exact local
model-free backend-failure case passes8.91s, raw SHA256
2f8f85f0a12f42d9a74b0ff3eb28bd9a98147a961f778b42d5466860a100abb4;
both workspace warning-denied lint modes/fmt/diff pass.
Separate complexity review replaces panic closures with Result::expect.
This fixes an evidence gap, not the unknown Windows failure; GLY267 remains open.

PR86 automatic main32482b5 independently passes all seven
[CI38876998823](https://github.com/Sigbound/glyd-compute/actions/runs/38876998823)
checks. Main's passing run does not erase PR87's separate failure or establish
historical causes. Source PR87 remains unmerged pending changed-source checks.
