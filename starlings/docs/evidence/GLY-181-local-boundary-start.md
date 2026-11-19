# GLY-181 local complete-boundary invocation

Issue: [GLY-181](https://linear.app/getglyd/issue/GLY-181).
PR/commit/CI: recorded with the reviewed source and automatic main result in Linear.

The trusted local `Worker::start_local_chunk` consumes complete ordered boundary
metadata under the current opaque `BackendHold`. It reuses core session,
sequence, invocation, lease, pending and retirement state rather than flattening
multiple tensors into the legacy single-tensor wire `StartStep`. No backend,
buffer, schema, authority or transport implementation is added here.

Every original first-frame template validates before the one invocation state
change. All templates must share the same complete scope, phase, position,
token count and boundary profile, with complete ordered indices/counts and a
checked aggregate scratch budget. Authentic reassembly/layout proof remains the
secured local owner's responsibility. Original 8 MiB wire frame metadata is
preserved even for completely reassembled tensors up to 64 MiB.

One pure injected-time control exercises two BF16 boundary tensors, malformed
second tensors and atomic refusal, scope/order/count/phase/extent/overflow and
aggregate-budget errors, foreign holds, unchanged single-tensor wire refusal,
exact pending output checks, duplicate/past invocation rejection, completion,
revocation/expiry, retained loaded-instance budgets and two 16 MiB reassembled
tensors with valid 8 MiB first-frame templates. First-stage token input follows
the existing `data_plan` direction from the final stage; wrong-direction input
is refused before mutation. No outgoing sample-to-new-decode identity mapping
is introduced, and no tensor body is allocated.

Commands use pinned rustup 1.97.1, offline/current lockfile,
`CARGO_BUILD_JOBS=2` and the existing debug compilation cache:

```sh
cargo test --offline --locked -p compute-core --test lifecycle local_complete_boundaries_start_once_with_retained_native_ownership -- --exact
cargo test --offline --locked -p compute-core --test lifecycle cancelling_inflight_work_retains_memory_until_acknowledged_and_cannot_revive -- --exact
cargo clippy --offline --locked --workspace --all-targets -- -D warnings
cargo clippy --offline --locked --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
git diff --check
```

The new control passed in 0.05 s (1.17 s cached compilation); the existing wire
cancellation/retained-memory control passed in 0.02 s. Both workspace lint modes
passed with warnings denied (3.91 s default, 2.49 s all features); formatting and
diff checks passed. The new control caught the initial local-step state remaining
`Admitted`: trusted local start now enters its running phase before publication,
while the legacy wire path keeps its existing behavior.

The selected control runs only pure model state and header validation, without
files, sockets, services, subprocesses, host/device inspection or model/native
work. No new dependency, workflow, timeout or permission is introduced. The
actual secured enqueue/output owner and backend processes remain separate
integration/qualification work. A core call or a provider completion receipt
does not attest native drain or complete GLY-181, M2 or any later milestone.
