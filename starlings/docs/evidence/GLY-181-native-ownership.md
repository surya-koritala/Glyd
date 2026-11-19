# GLY-181 local loaded-instance ownership

Issue: [GLY-181](https://linear.app/getglyd/issue/GLY-181).
PR/commit/CI: recorded with the reviewed source and automatic main result in Linear.

The delivered range-worker contract retains model weights, KV, state and scratch
between chunks and sessions. The core previously knew only whether an invocation
was pending. A local supervisor can now retain an explicit per-lease backend
hold before native preparation. Reservation release waits for both conditions;
step completion alone cannot free a loaded stage. No backend is started here.

`hold_backend` accepts only live prepared layer leases. Its opaque `BackendHold`
uses private Arc identity and has no serialization or automatic Drop cleanup.
Qualified `backend_unloaded` requires prior withdrawal, no pending invocation
and the same current preparation identity; an old/foreign token cannot clear a
successor. Existing qualified whole-backend shutdown clears holds and stops
reuse. Recovery refuses nonempty retained-backend markers; default/empty old CPU
records keep their original bytes. Marker removal alone proves no native fact.

One injected-time, pure lifecycle control covers release, expiry, revocation and
epoch handoff, with and without a pending step; snapshot/reservation retention
after step completion; foreign ownership; exact cleanup repeats; token Drop;
table reclamation and reused lease UUIDs; multi-lease complete backend shutdown;
recovery roundtrip/refusal and old CPU serialization/expiry compatibility.

Commands use pinned rustup 1.97.1, the current lockfile, offline compilation and
`CARGO_BUILD_JOBS=2` with the existing debug cache:

```sh
cargo test --offline --locked -p compute-core --test lifecycle local_backend_ownership_survives_step_completion_withdrawal_and_recovery -- --exact
cargo test --offline --locked -p compute-core --test lifecycle
cargo clippy --offline --locked -p compute-core --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
git diff --check
```

The exact new control passed in 0.26 seconds (cached compilation: 6.43 seconds).
Removing only the `finish_release` retained-backend guard made that control fail
in 0.02 seconds: release reported zero reserved bytes instead of the synthetic
4,194,304-byte lease. After restoring the guard, all 13 pure lifecycle controls
passed in 15.72 seconds. Workspace/all-target Clippy passed with warnings denied
in both default and all-feature modes (10.89 and 4.16 seconds); formatting and
diff checks passed. The existing CI jobs and timeout caps are unchanged.

These commands perform no model, socket, service, process, device, workload or
security instrumentation. No dependency, workflow, role or wire completion
operation changes. The pure control is not installed/native cleanup evidence.
The actual StageWorker/adapter owner, multi-tensor start, durable native recovery,
complete physical-pool mapping and independent/native milestone qualification
remain open. GLY-181, M2 and later milestones are not completed by this increment.
