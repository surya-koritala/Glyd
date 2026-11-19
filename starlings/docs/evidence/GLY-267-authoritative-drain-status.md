# GLY-267 authoritative drain observation

PR136's Windows backend run
[39444824389](https://github.com/Sigbound/glyd-compute/actions/runs/39444824389)
reports a started coordinator drain outliving the unchanged three-second admin
reply wait: `CONTROL_WAIT_EXPIRED phase=executing invitation_step=none`, followed
by `CONTROL_EXECUTION_SLOW elapsed_ms=3414`. A later status says `DRAINING`.
This does not identify which inner I/O or lock operation consumed that time and
does not attribute the archived GLY-267 failures.

The acceptance caller incorrectly required an immediate successful mutation
reply. It now reuses the existing one-shot mutation helper and existing bounded
read-only status wait, with no mutation retry or extended deadline. All four
drain acceptance call sites require `admission=closed` before proceeding.

The earlier `DRAINING` status only described Runtime request intent, set before
the authority's forced clock checkpoint and policy withdrawal. A failed policy
read could leave that intent true without closing admission. Status now reads
the existing authority epoch and draining flag in one Access lock. Worker status
uses its existing registry fence; absent/offline authority remains unknown.
No completion bit, new lifecycle, persistence change or native cleanup proof is
introduced. Forced clock checkpoints, failure handling, reservations and all
control/IPC/fixture deadlines remain unchanged.

The pure file control
`daemon::tests::drain_status_does_not_confuse_request_intent_with_authority_withdrawal`
holds a real SQLite exclusive lock and calls the real Runtime command. That
pre-withdrawal failure leaves local intent `DRAINING` but authoritative admission
open after the lock is released. A later explicit caller-authorized drain closes
it; a missing authority remains unknown. This control starts no CLI process,
listener, socket, service, model or hardware workload. Existing model-free CI
acceptance and lost-reply controls retain the one-shot command policy.

The exact pure control passed in 0.07 s with pinned Rust 1.97.1 and cached
offline compilation (at most two Cargo jobs). Both workspace/all-target Clippy
modes passed with warnings denied (10.77 s default, 4.81 s all features);
formatting and diff checks passed. Process/socket acceptance is left to the
existing platform CI, not executed locally. Commands:

```
cargo test -p compute-cli --bin glyd-compute daemon::tests::drain_status_does_not_confuse_request_intent_with_authority_withdrawal --locked --offline -- --exact
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
cargo fmt --all -- --check
```

Native retained-byte/device cleanup and original M2 investigations remain open.
