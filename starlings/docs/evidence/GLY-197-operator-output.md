# GLY-197 completed-command output failures

Issue: [GLY-197](https://linear.app/getglyd/issue/GLY-197).
Base: qualified main `d64e90ceb3076c2a5430c3f2b80b4e0259e74319`.

The shared successful admin acknowledgement and durable init, prepare, join and
routes acknowledgements now use one fallible standard-library writer. Successful
text and exit behavior remain unchanged. Write or flush failure returns
`OUTPUT_UNAVAILABLE: command completed but output failed; inspect state before retrying`
rather than panicking after the operation completed. It does not roll back or
repeat the operation, infer native cleanup or alter unknown-outcome handling.
Initialization retains its explicit authority/lock release sequence and phase
markers before returning the output result.

The pure writer control checks exact success bytes, an injected BrokenPipe write
and an injected flush failure. It opens no IPC/network endpoint, starts no command
or service and makes no native qualification claim. Existing CLI/daemon process
controls remain CI-only under the current local limits. This source fix does not
explain any archived platform failure or qualify installed lifecycle acceptance.

Local validation used the pinned 1.97.1 toolchain, cached/offline dependencies and
at most two Cargo jobs. The exact binary unit control
`cluster::tests::completed_acknowledgement_preserves_text_and_reports_write_or_flush_failure`
passes (0.00s execution; 14.49s build). Both default and all-feature
`cargo clippy --workspace --all-targets -- -D warnings` modes pass
(10.96s/4.57s), as do `cargo fmt --all --check` and `git diff --check`.
Exact-source and separate post-merge CI remain integration gates.
Independent source correctness and Ponytail complexity review pass on this scoped
change; this does not complete independent milestone/release acceptance.
Full GLY-197 remains open for its native readiness/allocation, per-node exclusion,
fit explanations and independent milestone/release acceptance.
