# GLY-176 cancellation response settlement

Issue: [GLY-176](https://linear.app/getglyd/issue/GLY-176).
Exact signed source, PR and source/main checks are recorded in Linear.

The cloneable request cancellation handle can run concurrently with generation
completion/status. Previously an ended status let another cancel caller return
and the service become reusable while the first cancellation HTTP operation was
still outstanding. That operation could subsequently fail and stop the reused
process. A status receipt and a settled local cancellation are distinct facts.

One pending-operation bit in the existing request mutex now gates both follower
cancel completion and service reuse. The first caller owns the bounded HTTP
operation; on failure it withdraws the process before clearing pending. No lock
is held over HTTP or process signalling. Pre-dispatch and already-ended old IDs
still require no cancellation HTTP request. The existing two-second budgets,
permanent shutdown and native cleanup/pin/reservation requirements are unchanged.

The pure state control observes normal and cancelled ended status during an
outstanding cancellation. Restoring the old ended-only predicate makes it fail;
restoring the pending guard passes. It executes no socket, process or workload.

The existing synthetic HTTP child gains only two gated response cases. CI starts
the real request and first cancellation, withholds that response while ordinary
status observes ended, and verifies generation cannot return for reuse. A released
successful response permits the next request and ignores the old handle; a late
failed response withdraws the service. It is software concurrency evidence, not
native completion or GPU cleanup proof. These process/HTTP cases run only in the
ordinary existing platform CI, with no new job, timeout or dependency.

Local commands use pinned Rust 1.97.1, offline/current lockfile, existing debug
cache and at most two Cargo jobs:

```sh
cargo test --offline --locked -p compute-adapters --lib service::glyd::tests::request_end_does_not_settle_an_outstanding_cancel -- --exact
cargo test --offline --locked -p compute-adapters --all-features --lib --no-run
cargo clippy --offline --locked --workspace --all-targets -- -D warnings
cargo clippy --offline --locked --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
git diff --check
```

Full installed-backend, native resource/drain, physical qualification and
independent milestone/release acceptance remain open.
