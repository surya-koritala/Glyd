# GLY-185 borrowed range input buffers

Issue: [GLY-185](https://linear.app/getglyd/issue/GLY-185).
PR/commit/CI: recorded with reviewed source and the separate main result in Linear.

The existing `partial::glyd::Worker::run_chunk` accepts borrowed byte views,
including the `Box<[u8]>` buffers retained by `ReceivedTensors`. Callers need not
copy whole boundary tensors into `Vec<u8>` to pass them to the range serializer.
The existing approved `BoundaryContract`, independent expected `StepIdentity`,
ordered `RunChunk` layouts and exact lengths still validate before any write.
At most eight small slice views are resolved once; raw boundary buffers are not
cloned, concatenated or retained by this mapping. Existing vector callers and
the provider's first-stage JSON token encoding remain unchanged.

One pure `Cursor`/vector pipe control passes boxed BF16 byte patterns through the
actual public adapter API and checks the exact emitted header, ordered body and
correlated response. Missing, short and oversized input buffers are refused before
any write or request-ID consumption; a later valid call remains usable. The
existing range lifecycle control retains cancellation/interleaving/framing checks.

Checks use the pinned 1.97.1 toolchain, current lockfile, offline cache and
`CARGO_BUILD_JOBS=2`:

```sh
cargo test --offline --locked -p compute-adapters --lib partial::glyd::tests::borrowed_range_inputs_preserve_boxed_payloads_and_refuse_lengths_before_writes -- --exact
cargo test --offline --locked -p compute-adapters --lib partial::glyd::tests::owned_range_wire_maps_interleaving_cancellation_fences_and_bounded_failures -- --exact
cargo clippy --offline --locked --workspace --all-targets -- -D warnings
cargo clippy --offline --locked --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
git diff --check
```

The new control passed in 0.03 s (1.24 s cached compilation after restoration);
the existing lifecycle/cancellation/framing control passed in 0.10 s. Both
workspace/all-target lint modes passed with warnings denied (9.00 s default,
4.58 s all features); formatting and diff checks passed. A deliberate negative
control removed only the byte-length rejection: the test failed in 0.03 s because
a short input was sent as `c5` instead of being refused. The exact source was
restored and compared before the selected control passed again.

No file, socket, service, subprocess, host inspection, model or native workload
is exercised by the controls. This changes no process supervision, package
delivery, native completion ownership, physical pool qualification, authority
fence, token feedback mapping, dependency or workflow. The secured owner must
still validate its registered boundary before admitting a step, retain the input
receipt, and prove qualified completion before releasing it. Full GLY-185,
GLY-186, GLY-187 and milestone acceptance remain open.
