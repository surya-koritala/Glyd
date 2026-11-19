# Bounded GCP1 input checks

`data_header` uses LLVM libFuzzer and AddressSanitizer through cargo-fuzz.
The shared `check.rs` exercises the actual fixed-prefix and strict header parsers
in both production-rejection and explicit fixture modes. It checks accepted
prefix bounds, prefix identity, header kind/payload consistency and semantic
encode/decode round trips. It never allocates the advertised tensor payload,
opens a listener, authorizes a peer or invokes a backend.

Seven committed synthetic seeds cover Tensor, Credit, ReceiptAck, ExecutionResult,
Cancel and Error, including one valid maximum-size16KiB header. Normal platform
tests replay the same oracle on every seed,
truncated inputs and an oversized header prefix. Seed fixtures grant no execution
permission. The optional runtime has its own locked dependency graph; the
existing dependency-policy job checks it without building instrumented binaries
on every platform. Both workspaces receive formatting checks. No CI job is added.

## Basic local replay

The `receive_cursor` target includes the same private cursor source used by
`FlowReceiver`. Its independent byte-map oracle checks ordered non-overlapping
writes, credit conservation, tensor completion and receipt-before-advance order.
The input has a count byte (modulo eight, plus one), one length byte per tensor
(plus one), then at most64 fixed18-byte actions: opcode, little-endian u64 offset,
little-endian u64 byte count and final-tensor flag. Opcodes0/1/2 mean additional
credit, chunk and finish. Other operations or invalid transitions stop the trace.
Trailing incomplete actions are ignored; this is not a wire parser.

Normal platform tests replay structured valid, truncated and mutated traces.
The Mac development path is limited to these small non-instrumented, in-process
checks: no sockets, listeners, process inspection, model or runtime starts.

```sh
CARGO_BUILD_JOBS=2 cargo test -p starlings-security --test receive_cursor --locked --offline
CARGO_BUILD_JOBS=2 cargo test -p starlings-protocol --test fuzz_seeds --locked --offline
```

The receive oracle owns at most2048 byte-presence flags and processes at most64
actions per input. It checks accounting, not payload values, TLS, live authority,
allocator behavior, native drain or the receipt's actual delivery. Existing CI
retains the integrated flow/lifecycle checks; no job or dependency is added.
No instrumented receive-cursor campaign has been run or qualified.

## Optional isolated campaign

Requires cargo-fuzz0.13.2 and installed nightly-2026-07-20; normal code and seed
tests retain Rust1.97.1. This requires a separately approved isolated test
environment; do not run instrumented campaigns or process-inspection symbolizers
on the development Mac. From the repository root, use a fresh writable corpus
directory outside the checkout so libFuzzer cannot rewrite committed seeds:

```sh
fuzz_corpus_dir=$(mktemp -d)
# On the approved isolated Linux runner, use its installed LLVM symbolizer.
export ASAN_SYMBOLIZER_PATH="$(command -v llvm-symbolizer)"
CARGO_BUILD_JOBS=2 CARGO_NET_OFFLINE=true cargo +nightly-2026-07-20 fuzz run \
  data_header --dev "$fuzz_corpus_dir" "$PWD/fuzz/seeds" -- \
  -max_total_time=30 -runs=10000 -rss_limit_mb=1024 -timeout=2 \
  -max_len=16400 -seed=184
```

Do not disable sanitizers or symbolization to hide diagnostic failures. The run
stops at the first of its time/run limits; a timeout, RSS failure, crash, warning
or incomplete log needs investigation. Keep the full log, actual executable hash,
source/toolchain identities and generated corpus. This campaign does not rebuild
the standard library. Its coverage counters include dependency code.

```sh
cargo test -p starlings-protocol --test fuzz_seeds --locked --offline
cargo clippy --manifest-path fuzz/Cargo.toml --all-targets --locked --offline -- -D warnings
cargo deny --manifest-path fuzz/Cargo.toml --config fuzz/deny.toml --locked check --deny warnings
```

The separate license policy permits only the used subset of the workspace's
licenses, plus NCSA for exactly libfuzzer-sys0.4.13. The runtime's declared
expression is `(MIT OR Apache-2.0) AND NCSA`. This is an unpublished test dependency;
the Mesh license stays undecided.

See [local evidence](../docs/evidence/GLY-184-parser-fuzz.md), the
[Rust Fuzz Book](https://rust-fuzz.github.io/book/cargo-fuzz/tutorial.html) and
[runtime package](https://docs.rs/crate/libfuzzer-sys/0.4.13).
GLY184 remains open for instrumented receive-state campaigns, allocator/copy instrumentation
and physical-link qualification. This target supplies no TLS/lifecycle/native
cleanup, model, performance or Exo qualification, and is not exhaustive proof.
