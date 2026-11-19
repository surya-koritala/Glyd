# Synthetic loopback transport measurements

The test-only harness measures GCP1 without a model, backend or external listener.
It reuses enrolled loopback TLS, live local admissions and the credited flow APIs.
Run from the repository root with a new absolute output path:

    STARLINGS_TRANSPORT_BENCH_OUTPUT="$PWD/transport.jsonl" cargo test -p starlings-security --lib data::tests::benchmark::record_bounded_transport_matrix --locked --offline -- --exact --ignored --nocapture

The output parent must exist; relative paths and existing files are refused before
fixtures start. The explicit matrix is excluded from normal CI. One8KiB/two-input
correctness check runs normally, without a performance threshold or extra CI job.
There are20 samples:8KiB/8MiB per tensor, one/two tensors and five trials per case.
Each sample has a fresh synthetic fixture and connection. Multiple tensors share
that connection, but persistent reuse across invocations remains GLY241 work.
The same10-second absolute flow deadline applies; no retry or expiry relaxation.

JSONL starts with actual test-binary SHA256, package/build/OS/architecture and
synthetic payload identity. Each row records tensor/trial identity, measured times,
exact-byte/core-state checks and retained payload accounting. A complete footer is
written only after all20 samples pass; a partial file is not a successful matrix.
Output contains no tensor bytes, keys, tokens, local paths or peer identifiers.

| Field | Timed boundary |
| --- | --- |
| tcp_tls_binding_setup_ns | Listener, TCP, TLS and GCP1 binding setup, excluding fixture construction |
| host_copy_baseline_ns | Explicit host copy of the synthetic payloads into one scratch buffer; allocations outside timing |
| receive_begin_policy_reserve_allocate_ns | Live admission, payload reservation and receiving buffer allocation |
| receive_read_validate_ack_ns | Receiving payloads, validating headers and writing their receipts, including blocked I/O |
| input_handoff_ns | Final live input-state check and owned buffer handoff |
| send_framed_tls_ns | Framing, validation and TLS enqueue, including blocked I/O |
| credit_wait_validate_ns / receipt_wait_validate_ns | Sender wait and validation for initial credits / receipts |
| application_elapsed_ns | Begin/transfer/handoff/thread scheduling/join wall time; excludes setup and copy baseline |

Sender and receiver phase times overlap and must not be added or subtracted to
infer TLS-only, scheduling-only or device cost. Application goodput is payload
bytes divided by application elapsed time; it is not raw physical-link speed.
The driver interleaves send/receipt operations; no one-round-trip-per-token claim.

The budget must remain held at allocation and after input handoff, then reach
zero after drop. Returned tensors must match all bytes and leave core worker state
unchanged. These are retained payload accounting checks, not RSS, allocator peak
or TLS-internal copy instrumentation. Those, raw physical-link speed, isolated TLS
and native device synchronization remain explicit nulls. Source/receiver fixtures,
copy buffers, metadata and TLS use additional unmeasured memory.

The local [raw record](../benchmarks/2026-10-16-loopback-transport/README.md) validates
the harness only. It is not wired-link, model/inference, native allocation/drain,
device profile, calibrated planner or Exo qualification. GLY184/175/241 and the
deployed183/native gates remain open.
