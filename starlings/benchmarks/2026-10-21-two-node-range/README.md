# Two Starlings nodes, real engine stages (2026-10-21)

Qwen3.5-9B (glyd linnet) split at layer 16 across two `glyd-engine rangeworker` processes, each driven by
Starlings' Glyd adapter. Each step's boundary (residual + MLP output, bf16) crossed Starlings' enrolled TLS data
channel between two enrolled nodes with live admission. 64 prompt tokens, 32 greedy tokens, chunks of 32.

Result: every token and every logit equal to the whole model's, bit for bit (both sha256 `3a282b16…`).

| decode, ms per token | |
| --- | --- |
| whole model (one rangeworker) | 9.99 |
| split, total | 20.86 |
| stage 0 + stage 1 (GPU work) | 4.34 + 5.70 |
| data channel transfer | 9.25 |
| admissions + start step | 0.93 + 0.46 |

The engine's own `rangecheck --workers` (pipes only) adds 0.33 ms a token over the whole model, so the extra ~11 ms
is Starlings. One policy check costs ~0.45 ms (SQLite lookups and certificate parsing per node of the plan) and the
transfer repeats it on every frame at both ends. That is the next fix.

Limits: both stages on one RTX 4080 SUPER (one machine, loopback TLS); the token returns to stage 0 through the test,
not the data channel; random prompt ids, so the output repeats a token (bit equality still covers every logit).

Machine: Ryzen 9 7950X3D, RTX 4080 SUPER (driver 595.91.07), Ubuntu kernel 7.0.0-38. Engine: glyd-gpu-private
main c90379d4, dev build, binary sha256 in record.json. Command, from the repository root on the PC, under the GPU lock:

    STARLINGS_RANGE_ENGINE=… STARLINGS_RANGE_MODEL=… STARLINGS_RANGE_RECORD=… \
      cargo test --release -p starlings-security --lib \
      data::tests::range::real_range_workers_across_enrolled_tls_equal_the_whole_model -- --exact --nocapture

## After remembering enrollment rows (record-row-memo.json)

Same test after the permission checks stopped rereading SQLite rows on every frame: still bit-exact, decode
20.86 -> 12.78 ms a token (transfer 9.25 -> 2.27, admissions 0.93 -> 0.20, start step 0.46 -> 0.12). What remains
per decoded token: 21 frame checks of ~60 µs, ~38 µs of each the access lock's file and clock checks.

## After reading the database version once per lock (record-lock-version.json)

The access lock reads the clock floor and `data_version` in one statement; checks under that lock skip their own
version read (this connection's writes still count at once). Still bit-exact: decode 11.94 ms a token
against 9.97 whole (transfer 1.55, admissions 0.13, start step 0.09).
What remains in the transfer is mostly waiting: every TLS read and write call rechecks both certificates (one
statement and a file check each, ~50-60 checks a step), and each send waits a credit round trip first.

## One write a frame, credit granted early (record-one-write-early-credit.json)

A data frame's prefix, header and payload go out in one write (one checked call and one TLS flight, not three
small segments). The test's receiver grants credit while stage 0 computes, as a pipeline would, so the sender's
credit is waiting when the boundary is ready. Still bit-exact: decode 11.47 ms a token against 9.95 whole
(transfer 1.12, admissions 0.14, start step 0.08). Every TLS read and write still rechecks both certificates.

## A passed check lasts 10 ms within a frame (record-recheck-window.json)

Inside one frame, a TLS read or write reuses a membership check that passed in the last 10 ms (owner approved,
2026-10-22). Outside frames every call still checks, and a revocation made through this process still shuts the
socket at once. Still bit-exact: decode 11.21 ms a token against 10.00 whole (transfer 0.79, admissions 0.15,
start step 0.09). Most of what is left is the data admission checks (about 15 a step, each with its own lock).
