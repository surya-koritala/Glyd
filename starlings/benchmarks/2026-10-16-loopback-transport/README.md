# Synthetic GCP1 loopback matrix

[Raw JSONL](raw.jsonl) SHA256
284e909ba9c63bafaf36830d8453e11b1e5573cabcc2ff2c45228436a125d5e0.
The22 lines contain a run header,20 samples and a successful completion footer.
All four8KiB/8MiB and one/two-input cases have five trials;125,952,000 payload bytes
were transferred and checked exactly. Every retained-payload ledger reaches zero
only after owned inputs are dropped; core snapshots remain unchanged.

This is the unoptimized local test harness, not a native/model benchmark. Actual
test-binary SHA256, OS/architecture and payload identity are in the header. Base
source is32482b527c6a237119739c5c518e600aa32bdd5e plus the harness:
benchmark.rs Git blob1b7b6e639fcab93648d95b69cca8d79e5dca843d;
parent tests.rs Git blob188c61a97061ce2758eaf8e148cf4eee21c1c760.
No native backend or physical link is exercised or qualified.

The explicit matrix passes10.93s, raw test-log SHA256
cbc9fe5553000fe921720247d7f64e73af1f09f88cdff2d16d11ca3422508270.
The first driver invocation stopped before fixtures at a missing relative output
parent; failed log7093dcd44ff9ea7785d0d1d62f8b6303df997c2cebf0d55c5f7cd4db7e3238ca
is preserved. The harness now requires an absolute new path. Independent checks
verify complete unique rows, byte/goodput arithmetic, actual binary identity and
refusal of relative/existing outputs without changing this raw artifact.

See [measurement boundaries and reproduction](../../docs/transport-measurements.md).
Phase times overlap; do not sum them, subtract them to estimate TLS, or interpret
loopback application goodput as wire/model performance. RSS/allocator peak,
TLS-internal copies, physical-link speed and device synchronization remain null.
No performance claim or milestone acceptance follows.
