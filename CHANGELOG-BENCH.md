## Harness + G1: liblz4 baseline, real 1M fuzz, OOB write fix

- Baseline switched to reference C liblz4. Against it the real gap is far
  wider than against lz4_flex: Silesia decomp 2.66 vs 5.66 GB/s (0.47x),
  comp 0.42 vs 0.89 GB/s (0.47x), ratio 1.90x vs 2.10x (0.91x). 0/12 files
  beat liblz4 on decompression.
- Fixed: Snappy compression was timed against a zero-length output slice, so
  it errored instantly and reported ~720,000 GB/s.
- Fixed: G1 gate was hardcoded to PASS. It now reads .g1-status.json written
  by the fuzz test and FAILs when absent or under 1,000,000 mutations.
- SECURITY: heap-buffer-overflow in the tail phase of both vector decoders.

## decode: branchless literal wildcopy
Silesia 1C decomp 2.694 -> 4.053 GB/s (0.488x -> 0.733x liblz4). Ratio and
compression unchanged. Removed the per-token `if lit_len > 0` guard and the
<=32 / >32 dual path in favour of an unconditional vector store loop.
Hash-table sweep (16KB..256KB) recorded: ratio and compression speed trade
off almost exactly, no setting reaches G2 and G4 together.

## compress: widen raw-block bypass from 2% to 4% savings
Silesia 1C decomp 3.882 -> 4.209 GB/s (0.708x -> 0.759x liblz4).
Total ratio 1.9022 -> 1.9002 (-0.1%). x-ray decode 6.15 -> 12.68 GB/s with
ratio 1.011 (1.001x liblz4, still clear of the 0.95 floor).
Rejected first: a sampling pre-scan (stride-16 4-byte hash hit rate). It
never fired on x-ray yet did fire on mozilla, costing 3.7% ratio there.
Measured savings beat predicted savings.
