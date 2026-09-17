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
