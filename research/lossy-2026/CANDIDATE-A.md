# Candidate A baseline

Candidate A retains its existing rounded values. Representation and execution changes must reconstruct those values exactly. No additional weight or activation quantization is part of this baseline. Existing lossless checks remain required.

The complete storage inventory encodes and decodes all 1,199 tensors of pinned Qwen/Qwen3.8-27B. It validates checkpoint hashes, counts every format byte, groups the results by tensor purpose and estimates empirical independent-symbol entropy. This is a storage measurement, not GPU memory or an absolute compression bound. The CPU encoder keeps the original byte format while reducing redundant packing and checksum work.

The prepared quality batch compares original BF16, Candidate A and the official FP8 checkpoint using the same Transformers BF16 arithmetic, eager attention and fixed input tokens. FP8 values are explicitly dequantized into the common arithmetic path; these results cannot describe native FP8 speed, memory or activation quantization. The original-weight native engine and native Candidate A also run, with the existing independent runtime-control bounds checked separately.

CI checks the tensor transform against Rust for every BF16 bit pattern, verifies the actual model parameter names and shapes without allocating weights, and exercises the installed FP8 dequantizer. Checkpoint configuration and tokenizer identities are checked at pinned revisions. Source checkpoint identity remains separate from the common evaluation identity in every result.

The expanded language screen uses 32 disjoint 1,024-token windows, each with 512 context and 512 scored tokens: 16,384 targets per arm. Every position is paired. Reports distinguish a detected likelihood degradation, an inconclusive result and no worsening established on this screen. This does not establish reasoning, coding, instruction/tool, long-context or vision acceptance; each needs its own held-out evaluation before broad quality claims.

## Prepared GPU batch

quality_match_job.sh runs one immutable batch with the CI-built A100 binaries, pinned container and dependencies, frozen evaluation, source metadata and checksums. Proposed maximum: one A100 80GB for 75 minutes, with the workload stopping at 70 minutes and deletion controlled by the outer launcher. The script does not create a pod. A separate approved spending cap and verified price are required before launch.

Order: verify prepared inputs and environment; download the two pinned checkpoints; run native kernel checks; score common BF16 and native BF16 control; require the existing control gate; score common Candidate A and common FP8; score native Candidate A and require its independent control; write paired comparisons. Candidate B is excluded. A failed control stops interpretation of the affected runtime.

## FP8 arithmetic census

Candidate A keeps three fraction bits, as e4m3 does, but BF16's eight exponent bits. `fp8_census.py` with the `glyd-fp8-census` binary (manual CI job `fp8-census`) reads every pinned matrix, applies the Candidate A rounding and counts values that are exactly e4m3fn values times the smallest power-of-two scale covering their block, for 128x128, 1x128 and 1x32 blocks and whole rows. Every rounded word and scale is checked in CI against an enumeration of all e4m3fn values. The same pass counts 7-bit e3m3 and 6-bit e2m3 formats (three fraction bits, subnormals, no NaN), which become e4m3 by an exponent offset: fixed-width storage that a kernel converts with a few bit operations instead of decoding. The counts decide whether an FP8-arithmetic variant could keep the fixed weights bit for bit, and how many values a narrower fixed-width store would need to hold apart or round; activations are not involved, and the census is not a quality, memory or speed result.

## Following gates

1. Complete encoded inventory and matched quality evidence.
2. Direct compact GPU loading and an exact decoder for Candidate A values.
3. First-party fused decoding and products, with separate prompt and generation profiles.
4. Same-GPU comparison against stock FP8 at 1, 8, 32, 64, 128 and 256 users, including peak memory and both latency metrics.
5. Cold/cached startup and the normal installation, run, serve and doctor flows.

Automatic routing and public speed claims require every load to improve throughput, first-token latency and subsequent-token latency while using less memory. No compact-format GPU result exists yet.
