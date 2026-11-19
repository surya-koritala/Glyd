# Compact execution validation

The immediate batch validates isolated compact matrices on one RTX 4090. It does not establish whole-model residency, native FP8 performance, model quality or serving throughput.

The prepared fixture set uses one tensor per compatible two-dimensional shape from the pinned Qwen3.8-27B inventory. All such tensors are complete except the explicitly declared 4,096-row embedding slice. Non-matrix tensors and incompatible shapes remain listed as uncovered. Five synthetic cases cover integer products, partial row blocks, inner-dimension tails, every compactable BF16 word and every BF16 word through the unchanged fallback. Every source range, rounded matrix and packed buffer has a frozen digest. Source range identity is checked against the pinned revision and Content-Range; ranges are not presented as full-shard hash verification.

The Rust preparation independently checks expanded buffers against the existing packer and reconstructs every rounded word. Device unpacking must match those words bit for bit, including special-value payloads. Products use 1, 2, 8, 16, 17, 32, 33, 64, 65, 127, 128, 129, 256, 512, 1,024 and 2,048 activation rows, with and without bias. Unsupported large products with an inner dimension not divisible by 64 must return the documented refusal. Guard words detect output overwrites. Integer products must match exactly; real tensor products retain the existing one-percent-of-maximum error bound and add a 0.004 normalized-RMS bound against FP32 products. Repetition and graph replay must reproduce the same output bits. These diagnostic product bounds do not change model lossless checks or task-quality policy.

Memory records separate logical weight-buffer bytes, host symbol bytes, allocator holdings, reserved memory, workspace, counters and compact-only peaks. Reference tensors and timing copies are allocated after the compact peak is recorded. CUDA event timings use five groups of replayed graphs, after warmup. Each timed operation reads its own weight copy, and the copies of one path span at least 192 MiB, so timed weights come from device memory rather than the RTX 4090's 72 MiB L2; at most 256 copies are made, so the tiny synthetic cases remain L2-resident. The first batch (`benchmarks/weights/candidate-a-compact-2026-10-07`) replayed one copy and is L2-warm.

The references are the existing first-party expanded representation, BF16 matrix multiplication and a diagnostic FP8 product: per-tensor e4m3 weights and activations through cuBLASLt with fast accumulation. Its activation quantization is excluded, so it bounds FP8 product time from below; it is not the official checkpoint's block-scaled serving path, and its error is recorded but not gated. If the FP8 product is unavailable for a shape, the reason is recorded. Decoder-only timing is diagnostic. No matrix timing is relabeled as model tokens per second.

Above 64 rows the compact product selects its block shape as the expanded representation does (128 tokens by two row blocks, or 256 by one), and both shapes are also forced, checked and timed. The small-product path still uses the generic implementation rather than the expanded type's specialized step implementation. These are diagnostic comparisons, not performance claims.

## FP8-arithmetic variant (diagnostic)

`fp8_variant.py` runs after the exact matrices and has no gate of the frozen protocol. Producer warps decode the compact weights and convert them in shared memory to e4m3 with one power-of-two scale a row (the smallest covering the row's largest magnitude; values below e4m3's range round to nearest even, so this is a separate numerical variant), X is converted to e4m3 with one scale a token, and mma.m16n8k32 sums in fp32 before both scales are applied. Three unit shapes are timed: 64 tokens by two row blocks, 128 by two, and 256 by one.

Each matrix is checked in its own process: the device's e4m3 bytes against PyTorch's conversion of the same scaled weights, the quantized X bytes and scales against PyTorch's, products against an fp32 product of the same e4m3 values and scales, repetition and graph replay. Error against the exact Candidate A product is recorded. It requires compute capability 8.9 or later; per-128 activation scales, the activation conversion's place in a model and quality are outside this batch.

## Native runtime discrepancies

The outlier-cases.json record identifies every previously recorded Candidate A target whose native/common log-probability difference exceeds one nat. It preserves exact prefixes and target IDs, source hashes, both original-weight controls and both Candidate A arms. It is a diagnostic selection from existing results, not a fresh quality evaluation set.

The common scorer processes each complete sequence in one eager call; the native scorer uses a 512-token prefill in chunks of 256 followed by incremental decoding. The first controlled diagnostic must hold weights, inputs and target position fixed while separating those execution modes:

1. Reproduce the selected target and neighboring positions with original weights and Candidate A in the original two modes.
2. In the common runtime, compare full-sequence, prefix-only and incremental-cache evaluation. In the native runtime, compare the original split with a fresh prefill ending immediately before the target. Record the full target logit, top logits and log-sum-exp, not only the target probability.
3. Capture per-layer residuals, normalization outputs and hybrid-state summaries at the first divergent position. Compare full-attention and linear-attention state updates before attributing the issue to compact weight decoding.
4. Isolate the first differing operation, repair a demonstrated implementation error, and rerun the untouched paired quality protocol. If it is expected arithmetic sensitivity rather than an error, retain it as an explicit quality limit.

The RTX 4090 matrix batch does not load the 27B model and cannot resolve those full-model discrepancies. Any later full-model diagnostic requires its own prepared, reviewed and capped job. No weight values or quality thresholds change in this batch.

## Execution and decision

Preparation runs on CI: host regressions, pinned tensor download and packing, CUDA compilation for SM80/SM89, and a fresh check of the exact immutable pod image. The image's existing Python/PyTorch stack is checked without installing packages. The resulting archive has a complete checksum inventory.

The proposed first execution is one RTX 4090, 45-minute outer cap, 40-minute workload cap, automatic deletion and no retry. A current quote, upload-time estimate, verified archive, spending-guard state and explicit batch approval are required before allocation. The preparation alone grants no spending permission.

A missing case, changed input, incorrect decoded weight, product error, non-repeatable product, unexpected refusal or time cap produces incomplete evidence and stops progression. If correctness passes, use the measured workspace/traffic/timings to choose the next implementation change. Whole-model loader integration, the actual weight/peak-memory target, native FP8 comparison at all serving loads, remaining quality gates and standard onboarding stay outstanding.
