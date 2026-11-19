# Candidate A compact GPU preparation

Candidate A's finite rounded BF16 words have four zero fraction bits. This experimental representation removes those bits from the existing matrix significance stream, retaining the existing matrix order and exponent buffers. Packing refuses finite values that are not already rounded. Non-finite payloads that need the omitted bits retain the existing complete tensor representation.

The Rust implementation checks the original matrix packing, verifies reconstruction and reports every resident weight-buffer byte. The CUDA implementation restores significance in registers and uses the existing first-party BF16 decoding and product operations. It supports whole-matrix decoding and products with 1–2,048 activation rows; products above 64 rows require an inner dimension divisible by 64 and select their block shape as the expanded representation does. It does not quantize activations.

This is separate from the measured 24.43534 GiB reference representation. The [first isolated-matrix batch](../../benchmarks/weights/candidate-a-compact-2026-10-07/README.md) decoded every case bit for bit at 6.7736 bits per weight across the model-shape matrices. Whole-model resident size, device-memory timing, performance and model integration are pending. No claimed savings or speedup should be extrapolated from the structural removal of bits. Existing lossless formats and production routing are unchanged.

CI checks the host representation across all BF16 patterns, special-value payloads, matrix ordering and invalid inputs. The shared device expansion helper is independently checked for all 65,536 possible four-value fields. CUDA compilation prepares artifacts before a separately approved GPU validation job. No kernel runs on a workstation.

Following validation must compare exact decoded weights, exercise products at each workload size, measure memory and time with CUDA events, then integrate the path into the native model loader and run matched end-to-end workloads against official FP8. Cold/cached startup, total VRAM, throughput, time to first token and time per token remain required. This research code is not included in release builds.

The [prepared validation protocol](VALIDATION.md) defines the next isolated-matrix GPU batch, its immutable input preparation, error bounds, memory accounting and the separate native-outlier investigation. Device results remain pending.
