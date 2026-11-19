# Qwen3.8-27B: optional Glyd lossy representation

Private research, separate from the lossless product and its exactness checks. The candidate is Glyd's Rust representation, intended for Glyd's own engine and CUDA implementation. The prototype neither depends on vLLM nor converts weights into standard FP8/FP4. FP8 is the active comparison baseline. Candidate B and FP4 work are deferred. See [the Candidate A baseline](CANDIDATE-A.md) for the current implementation and acceptance order.

## Model and comparison

Use Qwen/Qwen3.8-27B, released in 2026, a dense 27B model using the qwen3_5 architecture already under development in the engine. Hub metadata at preparation shows more than six million downloads in the preceding month. The exact createdAt fields, revisions and metadata for all three checkpoints are in models.json. First-party source: https://huggingface.co/Qwen/Qwen3.8-27B . Choose 27B before larger models so that representation work and consumer-card validation cost less; a larger model is a later generalization gate.

Reference checkpoints:

- BF16: Qwen/Qwen3.8-27B.
- FP8: Qwen/Qwen3.8-27B-FP8.
- NVFP4: nvidia/Qwen3.8-27B-NVFP4.

A named precision is not a quality guarantee. Establish a curve of quality, memory and speed; do not claim a universal winner from one operating point. Useful targets are less memory than FP8 at comparable quality, and better quality than NVFP4 at comparable memory. Neither has been demonstrated by this experiment yet.

## Stage 1: implemented reference study

rust/ is a standalone, dependency-free Rust reference codec. It rounds BF16 values to retain zero through seven fraction bits, with nearest-even rounding. Signed zero, infinities and NaN payloads are preserved; finite overflow saturates. Each block stores a small adaptive dictionary of exponents, fixed-width sign/fraction/index fields and full rounded BF16 exceptions. Unprofitable blocks use raw rounded words. CRC and structural checks reject malformed input. Seven fraction bits is the bit-exact control.

Every header, dictionary, exception and padding byte is included in the reported size. This is an experimental format, not a released model artifact or a stable API. No default path or lossless packer changes.

The CI study reads pinned safetensors headers for a complete inventory, then first/middle/last tensors and windows from embedding, head, full attention, linear attention, MLP, vision, MTP and other groups where present. It records offsets, hashes, dtypes and all omitted/non-BF16 tensors. It measures actual encoded bytes and reconstruction error. A sampled window is not a whole-model compression measurement. Numerical weight error is not an accuracy score. The workflow uploads metadata/results, not copied checkpoint weights.

Run only on CI or a prepared pod:

    cargo test --release --manifest-path research/lossy-2026/rust/Cargo.toml
    cargo build --release --manifest-path research/lossy-2026/rust/Cargo.toml
    python3 research/lossy-2026/sample.py --binary research/lossy-2026/rust/target/release/glyd-lossy-study --out /tmp/lossy27-results

All 65,536 BF16 bit patterns are checked at each retained precision against the declared rounded reference. Separate tests cover ties, finite overflow, exceptional values, block tails, dictionary escapes, corruption and truncation.

## Stage 2: initial language-quality screen completed

The initial A100 screen completed on 2,048 targets: native-control perplexity 6.930752 and Candidate A 6.922349. The -0.1212% change is within the window-bootstrap uncertainty, not evidence of improved task accuracy. Compact-format VRAM and speed remain unmeasured.

quality/ contains a standalone native-engine likelihood scorer. An explicit development-only weight transform applies the same rounded values the reference codec decodes; an exhaustive CI check compares all BF16 patterns between the two paths. All BF16 text weights, including norms and the small linear-attention parameters, receive the selected transform. Original FP32 values remain unchanged. A no-transform control is the default. The existing lossless GPU format carries these rounded values for this quality screen; the new compressed representation does not run on the GPU yet, and this screen cannot measure its final memory or speed.

The loader now accepts the pinned checkpoint's BF16 A_log and gated-normalization parameters and widens them exactly to the FP32 storage the kernels read. Its checkpoint inventory is checked against all 1,199 real tensor headers and the pinned configuration in CI. Vision and MTP remain excluded by this text forward pass, with their count recorded. This change establishes host-side compatibility, not GPU acceptance.

prepare_quality.py freezes eight disjoint 512-token windows spread through the WikiText test text at a pinned dataset revision and content hash. Each uses 256 context tokens followed by 256 teacher-forced targets: 2,048 scored tokens, without calibration or a chat template. quality_reference.py supplies an independent BF16 Transformers reference. quality_compare.py pairs every position, refuses identity/length mismatches, reports likelihood and top-1 agreement, and uses a paired window bootstrap to describe uncertainty. This small corpus is a screening tool, not a reasoning, coding, long-context or multimodal quality claim.

The immutable job first runs the existing GPU checks and the original-weight native control. Its screening bounds are at least 95% top-1 agreement, at most 0.05 nats mean absolute target-logprob difference, and at most 0.10 nats absolute mean NLL difference per window against the independent reference. A failed control stops both candidate runs. These bounds diagnose forward-pass compatibility; they do not replace any bit-exact weight check or define an acceptable product accuracy loss.

CI builds the A100 binaries and CUDA library and freezes the evaluation before any paid session. It also runs the reference dependency bootstrap in the exact pinned Runpod container image. The bootstrap creates an isolated Python environment, reuses verified CUDA wheels from the image, pins the reference dependencies and imports the actual model class. CI reproduces the system-Python installation refusal and checks that the corrected setup leaves system packages unchanged; the binary bundle depends on that passing check. quality_job.sh uses those artifacts, verifies their hashes, runs the reference, control, three-bit and two-bit candidates in one batch, and writes DONE on exit for deletion by the outer launcher. Proposed pod: one A100 80GB at no more than $1.59/hour, 90 minutes maximum (job workload stops after 80), $2.39 maximum GPU time before storage. Spending and the existing data-center block still require a specific approval; the script never provisions a pod itself.

Remaining gates:

1. Confirm the engine's Qwen3.8-27B language forward pass on the pinned checkpoint. Qwen3.5-family development support is not proof of this model's acceptance; vision, MTP and serving remain separate capabilities.
2. Select at most two promising stage-1 representations. Add an explicit lossy model manifest and engine loading path. The decoder must reproduce the chosen approximated weights exactly. Keep all existing lossless checks unchanged.
3. Measure all loaded weights and retained BF16 tensors, calibration metadata, padding and workspaces. Record vision and MTP inventory even if a particular language-only run does not execute them; do not claim whole-model multimodal support from that run.
4. Evaluate BF16, the two Glyd candidates and the pinned FP8/NVFP4 references with identical prompts, tokenization, generation limits, seeds and thinking settings. Verify that the reference checkpoints derive from the intended BF16 model and record any calibration differences.
5. Use held-out language likelihood for screening, followed by reasoning/math, code, instruction/tool use and long-context tasks; vision needs its own acceptance. Calibration material is disjoint from evaluation. Report uncertainty and each task's change, not only a blended score. A numerical error improvement alone does not pass.
6. Test lossless and lossy weights in the same engine to isolate the representation effect. Native FP8/NVFP4 implementations are external comparison tools only. Preserve their separate runtime/configuration in every record.

No quality-loss budget has been finalized. Explore the curve before making a release claim; at most one absolute percentage point per selected task is a candidate quality gate, not an agreed result.

## Stage 3: GPU execution, not implemented

Add first-party CUDA decoding and fused matrix products to the engine for the selected representation. A CPU decoder or an engine that permanently expands the whole model to BF16 does not establish GPU memory savings. Include scratch and expanded tensors in peak memory. Benchmark native FP8/NVFP4 against the actual Glyd engine on the same GPU at 1, 8, 32, 64, 128 and 256 users where both fit: output tokens/sec, TTFT, TPOT and peak VRAM. Report out-of-memory cases separately.

A speed claim requires every load and latency metric to pass the existing speed bar. Hardware-supported low precision has a compute advantage; fewer stored bits alone does not make decoding or inference faster. If one quality/memory point loses on speed, retain it only as research rather than enabling it by default.

## Spending and release boundaries

Stage 1 uses GitHub CI and creates no Runpod spend. A full BF16 27B reference needs substantially more GPU memory than a consumer card; request a concrete capped data-center reference run only after the native engine path, checks and immutable job are prepared. Consumer confirmation follows on the target RTX card. No new pod, funding-cap change, release, site claim or model publication is authorized by this study workflow.
