# Lossless FP8 weights on the GPU

Design note. The owner approved it on 2026-10-05 ("Go") in this build order, ahead of the KV work.

## Goal

`vllm serve <an FP8 checkpoint> --quantization glyd` holds the checkpoint's FP8 weights about 18–20% smaller, every byte exact. It serves at least as fast as vLLM's own FP8 path. Nothing ships before it is that fast; this is the owner's "nothing slower" rule.

## Measured room (2026-10-05, 13 checkpoints of 2026 models; `$S/ceiling/results-fmt2026`)

- FP8 e4m3 weights: an order-0 floor of 6.4–6.6 bits per value (18–20% saved). Coding the sign and exponent while keeping the 3 mantissa bits as they are gives 6.6 bits.
- Scales: 53–94% compressible, but only 3–9% of a checkpoint (phase 2).
- The BF16 parts of these models are already Glyd's. Whole models gain 18.5–21%.

## Layouts covered

| Layout | Scales | Models |
| :--- | :--- | :--- |
| Block-scaled FP8 | fp32 `weight_scale_inv` per 128×128 block | GLM-5.3-Flash, Qwen3.8-27B-FP8, MiMo's FP8 parts |
| Block-scaled FP8 | e8m0 per 32×32 block | DeepSeek-V4.1's FP8 parts |
| MXFP8 | e8m0 per 1×32 | MiniMax-M3-MXFP8 |
| Per-tensor FP8 | one scale | Mistral Small 4 |
| compressed-tensors / modelopt FP8 | as configured | Nemotron's FP8 parts |

Each covers dense Linears and fused MoE experts, with tensor parallel 1, 2, 4 and 8. The GPUs are Ada (sm_89) and Hopper (sm_90a), the ones with FP8 tensor cores. Ampere comes later: there vLLM serves FP8 through its own weight-only path.

## Design

1. **The format.** A fixed-rate code for e4m3 bytes, so any tile can be decoded on its own with a few integer operations:
   - the sign and exponent are coded against a window of the tile's common exponents;
   - the 3 mantissa bits are kept as they are;
   - values outside the window are escapes;
   - a tile with too many escapes is stored raw.

   The scheme is chosen by measurement on real 2026 tensors (Task 1). The target is within 2 points of the order-0 floor, about 16–18% saved on FP8 weights.
2. **Kernels:**
   - pack and unpack, exact;
   - decode into an e4m3 scratch tile;
   - a fused decode-and-FP8-matmul kernel for 1 to 64 tokens a step (Ada `mma.sync` e4m3, Hopper `wgmma` fp8), which is our own code.
3. **vLLM.** `GlydConfig` recognizes an FP8 checkpoint's quantization config and wraps vLLM's own FP8 method for it (Linear and FusedMoE).
   - At load, after vLLM's `process_weights_after_loading`, each e4m3 weight is packed and freed; scales stay vLLM's.
   - **Prompts:** decode the layer into a shared e4m3 scratch, then call vLLM's own FP8 apply on it. That is vLLM's installed kernel, called, not copied.
   - **Decode steps:** the fused kernel.
   - **MoE:** only the experts routed this step are decoded, into a compact scratch whose expert ids are remapped.
   - **`exact`:** every row takes the prompt path, so the logits are vLLM FP8's bit for bit.
    - Which modes: block-scaled and MXFP8 checkpoints are bit for bit in eager, in CUDA graphs and under torch.compile. A per-tensor or per-channel checkpoint (Mistral Small 4, ModelOpt FP8, compressed-tensors tensor/channel) is bit for bit in eager and in CUDA graphs without inductor, and under torch.compile only with vLLM's own activation-quantization op on (`--compilation-config '{"custom_ops": ["+quant_fp8"]}'`, in the run and in the vLLM FP8 it is compared with): by default its quantization is torch ops that inductor fuses into vLLM's graph and that run apart inside Glyd's product, so the last bit of a rounding moves (measured, Task 3).
    - `exact` on such a checkpoint under inductor without that flag is refused at config time, with that reason.
4. **Not in this build:** compressing the scales, Ampere, a BF16-of-FP8 detector (BF16 copies of FP8-native models, about 47% on disk), and the FP4 weights (3–6% room, not worth a kernel).

## Gates

- **Format and kernels:** every e4m3 byte pattern, every sampled real tensor and every edge tile unpacks bit for bit.
- **vLLM:**
  - the packed weights' bytes as vLLM logs them are at least 15% under FP8's on the FP8 Linears;
  - `exact` gives vLLM FP8's logits bit for bit;
  - by default, prompt logprobs and greedy agreement with vLLM FP8 are within the noise of vLLM FP8 against itself under batch changes.
- **Speed:** decode tokens a second at 1, 8, 32 and 64 sequences at least vLLM FP8's, and prompt throughput within 5%, on an RTX 4090 or L40S (Ada) and an H100.
- **Models:**
  - Qwen3.8-27B-FP8, whole, on one H100;
  - Mistral Small 4 with tensor parallel 2;
  - the real first layers of GLM-5.3-Flash, MiniMax-M3-MXFP8 and DeepSeek-V4.1-Flash, loaded as a few-layer model, since the whole models need 8 GPUs.
