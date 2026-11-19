# Quantized 2026 checkpoints under `--quantization glyd`: their 16-bit parts packed (2026-10-06)

`vllm serve CHECKPOINT --quantization glyd` on a quantized checkpoint (`glyd-gpu-py/glyd_gpu/vllm_quant.py`): the checkpoint's own
quantized layers run with vLLM's own method for its format, exactly as stock vLLM runs them; Glyd packs the 16-bit parts it keeps
(Linears it leaves bf16, the embedding, the LM head, vision-tower Linears where vLLM gives the tower the quantization config) by the
rules of a bf16 model. Each checkpoint was served by stock vLLM and by Glyd on the same pod, one vLLM after the other
(`gpu/vllm/quant_parts.py`, offline, in-process; vLLM 0.30.0, torch 2.13.0+cu130, transformers 5.18.0):

- **exact**: stock eager against Glyd exact eager, with every pack decoded as it is made and compared with its weights (`verify`): the
  greedy tokens and top-5 logprobs of 8 prompts (64 tokens each), the prompt logprobs (top 5) of a ~710-token text, and one image chat's
  tokens and logprobs. "Bit for bit" is all of them equal. Where they differ, stock eager is run again and compared with itself.
- **memory**: "Model loading took" (the weights on the GPU) and vLLM's KV cache (GiB, tokens, and requests of 8,192 tokens at once), at
  `--gpu-memory-utilization 0.9`, `--max-model-len 8192` (where a row names others, its own), compiled with CUDA graphs (vLLM's default),
  in Glyd's default mode (fused: each product decoded and multiplied in one kernel; its outputs are not stock's bits, its packs are the
  weights bit for bit). Exact mode's weights and KV cache are its eager run's (it keeps the LM head, and a table tied to it, as vLLM runs
  them, so it holds more). The KV cache GiB is what a user gains for requests; the weights column is vLLM's own count.
- **speed**: output tokens a second at 1 and 32 requests at once (512 random prompt tokens, 256 out, ignore_eos), Glyd's default mode,
  the median of 2 generate calls of one vLLM start (of 3 where a row says), stock's start before Glyd's on the same pod (not interleaved).
  Exact mode's compiled speed was not measured.
- **verify**: in runs 2 and 3 the default-mode run also decoded every pack as it was made and compared it with its weights (`verify`, an
  option a user does not run by default); in runs 1 and 2a it did not. On Qwen3.5-35B-A3B-GPTQ-Int4 a verifying load holds about 0.6 GiB
  beyond the model's own tensors (0.07 GiB without verify), so that row's weights and KV cache are of the heavier load (not looked into).
- **what was packed**: every packed layer's bf16 bytes and its pack's, and every 16-bit tensor of 1 M values or more left as it is.

## Results: L40S (48 GB, Ada)

Stock vLLM -> Glyd on the same checkpoint, same pod. Runs: `l40s-run1/` (commit `5c7ce578`), `l40s-run2a/` and `l40s-run2/` (`8da693ae`,
`8d498a63`: the sub-model table fix, `--max-num-seqs 64` for the hybrid FP8 checkpoints, the compiled run verified), `l40s-run3/`
(`bcdf1746`: main's packing of what vLLM builds without the quantization config, taken by a quantized checkpoint too: Qwen3.5/3.6/3.8's
embedding, and main's padded rows: their vision towers' 4,304-row Linears). Each row names the runs and commits its cells come from:
the five Qwen rows are run 3's; the three Gemma rows are of earlier commits and were not run again at this PR's head (see "Not done").

| Checkpoint | Format (vLLM's method) | Weights GiB, stock -> Glyd default mode | Saved | KV cache GiB (tokens), stock -> Glyd default mode | Glyd exact mode: weights GiB, KV tokens | Glyd exact mode vs stock eager | 1 user tok/s, default mode | 32 users tok/s, default mode | Runs (commit) |
| :-- | :-- | --: | --: | --: | --: | :-- | --: | --: | :-- |
| nvidia/Gemma-4-31B-IT-NVFP4 | NVFP4, bf16 attention (ModelOpt, Marlin on Ada) | 31.18 -> 25.25 | **5.93 GiB (19.0%)** | 5.86 (13,941) -> 11.82 (28,121) | 26.11, 28,060 | bit for bit (120 of 120 packs verified, + image) | 21.79 -> 26.07 (1.20x) | 230.88 -> 421.65: stock KV-bound (queues) | memory and speed: 2a (`8da693ae`, verify off); exact: 1 (`5c7ce578`) |
| cyankiwi/Qwen3.6-27B-AWQ-INT4, an **fp16** checkpoint (`--dtype bfloat16`, stock and Glyd) | INT4 AWQ (compressed-tensors) | 19.20 -> 17.46 | 1.74 GiB (9.1%) | 17.52 (176,368) -> 19.29 (194,198) | 18.20, 194,680 | equal to stock at `--dtype bfloat16` (111 of 111, + image); **not lossless**: vLLM rounds its fp16 parts to bf16 before Glyd packs them | 38.64 -> 40.24 (1.04x) | 523.08 -> 534.61 (1.02x) | 3 (`bcdf1746`, verify on) |
| google/gemma-4-12B-it-qat-w4a16-ct | INT4 W4A16 (compressed-tensors) | 8.28 -> 7.63 | 0.65 GiB (7.9%) | 29.53 (91,998) -> 30.21 (94,109) | 8.26, 93,920 | bit for bit (1 of 1, + image) | 74.74 -> 80.54 (1.08x) | 1027.99 -> 1072.36 (1.04x) | 1 (`5c7ce578`, verify off) |
| Qwen/Qwen3.8-27B-FP8 | FP8 block | 28.90 -> 27.13 | 1.77 GiB (6.1%) | 8.57 (86,256) -> 10.33 (103,604) | 27.92, 103,122 | bit for bit (111 of 111, + image) | 24.97 -> 25.62 (1.03x) | 389.46 -> 395.80 (1.02x) | 3 (`bcdf1746`, verify on) |
| Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 | INT4 GPTQ (auto_gptq, Marlin MoE) | 21.12 -> 19.97 | 1.15 GiB (5.4%); KV cache +0.97 GiB | 15.62 (453,485) -> 16.59 (481,572) | run 1: 20.63, 492,105 | **no bit-for-bit result**: stock eager differs from its own second run (runs 1, 2); packing only (every pack verified bit for bit) | 117.08 -> 145.48 (1.24x) | 1328.66 -> 1390.09 (1.05x) | 3 (`bcdf1746`, verify on) |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | NVFP4 W4A16 experts and head, FP8 attention (ModelOpt mixed, Marlin MoE) | 20.37 -> 19.84 | 0.53 GiB (2.6%) | 16.73 (485,668) -> 17.24 (500,297) | run 2: 20.20, 504,393 | **no bit-for-bit result**: stock eager differs from its own second run (run 2); packing only (every pack verified bit for bit) | 154.96 -> 159.93: parity within noise (stock's 2 calls 149.14 and 160.78; run 2: 0.998x) | 1338.97 -> 1350.37 (1.01x) | 3 (`bcdf1746`, verify on) |
| Qwen/Qwen3.6-35B-A3B-FP8 | FP8 block | 34.25 -> 33.39 | 0.86 GiB (2.5%) | 3.08 (89,526) -> 3.94 (114,102) | 33.72, 118,198 | bit for bit (111 of 111, + image) | 130.65 -> 137.62 (1.05x) | 952.70 -> 990.36 (1.04x) | 3 (`bcdf1746`, verify on) |
| RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic | FP8 dynamic (compressed-tensors) | 25.80 -> 25.19 | 0.61 GiB (2.4%) | 11.64 (55,378) -> 12.32 (58,592) | 25.63, 58,488 | bit for bit (136 of 136, + image) | 111.15 -> 120.38 (1.08x) | 1045.14 -> 1059.25 (1.01x) | 1 (`5c7ce578`, verify off) |

- **The Gemma rows are of earlier commits.** Gemma-4-31B-IT-NVFP4's memory and speed are run 2a's (122 packs: its tied embedding and LM
  head packed, after the sub-model fix `bcb688da`); its exact check is run 1's (120 packs), the configuration exact mode has at every
  commit since (exact mode keeps a tied embedding and LM head as vLLM runs them), but run 2a's own runs had no exact check. Run 2a's raw
  vLLM logs are lost (only its parsed `*.json` remain); the same numbers stand in runs with logs: run 1's stock (31.18 GiB, 13,935 tokens,
  21.82 and 233.88 tok/s) and run 2's Glyd (25.25 GiB, 28,104 tokens, 26.09 and 421.96 tok/s, verify on). The two compressed-tensors Gemma
  rows are run 1's alone: before the sub-model fix and main's padded rows (`bcdf1746`), so their vision towers' 4,304-row Linears (0.50 GiB
  on gemma-4-26B) stayed bf16 then; at this PR's head they are packed, never measured or checked exact.
- Before `bcdf1746` (runs 1 and 2) the five Qwen checkpoints kept their embedding (0.95-2.37 GiB) and vision Linears bf16 and saved 0.15-1.29
  GiB (Qwen3.8-27B-FP8 0.91, AWQ 0.93, GPTQ 1.29, Qwen3.6-35B-A3B-FP8 0.49, Qwen3.6-35B-A3B-NVFP4 0.15): those runs' directories hold them.
- GPTQ: its weights fell 1.15 GiB but its KV cache grew 0.97 GiB (15.62 -> 16.59): about 0.18 GiB more is held outside the weights. Run 3
  verified every pack as it was made, and vLLM's count is 0.60 GiB above the model's own tensors there, as in run 1's exact run (0.56); run
  1's default-mode run, without verify, was 0.07 above them (19.83 GiB, KV 15.62 -> 16.74 GiB at that commit). What holds it was not looked into
  (bf16 path code).

## Results: RTX 5090 (32 GB, Blackwell)

Stock vLLM -> Glyd on the same checkpoint, same pod. Runs: `rtx5090-run4/` (commit `8d498a63`) and `rtx5090-run5/` (`ae0fdaf7`). In both,
stock's runs and Glyd's alike, FlashInfer's NVFP4 and FP8 GEMMs are off (`VLLM_DISABLED_KERNELS`, the run's `qpenv`), so stock here is not
vLLM's default kernel choice: they compile on first use with nvcc, and the pod's pip CUDA toolkit has no `cublasLt.h` (its nvidia-cublas
13.1.1.3 wheel holds none). vLLM then runs its own CUTLASS FP4 GEMM (Gemma-4-31B-IT-NVFP4: NVFP4 weights and activations, in hardware) and
CUTLASS FP8 GEMM (Qwen3.6-35B-A3B-NVFP4's attention); Qwen3.6-35B-A3B-NVFP4's experts and LM head are weight-only NVFP4 and run on Marlin here
as on Ada. Both runs are of commits before `bcdf1746`: Qwen3.6-35B-A3B-NVFP4's embedding (0.95 GiB) was not packed in them (on the L40S, run 3,
it adds 0.38 GiB to the saving).

**The layout in these runs is not the default any more.** The rows below without a named layout ran the RTX 5090's own (12-bit) layout,
then the default for a quantized checkpoint. From this PR's last code commit, layout `auto` is the tiered layout for a quantized checkpoint
on every GPU (the 12-bit one kept 98.4% of gemma-4-12B QAT's embedding, saving 0.04 GiB against the tiered layout's 0.65 on the L40S).
gemma-4-12B QAT and Qwen3.6-35B-A3B-NVFP4 were not measured again with it on the RTX 5090 (see "Not done"); Gemma-4-31B-IT-NVFP4's rows are
the tiered layout's, given by name then and taken by `auto` now.

| Checkpoint | Format (vLLM's kernel here) | Glyd's layout | Weights GiB | Saved | KV cache tokens | Requests at once at the context | Glyd exact vs stock eager | 1 user tok/s | 32 users tok/s | Run |
| :-- | :-- | :-- | --: | --: | --: | --: | :-- | --: | --: | :-- |
| nvidia/Gemma-4-31B-IT-NVFP4 at `--gpu-memory-utilization 0.95 --max-model-len 4096 --max-num-seqs 32` | NVFP4 W4A4 (CUTLASS FP4), bf16 attention | tiered (`layout: mma`) | stock: out of memory loading its 31.18 -> 25.25 | stock does not load | - -> 5,559 | - -> 1.36 of 4,096 tokens | no stock run to compare (122 of 122 packs verified) | - -> 42.05 | - -> 285.40 (KV-bound) | 5, `@u95` |
| the same at 0.97 | | tiered | - -> 25.25 | | - -> 7,044 | - -> 1.72 of 4,096 tokens | | - -> 42.01 | - -> 310.89 (KV-bound) | 5, `@u97` |
| the same at vLLM's 0.9 and 8,192 tokens | | 12-bit, then tiered | stock: out of memory -> 26.72 (12-bit), 25.25 (tiered) | | no room for a KV cache in either layout (tiered: -0.06 GiB) | | | | | 4 |
| google/gemma-4-12B-it-qat-w4a16-ct | INT4 W4A16 (compressed-tensors, Marlin) | 12-bit (the default then) | 8.28 -> 8.24 | 0.04 GiB (0.5%) | 55,503 -> 55,607 | 6.78 -> 6.79 of 8,192 | bit for bit (1 of 1 packs verified, + image) | 141.20 -> 139.54 (**0.988x**) | 1675.08 -> 1645.57 (**0.982x**) | 5 |
| nvidia/Qwen3.6-35B-A3B-NVFP4 (`--max-num-seqs 64`, the median of 3) | NVFP4 W4A16 experts and head (Marlin), FP8 attention (CUTLASS) | 12-bit (the default then) | 20.37 -> 20.24 | 0.13 GiB (0.6%) | 151,552 -> 153,892 | 18.50 -> 18.79 of 8,192 | not run here (no bit-for-bit result on the L40S) | 239.34 -> 239.35 (1.000x) | 2660.68 -> 2616.76 (**0.983x**) | 5 |

KV-bound: 5,559 and 7,044 tokens hold fewer than 32 requests of 768 tokens (24,576), so the 32-user cells are queueing, not throughput at
32 users.

- **NVFP4 W4A4 on the RTX 5090: exact with vLLM's CUTLASS FP4 GEMM only; one FAIL with vLLM's default (FlashInfer's).** nvidia/Qwen3.8-27B-NVFP4,
  this branch at `8da693ae`, the benchmark session's pods (merged record `research/report-quantized-2026-10-06/`): with FlashInfer's NVFP4 GEMM
  (`FlashInferCutlassNvFp4LinearKernel`, vLLM's default on this GPU; pod `qbm-5090-run4`), 1 of 2 Glyd exact runs differs from stock eager (the
  re-check, `records/rtx5090-qbm-run4/nvidia_Qwen3.8-27B-NVFP4-recheck`: the ~700-token prompt's logprob at position 520, -0.14144 against
  -0.07740) while stock eager equals its own second run in every field: **FAIL**, cause not found. With vLLM's CUTLASS FP4 GEMM
  (`CutlassNvFp4LinearKernel`, FlashInfer's NVFP4 kernels off, stock and Glyd; pod `qbm-5090b`): bit for bit, 83 of 83 packs verified. So on
  Blackwell this record's NVFP4 W4A4 rows (Gemma-4-31B-IT-NVFP4) are of the CUTLASS kernel; with vLLM's default kernel NVFP4 W4A4 is not
  shown exact.
- **Gemma-4-31B-IT-NVFP4 serves on a 32 GB card with Glyd and not without it**: stock vLLM runs out of memory loading its weights (31.18 GiB
  on the L40S; this GPU holds 31.36), at 0.9 and at 0.95; Glyd's tiered layout holds them in 25.25 GiB and serves a 4,096-token context at
  0.95 of the GPU (5,559 tokens of KV cache; 7,044 at 0.97), 42 tokens a second for one user. Not at vLLM's defaults: at 0.9 neither layout
  leaves room for a KV cache.
- **Slower than stock where little is packed** (12-bit layout): gemma-4-12B QAT 0.988x at one user and 0.982x at 32 (the 2 calls of each
  within 0.2%); Qwen3.6-35B-A3B-NVFP4 1.000x at one user and 0.983x at 32 (the 3 calls of each spread 2.3% and 2.1%, overlapping).
- **Bit for bit on the RTX 5090**: gemma-4-12B QAT (tokens, logprobs, prompt logprobs and the image answer). Gemma-4-31B-IT-NVFP4 has no
  stock run to compare with on this GPU (its exact check is the L40S's).
- **No default changes**: `glyd run` and `glyd serve` still refuse a quantized checkpoint; `vllm serve CHECKPOINT --quantization glyd` is
  the user's choice. The speed bar (at least stock's at every load measured) is not met on the RTX 5090 (three cells below stock, above)
  nor shown on the L40S (Qwen3.6-35B-A3B-NVFP4 at one user: parity within noise, 0.998x in run 2).

## What the L40S numbers say

- **Exact mode bit for bit with stock eager on 5 of 8 checkpoints, at their own bits**: Qwen3.8-27B-FP8 and Qwen3.6-35B-A3B-FP8 (run 3,
  `bcdf1746`), Gemma-4-31B-IT-NVFP4, gemma-4-26B FP8-dynamic and gemma-4-12B QAT (run 1, `5c7ce578`): tokens, logprobs, prompt logprobs and
  image answer equal, every pack decoded to its weights' bytes. The fp16 AWQ checkpoint is equal to stock at `--dtype bfloat16` but is not
  lossless: both round its fp16 parts to bf16. **No bit-for-bit result** for the two checkpoints whose experts run on vLLM's Marlin MoE kernel
  (Qwen3.5-35B-A3B-GPTQ-Int4: auto_gptq; Qwen3.6-35B-A3B-NVFP4: ModelOpt mixed): stock vLLM eager is not the same from one run to the next
  there (`stock-eager-2.json` differs from the first in tokens, logprobs, prompt logprobs and the image answer, with Triton's autotuned
  choices kept across runs), so these formats are covered by packing only: every pack verified bit for bit (run 3's default-mode runs: 272 of
  272 and 111 of 111), the end-to-end comparison undecided.
- **Less memory on all 8**, by the share of the checkpoint left 16-bit that Glyd reaches: 19.0% of the weights on Gemma-4-31B-IT-NVFP4
  (bf16 attention and a bf16 embedding), 9.1% on the AWQ checkpoint (its fp16 embedding and LM head, rounded to bf16 by vLLM), 2.4-7.9% on
  the others; the KV cache grows by what is freed, a little less on GPTQ (above): Gemma-4-31B-IT-NVFP4 2.0x the tokens (3.43 requests of
  8,192 tokens at once against 1.70), Qwen3.6-35B-A3B-FP8 1.27x, Qwen3.8-27B-FP8 1.20x. Exact mode holds more (the LM head stays vLLM's):
  Gemma-4-31B-IT-NVFP4 26.11 GiB against 25.25, Qwen3.8-27B-FP8 27.92 against 27.13.
- **Speed** (default mode; one start each, stock's first): at one user 1.03-1.24x stock's tokens a second, Qwen3.6-35B-A3B-NVFP4's 1.03x
  within stock's own spread (its two calls 149.14 and 160.78; Glyd's 159.19 and 160.67) and 0.998x in run 2, so parity there; at 32 users
  1.01-1.05x on 7 of 8. Gemma-4-31B-IT-NVFP4's 32-user cell is not a speed ratio: stock's KV cache holds 13,941 tokens, fewer than 32 requests
  of 768 need, so stock queues (Glyd 421.65 against 230.88 tokens a second there is the memory freed). Exact mode's compiled speed was not
  measured.
- **What stays 16-bit** (the inventories, `*/glyd.json`):
  - Gemma 4's vision tower in a ModelOpt checkpoint (1.03 GiB on Gemma-4-31B-IT-NVFP4): vLLM gives the tower the quantization config
    only for compressed-tensors, bitsandbytes and torchao checkpoints, and a tower vLLM builds without it stays bf16 for bf16 models too.
  - LM heads a checkpoint quantizes (Qwen3.6-35B-A3B-NVFP4's, NVFP4): the checkpoint's.
  - The formats' own 16-bit scales (GPTQ's `w13_scales`, compressed-tensors' `weight_scale`, FP8's `weight_scale_inv`): the checkpoint's
    format, left as vLLM holds them.
  - In exact mode the LM head stays vLLM's own matrix and product (the bf16 rule: its logits bit for bit, without a buffer of its size).
- Run 1's Gemma-4-31B-IT-NVFP4 kept its tied embedding and LM head (2.62 GiB) bf16 in its default mode: vLLM makes Gemma 4's language model as
  a sub-model under a copy of its config, which Glyd took for a speculative draft. Fixed for quantized checkpoints in `bcb688da`; run 2a
  measures it (5.07 -> 5.93 GiB saved). A bf16 Gemma 4 has the same cause (not changed here: bf16 work is paused).
- Not measured: MTP heads and speculative decoding on a quantized checkpoint (no run used `--speculative-config`; a draft's quantization
  config is covered by CPU tests only, `test_method_of_the_model`).
- The options a run recorded (`glyd_key`, vLLM's additional_config) are unresolved for Gemma-4-31B-IT-NVFP4 alone (`{"layout": "auto",
  "kv": "off"}`). `kv: off` there is the write `maybe_update_config` makes for the model's own config, followed by its refusals, so that path
  ran; the resolved options are most likely written into the copy of vLLM's config that Gemma 4's language model is made under, where its
  first Glyd layer is (a compressed-tensors Gemma 4's first one is in its vision tower, made under vLLM's own config, and its options are
  resolved). Not confirmed on a GPU. `quant_parts.py` now records the options of the model's own Glyd config (`inventory.opts`), and no
  longer sets `GLYD_KV` (every run here had `GLYD_KV=off`).

## Not done

- No pod ran after the review: the Runpod balance was $14.82 against the guard's $12 floor for new pods, with three pods running and others
  queued. Not run, for that reason: the three Gemma rows at this PR's head (all four steps, verify off, starts in the order A B B A, 3 calls
  each); gemma-4-12B QAT and Qwen3.6-35B-A3B-NVFP4 on the RTX 5090 with the tiered layout `auto` now takes; NVFP4 W4A4 on the RTX 5090
  with vLLM's default (FlashInfer) kernels (it needs `cublasLt.h`, which the pod's pip CUDA lacked); the GPTQ row with verify off at this
  commit.

## Runs

    GPU="NVIDIA L40S" MAX_MIN=120 NAME=qbp-l40s3 RP_DISK=250 UPLOAD="qp_src.tar qpenv" bash rp_run.sh job.sh
    GPU="NVIDIA GeForce RTX 5090" MAX_MIN=80 NAME=qbp-5090r5 RP_DISK=150 UPLOAD="qp_src.tar qpenv" bash rp_run.sh job.sh

`job.sh` (this directory): vLLM 0.30.0 and nvcc from pip, the library built from `gpu/glyd_gpu.cu` for the pod's GPU alone, the plugin's
tests against the real vLLM (`test_vllm_quant.py` must pass or nothing runs), then `quant_parts.py` on each checkpoint of `qpenv` (in each
run directory) as its download finishes. `qp_src.tar`: `git archive` of the commit's `glyd-gpu-py/`, `gpu/glyd_gpu.cu`, `gpu/glyd_gpu.h`,
`gpu/vllm/quant_parts.py`. `table.py GPU DIR ...` makes a table from the `summary.json` files. The runs here are of `quant_parts.py` before its
last change (default-mode run verified from run 2 on, `GLYD_KV=off`, one start each, 2 calls); it now measures without verify (verify in the
exact run only), starts stock and Glyd twice in the order A B B A with 3 calls each, and leaves `GLYD_KV` unset.

Raw logs: each run directory holds vLLM's own log of every step (`<checkpoint>/<step>.log`, the source of "Model loading took", the KV
cache and the concurrency) and the pod's `job.log`, except `l40s-run2a/`, whose raw logs are lost (its `*.json` hold the values parsed from
them; runs 1 and 2 hold the same numbers with their logs, above). `5090-attempts/` measured nothing and keeps its step notes only.

- `l40s-run1/`: all 8 steps of 7 checkpoints. Qwen3.8-27B-FP8's and Qwen3.6-35B-A3B-FP8's compiled runs did not start, stock's and
  Glyd's alike: vLLM's default 256 sequences need more Mamba cache blocks than the L40S had left (164 and 120 for stock).
- `l40s-run2a/`: Gemma-4-31B-IT-NVFP4 stock and Glyd compiled with the table fix; stopped while the next checkpoint loaded, to add the
  compiled run's verification.
- `l40s-run2/`: the two FP8 checkpoints compiled at `--max-num-seqs 64` (both), Qwen3.6-35B-A3B-NVFP4 (all steps), GPTQ's exact check again
  with Triton's autotuning kept across runs (`TRITON_CACHE_AUTOTUNING=1`, still undecided), Gemma-4-31B-IT-NVFP4's compiled Glyd run verified.
- `5090-attempts/`: three RTX 5090 runs that measured nothing: the library did not build (the job's sources lacked `glyd_gpu.h`); the
  plugin's tests failed on this GPU (the FP8 development path's GPU rule, in a routing test since made GPU-independent); FlashInfer's
  FP8 GEMM failed to compile on first use (the pod's pip CUDA toolkit has no `cublasLt.h`).
- `l40s-run3/`: the five Qwen checkpoints at `bcdf1746` (their embedding packed after the load): all steps on the three whose exact check
  decides, stock and Glyd compiled on the two MoE checkpoints whose stock eager differs from itself.
- `rtx5090-run4/`: Gemma-4-31B-IT-NVFP4 at vLLM's 0.9 and 8,192 tokens (stock out of memory; Glyd's 12-bit and tiered layouts without
  room for a KV cache); ended while the next checkpoint loaded (its launcher stopped with the session that ran it).
- `rtx5090-run5/`: Gemma-4-31B-IT-NVFP4 at 0.95 and 0.97 with a 4,096-token context (`@u95`, `@u97`: the same checkpoint again with other
  flags), gemma-4-12B QAT (all steps), Qwen3.6-35B-A3B-NVFP4 (stock and Glyd compiled, the median of 3). The pod's pip downloads took
  22 minutes of its 58.
- A first L40S run (`qbp-l40s`, 3.6 min) ended with this session before any checkpoint ran.

## Spend

Runpod, from the account's pod ledger (each pod's time at its hourly price): L40S, 5 pods, 4.70 h at $1.09: $5.12 (run 3: 83 min, $1.51);
RTX 5090, 6 pods, 1.74 h at $0.99: $1.73 (run 5: 59 min, $0.98). Total **$6.85**.
