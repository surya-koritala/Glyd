# FP8 checkpoints under vLLM with their FP8 weights packed lossless (2026-10-06)

Private record. vLLM 0.30.0 serving 2026 FP8 checkpoints with `--quantization glyd` (`GLYD_FP8=1`, a development build's opt-in): every FP8
Linear's and expert's weight, as vLLM's own FP8 kernel is fed it, held in the lossless FP8 format (`f8codec`: fp8ref layout 2) and decoded
bit for bit into one scratch before each product, which vLLM's own kernel then runs; for a mixture of experts only the experts a step routes
to are decoded (`f8_unpack_experts`). Stock vLLM and Glyd (`exact` on) on the same checkpoint, on the same pod, one after the other:
`gpu/fp8/check_fp8_vllm.py` (job: `job/q8_job.sh`; each run's settings: `job/env-*.txt`).

The main record is `l40s-d/`, run at commit `b800b5dc` (the code of this PR, latest private main merged). The other directories are earlier
runs at earlier commits, kept as measured (Files, below).

## Result: L40S (Ada, 48 GB), `l40s-d/`

| Checkpoint | Weights GiB, stock / Glyd (less) | FP8 codes, bits a value | KV cache tokens, stock / Glyd | Users at 8,192 tokens, stock / Glyd | Outputs | Output tok/s, 1 user: stock (vLLM's defaults) / Glyd | 8 users: Glyd | 32 users: stock (vLLM's defaults) / Glyd |
| :-- | --: | --: | --: | --: | :-- | --: | --: | --: |
| Qwen/Qwen3.6-35B-A3B-FP8 | 34.25 / 30.15 (12.0%) | 6.725 | 113,943 / 268,101 | 13.9 / 32.7 | bit for bit (a) | 130.5 / 83.9 (0.64) | 237.5 | 734.5 / 428.7 (0.58) (b) |
| Qwen/Qwen3.8-27B-FP8 | 28.90 / 25.63 (11.3%) | 6.745 | 104,740 / 143,360 | 12.8 / 17.5 | bit for bit | 24.5 / 11.1 (0.45) | 74.9 | 292.2 / 196.7 (0.67) |
| RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic (c) | 25.80 / 22.94 (11.1%) | 6.715 | 58,052 / 71,285 | 7.1 / 8.7 | bit for bit | 109.1 / 80.0 (0.73) | 239.8 | 847.8 / 499.1 (0.59) |
| RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8 (c) | 30.20 / 26.84 (11.1%) | 6.717 | 631,808 / 915,456 | 77.1 / 111.8 | bit for bit | 141.7 / 91.8 (0.65) | 207.7 | 907.8 / 428.0 (0.47) |

- Weights: what vLLM logs as "Model loading took": the FP8 Linears and experts, the 16-bit parts the checkpoint keeps (this path leaves them
  as vLLM has them) and Glyd's scratches (the Linears' one, the size of the largest Linear, and for a mixture of experts one layer's experts:
  0.17-1.22 GiB). The FP8 bytes themselves are 15.7-16.1% smaller (6.715-6.745 bits a value); the whole loaded model 11.1-12.0%. The weights
  check (stock's less what the packs save, plus the scratches, within 0.25 GiB) passes on all four.
- Outputs: 4 prompts in one batch (6, 15, 32 and 1,860 tokens; Nemotron's tokenizer 6, 15, 31, 1,860; gemma in its chat template, 19, 30,
  45, 1,873), greedy, 48 tokens out: the tokens, their log probabilities and the top 20 prompt log probabilities of every prompt position
  (38,569-40,437 values) equal bit for bit; `check_fp8_vllm.json` holds each run's SHA-256 of them (`sha256`). The greedy tokens of each speed
  run's first batch (1 and 32 requests of 1,024 tokens, 256 out) are the same too (with (a)'s rerun for Qwen3.6-35B-A3B's 32). Compiled with
  inductor's deterministic mode and CUDA graphs (FULL_AND_PIECEWISE) in both runs, plus (c)'s flags: exactness is checked there, not at vLLM's
  default compile. The same FP8 kernels in both (vLLM's lines: Marlin FP8 for Qwen's block-scaled Linears, CUTLASS scaled_mm for the
  compressed-tensors ones, Triton for the experts).
- `--poison`, all four: every pack decoded back at load bit for bit the bytes it was made from (240, 256, 180 and 150 packs: every FP8 Linear
  and expert stack), and with the Linears' scratch overwritten after every product and the experts buffer set to e4m3 NaN before every call,
  the outputs are still stock's bit for bit.
- (a) Qwen3.6-35B-A3B-FP8, 32 requests: with each run's own KV cache, requests 30 and 31 differ from stock's from tokens 43 and 45. Stock's
  153 KV blocks admit 30 of the 32 requests at once (a hybrid model's request takes a block for each group of its linear-attention layers),
  Glyd's 360 blocks all 32, so the last two run in other batches. Glyd again with stock's 153 blocks (`glyd-graph-sameblocks`): the same
  tokens as stock. (This was the FAIL of the earlier run `l40s-c/`, which had no such rerun.)
- (b) Stock runs 30 of the 32 at once (a). Glyd with stock's 153 blocks (the same batches): 372.5 tok/s, 0.51 of stock's 732.3 in the same
  compile.
- (c) Per-channel W8A8 checkpoints (compressed-tensors, CUTLASS scaled_mm): compiled with vLLM's activation quantization op on and its norm
  and activation quantization fusions off, in both runs (`custom_ops: ["+quant_fp8"]`, `pass_config: {fuse_norm_quant: false, fuse_act_quant:
  false}`). With the fusions on (the default once the op is on), vLLM fuses the quantization of a Linear's input into the RMSNorm or the
  activation before it, and Glyd's product, which quantizes inside its op, differs from stock from the second token
  (`l40s/RedHatAI__gemma-4-26B-A4B-it-FP8-dynamic-fusions-on/`; stock run twice there gives the same bits). `exact` is refused under
  torch.compile on such a checkpoint unless the fusions are off (`vllm_fp8.compiled_exact`).
- Users: vLLM's "Maximum concurrency for 8,192 tokens per request". The KV cache gains what the weights lose, and more where the scratch is
  small against the model (Qwen3.6-35B-A3B: 2.35 times the tokens).
- Speed: output tokens a second, 1,024-token prompts and 256 tokens out each, the median of 3. Stock's reference is a run of its own at vLLM's
  default compile (`<checkpoint>/serve/`: `--modes serve --only stock`, nothing set, Triton's autotuners free). Glyd's figures are its exact
  runs (inductor's deterministic mode, Triton's autotuners kept at their first config, (c)'s flags); stock in that same compile, 1 user / 32
  users: Qwen3.6-35B-A3B 130.9 / 732.3, Qwen3.8-27B 24.5 / 291.9, gemma 106.5 / 869.6, Nemotron 140.0 / 920.0, so that compile does not slow
  stock here. Glyd is 0.45-0.73 of stock at vLLM's defaults (0.51 at Qwen3.6-35B-A3B's 32 users in the same batches):
  every product reads the pack, writes the weight out whole and reads it again. The FP8 path stays opt-in (a development build's
  `GLYD_FP8=1`): it does not meet the bar (at least stock's speed at every load).
- Memory while serving: torch's peak over its start during the exact check's generate (its prompt log probabilities included, which vLLM's
  memory profile does not count) is the same for stock and Glyd on all four (0.89, 0.91, 1.43, 0.48 GiB): no extra memory at run time here.
- The gates on this GPU at this commit: `check_f8.log` (the format's pack, unpack, experts' unpack, f8_linear and their CUDA graphs: every case
  ok), `tests.txt` (the plugin's FP8 tests against vLLM 0.30.0: 21 ok).

## RTX 5090 (32 GB): earlier runs, not repeated at this commit

| Checkpoint | Run | Weights GiB, stock / Glyd (less) | KV cache tokens, stock / Glyd | Users, stock / Glyd | Outputs | Output tok/s, 1 user: stock / Glyd | 8 users: Glyd | 32 users: stock / Glyd |
| :-- | :-- | --: | --: | --: | :-- | --: | --: | --: |
| RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic (c) | `rtx5090-b/`, `a2907d7d`, utilization 0.93, `--max-model-len 4096` | 25.80 / 22.94 (11.1%) | 7,867 / 21,567 | 1.9 / 5.3 at 4,096 | bit for bit (38,671 values) | 202.7 / 124.5 (0.61) | 529.1 | 736.1 / 625.7, KV-bound (d) |
| Qwen/Qwen3.8-27B-FP8 | `rtx5090-c/…-full`, `39400c40`, utilization 0.95, `--max-num-seqs 32` | 28.51, then out of memory / 25.15 (11.8%) | does not start / 24,576 | - / 3.0 at 8,192 | not compared (e) | | | |

- Stock on the RTX 5090 is in the same compile as Glyd (`graphq`); no run at vLLM's defaults there.
- (d) At 4,096 tokens stock's 7,867-token cache holds about 6 of the 32 requests (1,280 tokens each) at once, Glyd's 21,567 about 16: the two
  run different batch sizes, so 0.85 there is the larger cache, not a like-for-like speed.
- (e) Qwen3.8-27B-FP8, the whole model, at utilization 0.95: stock loads 28.51 GiB and runs out of memory profiling the vision encoder; Glyd
  loads 25.15 GiB (same pod), profiles (vLLM: 24,576 KV tokens, 3.0 requests of 8,192) and serves the check's 4 prompts; every pack decoded
  back at load bit for bit (256), and its outputs with the scratch poisoned equal its own (`sha256.json`). There is no stock output to compare
  them with, and vLLM runs these Linears there with CUTLASS's block-scaled W8A8 kernel (`CutlassFp8BlockScaledMMKernel`), which no run here
  compares with stock: from commit `b800b5dc` `exact` refuses that kernel ("not measured against stock vLLM yet"), so on the RTX 5090 this
  checkpoint runs with exact off. Measured: it loads and serves on a 32 GB card with Glyd where stock does not start; not measured: that its
  outputs are stock's. Language model only: at 0.9 (`rtx5090-b/`) stock 27.64 / Glyd 24.27 GiB and neither starts (cache memory -4.73 / -1.51
  GiB); at 0.95 (`rtx5090-c/…-lm`) neither either (stock -3.16 GiB; Glyd 0.07 GiB, where one 8,192-token request needs 0.67).
- gemma at 0.95 and 8,192 tokens (`rtx5090-a/`, `rtx5090-c/`): stock starts (10,398 and 10,459 KV tokens), Glyd runs out of memory
  allocating 446 MiB for the check's prompt log probabilities. Not explained: on the L40S the two use the same memory over their start in that
  step (above).
- RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8 does not load on the RTX 5090 in either run (out of memory allocating 630 MiB while
  loading, `rtx5090-a/`): Glyd packs a layer once vLLM has loaded the whole checkpoint, so loading needs stock's memory (30.2 GiB of weights).

## Not measured, not done

- f8_linear (the product from the pack, the weight never written out) is off in every build: every step here took the decode and vLLM's
  product. Only a development check's `GLYD_F8_FUSED_MAX` turns it on (`check_fp8_vllm --fused`, not run here); it does not quantize the
  activations (W8A16), so it is not a W8A8 checkpoint's numerics. #44's record (`benchmarks/gpu/fp8-speed-consumer-2026-10-06`) has it at
  0.94-1.36 times vLLM's own FP8 product's time at 1-8 tokens on an RTX 4090 and an RTX 5090 and slower beyond. `vllm_fp8.fused_max` does
  not read main's `kernels.F8_LINEAR_TOKENS` (by GPU and kind of scales, no L40S); taking it in vLLM needs a run of its own end to end.
- `exact` refuses what was never compared with stock vLLM: CUTLASS's, Triton's, DeepGEMM's and FlashInfer's block-scaled W8A8 kernels (the RTX
  50 series and Hopper run Qwen's FP8 Linears with them), MXFP8 checkpoints and ModelOpt FP8 experts. Per-tensor `fp8` checkpoints, ModelOpt
  FP8 Linears and Hopper are not measured either.
- RTX 4090 (24 GB): not run. From the weights measured here (not a measurement on a 4090), none of these checkpoints fits 24 GB either way:
  the smallest Glyd weights are 22.94 GiB (gemma), before CUDA's context and the activations. And since Glyd packs only after vLLM has loaded
  the whole checkpoint, a checkpoint that does not load stock does not load with Glyd either on this path; packing layer by layer during the
  load is not done.
- The experts' scratch is one whole layer's experts (0.74-1.22 GiB); a buffer for the routed experts alone is not done.

## The 1.5 GiB on the 27B models (fixed)

The earlier runs' Qwen3.8-27B-FP8 weights were 27.04 GiB with Glyd (`l40s/`, `l40s-b/`): 1.5 GiB more than the packs and the scratch. vLLM
loads under `max_split_size_mb` 20, where a pack of 20 MiB or more took a freed weight's block whole (up to 20 MiB more than it asked for),
and vLLM counts the block. The development memory log (`GLYD_F8_MEMLOG=1`, a line a Linear) shows it: in `l40s-b/Qwen__Qwen3.8-27B-FP8/glyd-graph.log`
the allocated bytes grow by up to 7.9-8.8 MB more than the pack over each qkv_proj, o_proj and out_proj and 16.7 MB over each down_proj, 1.1 GB over the
256 Linears apart from the scratch; with the allocator's free blocks given back before each pack (`vllm_fp8._release`), 0.13 GB
(`l40s-c/Qwen__Qwen3.8-27B-FP8/glyd-graph.log`, `rtx5090-c/`), and the weights 25.63 GiB on the L40S (`l40s-c/`, `l40s-d/`).

## Files

- `l40s-d/` (main, `b800b5dc`): per checkpoint `check.txt` (the PASS/FAIL/INFO lines), `check_fp8_vllm.json` (every run: weights, KV
  cache, concurrency, kernels, speed, the packs' bytes, the outputs but the prompt log probabilities, and the SHA-256 of the outputs),
  `stock-*.log` and `glyd-*.log` (vLLM's logs, with the memory log; `-sameblocks`, `-poison`), `serve/` (stock at vLLM's defaults);
  `check_f8.log`, `tests.txt`, `steps.txt`, `env.txt`, `log/` (build, downloads).
- Earlier runs, each with its `COMMIT`, kept as measured:
  - `l40s/` (`52f32bce`): Qwen3.6-35B-A3B, Qwen3.8-27B and Qwen3.6-27B-FP8 before the allocator fix (Qwen3.6-27B-FP8: 28.90 / 27.03 GiB,
    the same as Qwen3.8-27B's, measured only here), gemma with the fusions on ((c)).
  - `l40s-b/` (`7a6501f5`): Qwen3.8-27B Glyd alone with the memory log before the fix, gemma and Nemotron with the fusions off.
  - `l40s-c/` (`5e1446c6`): the four checkpoints after the fix, with `--poison`; the 32-user FAIL of (a); `sha256.json` from its runs' JSONs.
  - `rtx5090-a/` (`1959f272`), `rtx5090-b/` (`a2907d7d`), `rtx5090-c/` (`39400c40`, `sha256.json`): the RTX 5090 runs above.
  - On the earlier pods `check_f8.py`'s case_linear failed (`f8_linear: invalid argument`, fixed by #44, merged) and on the first four its
    case_graph (an empty capture's own allocation, fixed on main); at `b800b5dc` every case passes.
- Two L40S tries stopped before any checkpoint ran (the job's gate counted f8_linear's check; the plugin's entry point was not declared for
  the tree's package); their minutes are in the spend.

Spend: Runpod $5.33 for nine pods (the guard's ledger: L40S $0.11 + $0.16 (the two stopped tries) + $1.10 + $0.40 + $0.76 + $1.57 (`l40s-d`),
RTX 5090 $0.32 + $0.44 + $0.47).
