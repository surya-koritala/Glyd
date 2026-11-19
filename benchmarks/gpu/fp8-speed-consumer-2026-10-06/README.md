# f8_linear on consumer GPUs: RTX 5090 and RTX 4090, against vLLM 0.30's own FP8 path (2026-10-06)

Private record. `f8_linear` (the product of 1-64 tokens straight from the lossless FP8 pack, the weight never written out) timed against vLLM
0.30.0's own FP8 kernel, and against the decode route (the pack decoded into a scratch, then vLLM's kernel on it, both in one CUDA graph), at the
Linear shapes of 2026 FP8 checkpoints. `gpu/fp8/bench_f8.py` and `gpu/fp8/check_f8.py`; real weight bytes from the checkpoints
(`gpu/fp8/fetch_samples.py`); one CUDA graph a runner, best of 5. Round 1 (commit 3e60e481 and the fp8-speed branch's a6c4240f) chose defaults that
review rejected; round 2 (8df2d686, 851cf2e8, e1970149, 72872c13: the library is e1970149's) is the code PR #44 merged. Round 3 (branch
fp8-speed-review, after PR #44's review, merged in PR #52 as 6228590a) fixed the bench and the checks; its first RTX 5090 pod was refused by the
Runpod guard, and it ran on one RTX 5090 pod at 6228590a (`rtx5090-6228590a/`, `job3.sh`; the kernels are e1970149's: round 3 changed comments
only in `glyd_gpu.cu`).

## Result

- **Not faster than vLLM's own FP8 kernels at every load, so nothing here is a default or in a served path.** f8_linear is slower on every shape
  measured at 32 and 64 tokens (Mistral-Small-4's unconfirmed RTX 5090 kernel aside, below); at 1-8 tokens it is at or under vLLM's time on some
  shapes and models on the RTX 4090 and on some shapes (no model's sum) on the RTX 5090. A release library has one
  plan (bf16 X, the grid of the most CTAs); W8A8, at least m patches a CTA, a step ahead and the spills' pass are development builds' (`GLYD_F8_LIN_*`),
  W8A8 on compute capability 12.x only. Nothing calls `f8_linear` by default: `vllm_fp8.py` on main is an identity codec, and PR #37's fused path is a
  development variable.
  - RTX 4090 (851cf2e8, the same kernel as e1970149's default plan), per-shape sums of f8_linear's times over vLLM's at tensor-parallel 1, seven
    checkpoints: 0.94-1.16 at 1 token, 0.96-1.33 at 8, 1.11-1.65 at 32, 1.68-1.96 at 64. At or under vLLM: Qwen3.8-27B-FP8 (0.94, 0.96) and
    MiniMax-M3-MXFP8 (0.95, 0.99) against Marlin W8A16 at 1 and 8 tokens; at tensor-parallel 1-8 (0.83-1.47, 0.96-1.75, 1.11-2.45, 1.68-2.63) also
    Mistral-Small-4-119B-2603 at tensor-parallel 8 and 1 token (0.83, against CUTLASS W8A8). Per shape at most vLLM's time: 22 of 100 at 1 token,
    12 of 100 at 8, none at 32 or 64 (per shape 0.63-1.68, 0.91-2.06, 1.02-3.01, 1.49-2.63).
  - RTX 5090 (72872c13): **only gemma-4-26B-A4B-it-FP8-dynamic has a valid vLLM number**: 1.22 at 1 token, 1.33 at 8, 2.15 at 32, 2.62 at 64 (per
    shape 1.00-1.41 at 1 token, none under vLLM's time), against CutlassFP8ScaledMMLinearKernel, which a served run selects there ("Selected
    CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8" on an RTX 5090: `research/report-quantized-2026-10-06/records/rtx5090-qf8-run2/`).
    **The block-scale rows (Qwen3.6-35B-A3B-FP8 at tensor-parallel 1 and 2, Qwen3.8-27B-FP8) are withdrawn:** bench_f8.py stood in model type
    qwen3 for every model, so vLLM picked DeepGemmFp8BlockScaledMMKernel with UE8M0 scales, but stock vLLM 0.30 turns DeepGEMM off on compute
    capability 10.x/12.x for these checkpoints' model types (qwen3_5_moe_text, qwen3_5_text: `vllm/utils/deep_gemm.py`'s
    `_DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES`; "Falling back to CUTLASS") and serves them with its CUTLASS block-scaled kernel. The same holds for
    round 1's and 8df2d686's RTX 5090 runs (`rtx5090/`, `rtx5090-before/`, `rtx5090-8df2d686/`). Mistral-Small-4-119B-2603 has no vLLM number
    on the 5090 at 72872c13 (below).
  - **RTX 5090 (6228590a, round 3's bench), the block-scale rows against stock's kernel:** stock vLLM 0.30 started on each checkpoint here
    (`bench_f8.py --served`) selected CutlassFp8BlockScaledMMKernel for Qwen3.6-35B-A3B-FP8 and Qwen3.8-27B-FP8 ("Auto-disabled DeepGemm for
    model_type=qwen3_5_moe_text / qwen3_5_text on Blackwell ... Falling back to CUTLASS") and CutlassFP8ScaledMMLinearKernel for
    gemma-4-26B-A4B-it-FP8-dynamic, and the bench timed against the same kernels. Per-shape sums of f8_linear's times over vLLM's:

    | model | tp | vLLM's kernel | M=1 | M=8 | M=32 | M=64 |
    | :--- | ---: | :--- | ---: | ---: | ---: | ---: |
    | Qwen3.6-35B-A3B-FP8 | 1 | CUTLASS block-scaled | 1.19 | 1.24 | 1.96 | 2.75 |
    | Qwen3.6-35B-A3B-FP8 | 2 | CUTLASS block-scaled | 1.17 | 1.25 | 2.15 | 3.31 |
    | Qwen3.8-27B-FP8 | 1 | CUTLASS block-scaled | 1.03 | 1.04 | 1.83 | 2.43 |
    | gemma-4-26B-A4B-it-FP8-dynamic | 1 | CUTLASS W8A8 | 1.22 | 1.34 | 2.21 | 2.69 |

    Slower than stock's kernel in every per-shape sum. Per shape at most vLLM's time (block): 5 of 19 at 1 token, 3 of 19 at 8, none at 32 or 64
    (0.69-1.49, 0.85-1.49, 1.62-2.47, 2.33-3.96): Qwen3.6-35B-A3B's o (0.90 at tp 1, 0.92 at tp 2) and expert_gate_up (0.73 and 0.89; 0.69 and
    0.85 at tp 2), Qwen3.8-27B's down (0.96, 0.97). gemma's sums repeat 72872c13's (1.22, 1.33, 2.15, 2.62) within 3%. Nothing was left out of
    the sums (every copy set held at least 3 L2s).
  - **RTX 5090 (6228590a), Mistral-Small-4-119B-2603 (per-tensor scale): measured, not a stock comparison.** vLLM's per-tensor kernel ran this
    time (FlashInferFP8ScaledMMLinearKernel, FlashInfer's `bmm_fp8` built with `libcublas.so` and `libcublasLt.so` linked as `job3.sh` does), and
    f8_linear's per-shape sums over it are 0.35, 0.41, 0.67, 1.02 at tensor-parallel 1 (0.32, 0.40, 0.65, 1.02 at 2). But stock vLLM 0.30 did not
    start on the checkpoint (`--served`: "ImportError: cannot import name 'PixtralRotaryEmbedding' from 'transformers.models.pixtral...'"), so no
    served run confirms the kernel; its times do not change with the tokens (4096 x 4096: 51.9-52.2 us at 1-64 tokens; 1344 x 4096: 58.4-59.6
    us), and its numerics gave NaN on qkv_a at 1 token. These rows are left out of every claim and of `F8_LINEAR_TOKENS`.
- **Against the decode route** (the data of `glyd_gpu.kernels.F8_LINEAR_TOKENS`: the most tokens a step at which f8_linear was faster than the route,
  by GPU and kind of scales, opt-in data for a caller's fused path, since both are slower than stock vLLM): per-shape sums of f8_linear's times over
  the route's, every model and tensor-parallel size run, shapes left out as in `tables.py` (below).

  | GPU | kind (models) | M=1 | M=8 | M=32 | M=64 | F8_LINEAR_TOKENS |
  | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
  | RTX 4090 | block (GLM-5.3, Qwen3.6-35B, Qwen3.8-27B) | 0.40-0.96 | 0.42-1.13 | 0.50-1.59 | 0.77-1.77 | 1 |
  | RTX 4090 | tensor (Mistral-Small-4) | 0.59-0.79 | 0.88-0.92 | 1.14-1.81 | 1.16-2.19 | 8 |
  | RTX 4090 | mx (MiniMax-M3) | 0.46-0.95 | 0.48-1.07 | 0.72-1.33 | 0.85-1.50 | 1 |
  | RTX 4090 | ds32 (DeepSeek-V4.1) | 0.70-0.83 | 0.75-1.00 | 1.00-1.34 | 1.10-1.48 | 1 |
  | RTX 4090 | channel (gemma-4) | 0.78 | 0.85 | 1.03 | 1.23 | 8 |
  | RTX 5090 | channel (gemma-4) | 0.76 | 0.87 | 1.39 | 1.71 | 8 |
  | RTX 5090 (6228590a) | channel (gemma-4) | 0.72 | 0.83 | 1.35 | 1.69 | 8 (unchanged) |
  | RTX 5090 (6228590a) | block (Qwen3.6-35B, Qwen3.8-27B) | 0.48-0.70 | 0.50-0.77 | 0.89-1.31 | 1.13-2.02 | - (not entered) |
  | RTX 5090 (6228590a) | tensor (Mistral-Small-4) | 0.27-0.28 | 0.33-0.34 | 0.54-0.55 | 0.84-0.85 | - (vLLM's kernel unconfirmed) |

  Per shape the route is faster at 8 tokens on 24 of 100 RTX 4090 shapes (up to 1.27x; block 9, ds32 7, mx 7, tensor 1 at 1.00) and on 1 of 7
  on the RTX 5090 at 72872c13 (gemma's 2816 x 2112 mlp_down, 1.03; 0.96 at 6228590a); at 1 token on one RTX 4090 block shape (1.04). At
  6228590a the route is faster on no RTX 5090 block or channel shape at 1 or 8 tokens (at 8 at most 0.88 and 0.96). Between 1 and 8 tokens and
  between 8 and 32 nothing was measured, so a kind that loses anywhere at 8 tokens gets 1. The RTX 5090's block scales are not entered in
  `F8_LINEAR_TOKENS` although f8_linear is faster than the route at 1 and 8 tokens there: on that GPU f8_linear is above stock vLLM's own time
  in every per-shape sum (above). The route itself is 1.16-2.88 times vLLM's time per shape on the RTX 4090 (per-model sums 1.20-2.33) and
  1.32-1.67 on the RTX 5090's gemma shapes (sums 1.52-1.60); at 6228590a 1.32-2.62 on the RTX 5090's block shapes (sums 1.57-2.16) and 1.38-1.75
  on gemma's (sums 1.59-1.69). Ties: differences under 2% (round 1 measured the same kernel twice at 0.997-1.021 of itself).
- **Two defects the review's checks found, both fixed in PR #44 (every f8_linear path had them, main's too):**
  1. A plain patch's last row block was read with 16-byte loads at a word of `var` that is 16-byte aligned only by chance (hard fault on a GPU).
     Now read by words. check_f8 puts plain patches at the first and the last patch and requires some at unaligned words (19 on each GPU, 11 in
     the CPU emulation, whose loads assert their alignment).
  2. Activations of about 2^-16 of the usual and below lost bits: the bf16 weights are 2^-120 times the e4m3 values, so x w fell into fp32's
     subnormal range (on the RTX 5090 at 8df2d686, x at 2^-20 of the usual: 267 of 2,048 outputs outside the bound, 1.2% off). X goes into the
     MMAs times 2^64 (exact; one bf16 FMA a register of X) and the sums out times 2^56: checked from 2^-40 to 2^20 of the usual.
- **W8A8 (development builds) is not within f8_linear's bound on Ada:** on the RTX 4090 one output in 1,690 (tensor scale, 130 x 300, 13 tokens)
  was 1% off where the sum of |x w| was 100 times it (the e4m3 MMA's sums), so `f8_lin_a8` takes it on compute capability 12.x only.
- **Numerics** (`bench_f8.py --numerics`, rms error over the rms of the exact float64 product): f8_linear 1.60-1.72e-3 for every kind on both
  GPUs; vLLM's CUTLASS W8A8 (tensor and row scales on the RTX 4090, row scales on the RTX 5090) 2.6-3.0e-2; its Marlin W8A16 (block scales, MXFP8
  and DeepSeek's on the RTX 4090) 1.6-2.3e-3. The RTX 5090's DeepGEMM rows (3.7-3.8e-2) are withdrawn with the timings: not the kernel stock vLLM
  serves those checkpoints with. At 6228590a on the RTX 5090: stock's CUTLASS block-scaled kernel 2.46-2.60e-2 (Qwen3.6-35B, Qwen3.8-27B), CUTLASS
  W8A8 2.58-2.69e-2 (gemma), FlashInfer's per-tensor kernel 2.57-2.76e-2 and NaN once (Mistral-Small-4's qkv_a at 1 token); f8_linear 1.61-1.68e-3
  on all four.
- **vLLM's per-tensor FP8 kernel on the 5090:** round 2's link found no `libcublas.so` and `libcublasLt.so` (`-lcublas -lcublasLt`) in the CUDA
  wheels; round 3's `job3.sh` links both beside nvcc's libraries and FlashInfer's `bmm_fp8` built and ran at 6228590a (times above), but stock
  vLLM did not start Mistral-Small-4-119B-2603 to confirm it is the kernel a served run selects.
- **compute-sanitizer:** memcheck 0 errors and racecheck 0 hazards on `check_f8.py --only case_lin_quick` (every kind; 1, 33 and 64 tokens; plain
  patches at unaligned words; K split between 7 CTAs; the default and the development plans) on the RTX 4090 (e1970149) and the RTX 5090
  (72872c13), and on the 4090 again at 72872c13, with the pod image's CUDA 12.8 compute-sanitizer (the CUDA 13.0 wheel's could not start a
  process there). Round 3's quick cases (with dense patches and a step ahead with the spills' pass) on the RTX 5090 at 6228590a: memcheck 0
  errors, racecheck 0 hazards, the image's sanitizer (2025.1.0) again (the CUDA 13.0 wheel's, 2025.3.1: "Target application terminated before
  first instrumented API call").
- **check_f8 at 6228590a (round 3's checks), RTX 5090: all passed.** The 60 samples of the seven repos bit for bit; f8_linear's 89 matrices within
  the bound (worst 3.64e-3 of the sum of |x w|), 28 plain patches at unaligned words, row blocks of 17-64 tokens by the decode's path dlits 985,
  marks 1,510, var 11,920, spills 11,640, dvar 120 (each above 0, as `lin_branches` requires); W8A8 worst 2.51e-3. The CPU emulation of
  case_linear: all passed (60 matrices, worst 3.51e-3; 20 plain patches at unaligned words; paths marks 658, var 440, spills 160, dlits 120,
  dvar 120).
- **ncu (round 3): not measured.** `ncu` started on the pod but the host does not let a container read the GPU's performance counters
  ("ERR_NVGPUCTRPERM"), so there are no executed-instruction or issue-slot counts (`rtx5090-6228590a/ncu.csv`).
- **The cost of the 2^64 is an estimate, not a measurement:** it was never timed against 8df2d686 on one pod. Across the two RTX 5090 pods (other
  hosts and drivers, 580.126.16 and 595.91.07) f8_linear's own times changed by 0.972, 0.975, 1.034 and 1.066 at 1, 8, 32 and 64 tokens while
  vLLM's changed by 0.929 (geometric mean) and the route's by 0.945-0.952: about 3-5% at 1-8 tokens and 9-12% at 32-64, by which runner
  normalises the pods. Its static cost is one bf16 FMA a register of X (128 a step at 64 tokens, 3.5% of the step's code); scaling X once a step
  in shared memory instead of in every warp's fragments would cost less (not built).
- **L40S: not measured**, and the RTX 4090's ratios are not carried over to it (another memory bandwidth and SM count); `F8_LINEAR_TOKENS` names
  the RTX 4090 and RTX 5090 only.
- **The fp8-speed branch** (session fp8-speed-consumer took it over at 01:47: no commits, file changes or `fp8s` pods of that session after it)
  is merged in PR #44 as of ecdcfc1c.

## Round 3: the review's fixes (branch fp8-speed-review, PR #52; run on an RTX 5090 at 6228590a)

- `bench_f8.py`: vLLM's kernel is chosen under each checkpoint's own model types (`MODEL_TYPES`, from `research/models-2026/models.json`); with
  `--served` it first starts stock vLLM on each checkpoint (dummy weights: config and tokenizer only) and keeps its "Selected X for ..." lines, and a
  shape whose bench kernel is not one of them gets no vLLM or route time; each runner's copies hold at least 3 times the GPU's L2 (no cap of 64);
  block-scale shapes vLLM refuses at a tensor-parallel split are left out.
- `tables.py` leaves out of every sum and count, and marks: shapes whose copies held under 1.5 L2s (the older runs' cap: RTX 4090 18 shapes, 72 of
  472 shape-token rows at 851cf2e8; RTX 5090 4 shapes), block-scale shapes vLLM refuses, and the RTX 5090 Qwen rows timed against DeepGEMM. Every
  run's `tables.md` is regenerated with it; the numbers above are its.
- `check_f8.py`: the decode a step ahead together with the spills' pass (both plans' kernels were never run before); dense patches (`lin_dense`)
  at 33 and 64 tokens and `lin_crafted` patches at the dense pass's limits (768 values past level 1 a row block; 8 of a lane's 24 past level 2, and
  12); `lin_branches` counts the row blocks of 17-64 tokens by the decode's path (variable part over VB words, the pass's fallbacks for spills and
  marks, its literals, its variable part read from `var`) and requires each above 0; gemma's 2816 x 704 (a patch and a half's tail) with a row's
  scales at 1, 8, 33 and 64 tokens; `case_lin_quick` (the sanitizers' input) with dense patches and the combined plan.
- `glyd_gpu.cu`: comments only (the default's measured claim, `f8_geo`'s padding words, the non-volatile `mma.sync` and its warp-uniform guards).
- `sass/diff_3b69f0d3_8df2d686.txt` (`sassdiff.py`): the 266 non-FP8 device functions have the same SASS as main's on sm_89 and sm_120.
- `job3.sh`: one RTX 5090 pod for all of it (the emulation, check_f8, the sanitizers, bench_f8 --served on the four checkpoints, ncu). Refused:
  `rp_run.sh` exit 5, the guard's list of allowed pod prefixes did not hold `fp8s` (Runpod balance then about $32). Run later the same day as
  fp8s-5090r3 (fp8s-5090r2 before it got no ssh port from Runpod in 15 minutes: no job): every step ran, results in `rtx5090-6228590a/`.

## Not done

- Instruction counts executed (`ncu` smsp__inst_executed and issue slots): `job3.sh`'s step ran but the pod refused the counters
  (ERR_NVGPUCTRPERM); only static `cuobjdump` counts exist.
- A served run confirming vLLM's per-tensor kernel on the RTX 5090 (stock vLLM 0.30 did not start Mistral-Small-4-119B-2603: a transformers
  ImportError), and why that kernel's times do not change with the tokens and gave NaN once.
- The L40S.
- The 2^64's cost on one pod.
- Runners interleaved and SM clocks read per runner (they run vLLM, then ours, then the route; `clocks.csv` samples every second).
- KV-head replication of qkv at tensor-parallel sizes past the KV heads (bench_f8 splits N by tp).

## The runs

| directory | GPU, pod | code | what |
| :--- | :--- | :--- | :--- |
| `rtx5090-6228590a/` | RTX 5090, fp8s-5090r3 (round 3) | 6228590a (PR #52; e1970149's kernels) | check_f8 (GPU, the seven repos' samples): all passed; the CPU emulation of case_linear: all passed; the sanitizers: 0 errors, 0 hazards; bench_f8 --served (4 models, tp 1 and 2: stock's kernels checked, vLLM, f8_linear, the route, numerics); clocks; ncu refused the counters |
| (none) | RTX 5090, fp8s-5090r2 (round 3) | 6228590a | no ssh port from Runpod in 15 minutes: no job |
| (none) | RTX 5090, fp8s-5090r (round 3) | fd723af9 | refused by the Runpod guard: not run |
| `rtx5090-72872c13/` | RTX 5090, fp8s-5090f | 72872c13 (PR #44's library, e1970149's) | check_f8 (GPU, the seven repos' samples): all passed; the sanitizers; bench_f8 (4 models, tp 1 and 2: vLLM, f8_linear, the route, numerics); clocks |
| `rtx4090-72872c13/` | RTX 4090, fp8s-4090c | 72872c13 | check_f8: all passed; the sanitizers; the CPU emulation: all passed |
| `rtx4090-e1970149/` | RTX 4090, fp8s-4090b | e1970149 | check_f8 (stopped at the new graph-capture check: the test captured on another stream than its workspace's, fixed in 72872c13), the sanitizers, the CPU emulation: all passed |
| `rtx4090-851cf2e8/` | RTX 4090, fp8s-4090a | 851cf2e8 (e1970149's kernels; development W8A8 still taken on Ada) | check_f8 (stopped at the W8A8 case above), bench_f8 (7 models, tp 1-8), clocks |
| `rtx5090-8df2d686/` | RTX 5090, fp8s-5090c | 8df2d686 (before the 2^64) | check_f8 (the small-activation case failed: defect 2), bench_f8 as above, the emulation |
| `rtx5090/` | RTX 5090, fp8s-5090b (round 1) | 3e60e481 (bench_A, check_A); its W8A8 kernels at two CTAs an SM (bench_B, check_B) | the development variants: W8A8, at least 2 or 4 patches a CTA, a step ahead, the spills' pass |
| `rtx5090-before/` | RTX 5090, fp8s-5090a (round 1) | main's f8_linear (f4a34f71) | bench only; another pod than `rtx5090/`'s; check_f8 failed on it (the shared memory limit on sm_120, fixed by fp8-speed's 3bac5386), so its outputs were never validated |
| `rtx4090/` | RTX 4090, fp8s-a5 (the fp8-speed session) | a6c4240f | check_f8, bench_f8 without the route; superseded by `rtx4090-851cf2e8/` |

Each holds `bench.json` (or `bench_*.json`), `tables.md` (`python3 tables.py BENCH.json`), the logs (bench logs without FlashInfer's build output)
and `machine.txt`. `job3.sh` is round 3's pod job, `job2.sh` round 2's, `job.sh` round 1's.

## RTX 4090 (851cf2e8): per-shape sums at tensor-parallel 1, time over vLLM's

| model | vLLM's kernel | ours M=1 | M=8 | M=32 | M=64 | route M=1 | M=8 | M=32 | M=64 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-5.3-Flash | Marlin W8A16 | 1.01 | 1.05 | 1.24 | 1.72 | 2.02 | 2.04 | 2.02 | 2.06 |
| Qwen3.8-27B-FP8 | Marlin W8A16 | 0.94 | 0.96 | 1.11 | 1.68 | 2.33 | 2.32 | 2.23 | 2.17 |
| MiniMax-M3-MXFP8 | Marlin MXFP8 W8A16 | 0.95 | 0.99 | 1.47 | 1.80 | 2.06 | 2.07 | 2.05 | 2.12 |
| Mistral-Small-4-119B-2603 | CUTLASS W8A8 | 1.16 | 1.33 | 1.65 | 1.74 | 1.48 | 1.48 | 1.44 | 1.50 |
| DeepSeek-V4.1-Flash | Marlin MXFP8 W8A16 | 1.04 | 1.12 | 1.65 | 1.96 | 1.49 | 1.51 | 1.65 | 1.78 |
| Qwen3.6-35B-A3B-FP8 | Marlin W8A16 | 1.13 | 1.22 | 1.51 | 1.76 | 1.30 | 1.32 | 1.37 | 1.48 |
| gemma-4-26B-A4B-it-FP8-dynamic | CUTLASS W8A8 | 1.08 | 1.16 | 1.48 | 1.91 | 1.38 | 1.37 | 1.43 | 1.54 |

DeepGEMM does not run on Ada, and served runs there select the same kernels for block and row scales ("Selected MarlinFP8ScaledMMLinearKernel for
Fp8LinearMethod" on an RTX 4090 in `benchmarks/gpu/fp8-vllm-2026-10-05/f8a1/`; that and "Selected CutlassFP8ScaledMMLinearKernel for
CompressedTensorsW8A8Fp8" on an L40S in `research/report-quantized-2026-10-06/records/l40s-qf8-run5/`); the MXFP8, DeepSeek and per-tensor rows
have no served run's line to check against.
Round 1's run of the fp8-speed kernel (a6c4240f, `rtx4090/`, another pod) is in its `tables.md`; the difference mixes X's 2^64 with the other
changes since a6c4240f and the pods.

## Round 1's development variants on the RTX 5090 (3e60e481, `rtx5090/`)

W8A8 with at least 2 patches a CTA was 0-17% faster than bf16 X in per-shape sums at 32-64 tokens (a ratio of our two kernels, whatever vLLM's
kernel), but slower on 12 of the 70 shapes (up to 1.58x on Qwen3.6-35B-A3B's 2048 x 512 expert_down, where 64 patches at 2 a CTA leave 138 of the
170 SMs idle), and its error is about 15 times bf16 X's (2.4-2.7e-2 rms); against vLLM, on gemma (the one model whose vLLM kernel there is
stock's), 1.95 and 2.19 times its time at 32 and 64 tokens. Round 1 made it the default at 17-64 tokens on compute capability 12.0; round 2 took
that back. Main's f8_linear before the fp8-speed branch: `rtx5090-before/` (another pod, unvalidated).

## Why it is slower

`sass/sm_89.txt`, `sass/sm_120.txt` (`sass_counts.py` on `cuobjdump -sass` at 851cf2e8, nvcc 13.0.2): each kernel's instructions, its step loop
and its MMAs, and ptxas' registers. These are static counts: every path of the step loop once (the plain patches', the literals' and the
unstaged variable parts', which rarely run), so an upper bound on a step's work, not instructions executed; no profiler counters were taken
(`job3.sh`'s ncu step ran at 6228590a, but the pod refused the counters: ERR_NVGPUCTRPERM). A step's loop is 2.6-3.9 k instructions of code (the default plan's kernels, both architectures) against 8
(1-8 tokens) to 64 (33-64 tokens) bf16 MMAs, and at 1-8 tokens f8_linear reads 16% fewer bytes than vLLM's kernel yet takes longer, which points
at the decode; measuring it (`ncu`'s executed instructions and issue slots) is the next step of any kernel work.

## Method, and its limits

- "Per-shape sums" add each distinct shape of a model once: not a decode step's time (no layer counts), and the expert shapes are timed against
  vLLM's dense FP8 Linear kernel, while vLLM runs routed experts through its fused MoE kernels. `tables.md`'s "dense" rows leave them out.
- vLLM's side is its kernel object built and processed as its Linear methods build them, applied eagerly inside a CUDA graph, with a stand-in
  model config (from round 3: the checkpoint's own model types, and with `--served` its kernel checked against stock vLLM's selection on the
  checkpoint). In a served model vLLM's compile passes can fuse X's quantization into the norm before it; that would favour vLLM.
- Copies: rounds 1-2 held each runner's weights on at least 200 MB of copies but at most 64 copies, so small shapes' copies held 8-67 MB on the
  RTX 4090 (L2 72 MiB) and some of their weights may have come from L2: `tables.py` leaves shapes under 1.5 L2s out. Round 3's bench holds 3 L2s.
- Clocks: round 2 logged `nvidia-smi` clocks, power and temperature every second (`clocks.csv`): 2,700-2,730 MHz on the 4090 under load, no
  throttle reason but idle; the runners are timed one after the other, not interleaved. Round 1 did not log them. Round 3 (RTX 5090, 6228590a,
  maximum 3,090 MHz): 2,400 MHz in most busy samples at 14:30-14:32 UTC and 2,932 MHz at 14:34-14:35, which runner each sample caught not
  recorded; no throttle reason in 557 of 560 samples.
- Tensor-parallel sizes: `bench_f8.py` splits only the models it marks (Qwen3.6-35B-A3B-FP8, Mistral-Small-4-119B-2603, GLM-5.3-Flash,
  MiniMax-M3-MXFP8, DeepSeek-V4.1-Flash), N or K by tp (not replicating KV heads); Qwen3.8-27B-FP8 and gemma-4-26B-A4B-it-FP8-dynamic run at
  tensor-parallel 1 only.

## Reproduce

```
GLYD_GPU_LIB=lib/libglyd_gpu_cuda13.so PYTHONPATH=glyd-gpu-py python gpu/fp8/check_f8.py --samples SAMPLES
GLYD_GPU_LIB=... PYTHONPATH=glyd-gpu-py python gpu/fp8/bench_f8.py --samples SAMPLES --models Qwen3.6-35B-A3B-FP8,Qwen3.8-27B-FP8,gemma-4-26B-A4B-it-FP8-dynamic,Mistral-Small-4-119B-2603 \
    --tp 1,2 --served --numerics --no-unpack --out bench.json
python3 tables.py bench.json > tables.md
```

Pods: `job2.sh` and `job3.sh` (vLLM 0.30.0, torch 2.13.0+cu130, CUDA 13.0 wheels for nvcc; drivers 580.126-595.91, in each `machine.txt`). Runpod: round 1
fp8s-5090a $0.30, fp8s-5090b $0.46; round 2 fp8s-5090c $0.25, fp8s-4090a $0.11, fp8s-5090d $0.03 (no ssh in 15 minutes: no job), fp8s-4090b
$0.15 (a re-run after the test bug of `rtx4090-e1970149/`), fp8s-5090f $0.24, fp8s-4090c $0.15: $0.94; round 3 fp8s-5090r none (refused),
fp8s-5090r2 $0.03 by the budget guard's count (no ssh in 15 minutes: no job), fp8s-5090r3 $0.37 (22.5 minutes at $0.99 an hour): $0.40. $2.09 in
all, under the session's $8 cap, but eight pods where the brief allowed two batched ones (round 2 alone used six, one never ran its job and one re-ran after a
test bug): trial and error on pods, against the owner's rule; any next run is one batched pod (`job3.sh`).
