# The route SPLIT on an H100 SXM, Qwen3-14B (2026-10-01)

o3_job.sh (the settle on one Hopper GPU, `../jobs/`) on one NVIDIA H100 80GB HBM3, with the route forced on: the tree's
route for an H100 SXM was v0.25.1's, so this is a measurement of what the route would give there, against v0.25.1's
routes in the same process. The route was extended to the H100 SXM after it (`GLYD_GPU_H100`, below).

## Setup

- **Machine:** NVIDIA H100 80GB HBM3, the SXM5 (compute 9.0, sm_90a, 132 SMs, 81,559 MiB of HBM3, power limit 700 W,
  1,980 MHz at most), driver 580.126.20. Xeon Platinum 8480+ (26 CPUs), 221 GB of memory, x86_64. Lambda
  `gpu_1x_h100_sxm5`, one of the session's seven jobs (68 minutes, `../../vllm-m6-h100-2026-10-01/session/`).
- **Software:** torch 2.14.1+cu130, CUDA 13.0 (nvcc 13.0), transformers 5.18.0, in `~/gpuenv`.
- **Tree:** 7fe66a2 (release-0.26.0, the v0.26.0 candidate, C API 7). The library for sm_90a and the JIT build were built
  in the job, which had both by 96 seconds in.
- **Route:** SPLIT forced on by `GLYD_SPLIT_MIN=2048` (`forced.txt`). The knob sets the route's lower bound for every
  matrix; Qwen3-14B's merged Linears are qkv [7,168, 5,120], o [5,120, 5,120], gate_up [34,816, 5,120] and down
  [5,120, 17,408], all with O and K at least 5,120, which the rule for Hopper (a GH200's: 2048-8192 tokens, O and K at
  least 5120) takes whole. So for this model the forced route is the rule's own. The decode took 12 of the 132 SMs
  (120 for cuBLAS) at 2048 and 4096 tokens and 4 at 8192 (the rule gives 4 from 6144; `layer-Qwen3-14B.txt`).
- **Run:** the job's steps, 289 seconds in all, every one exit 0 (`steps.txt`, `windows.txt`):
  1. `check_capi.py`: 7,614 calls through both hosts (the JIT build and the library) bit for bit identical, 260,052
     routes as v0.25.1's rule (check_capi's "main's rule"), the SPLIT rule's 57,876 routes pinned (this tree's rule gave
     an H100 SXM, code 90, never SPLIT), the SPLIT decode bit for bit on 1-200 SMs, GLinear by the route as on an A100:
     products within 1e-2 of fp32 and the same bits run to run (`check_capi.txt`).
  2. `test_gpu.py`'s split tests: `test_c_header`, `test_split_order`, `test_split_route` ok (`test_split.txt`).
  3. `split_stress.py`: every Qwen3 layer's matrices (0.6B-32B) through the ring at 769-4096 tokens: 16,512 products, 0
     failures, on a split of 12 + 120 SMs (`split_stress.txt`).
  4. `e2e.py MODEL --format mma12 --fused --merge --tokens 32 --baseline --prefill 2048,4096,8192 --without-split
     --rounds 3` on Qwen/Qwen3-14B: bf16, then SPLIT and v0.25.1's routes, 3 rounds each way in turn
     (`e2e-Qwen3-14B.txt`).
  5. `layer.py`: layer 10's products (`layer-Qwen3-14B.txt`).
  6. `split_stress.py`, the split skewed both ways (`--sms=-2,1`) at 769, 2048 and 8192 tokens: 7,920 products, 0
     failures, on 124 SMs for the decode and 8 for the products, then 4 for the decode and 128 for the products
     (`split_stress_skewed.txt`; at 8192 tokens GLinear's own ring was made for 14B alone, the stress's GLinears being
     made as on an A100, whose rule ends at 4096 for a matrix not both at least 5120).
- **The job as it ran** is `o3_job.sh` here; `../jobs/o3_job.sh` is the same script with an H100 SXM taking the shipping
  route, as it now does. Its summary: `summary.txt` (CHECKS PASS: 6 steps run, each exit 0). The hostname in `machine.txt` (the
  instance's address) is written `host`.

## Forward pass, SPLIT against v0.25.1's routes (`summary.txt`, `e2e-Qwen3-14B.txt`)

One forward pass over a prompt of that many tokens, the Linears merged, the median of 3 rounds each way in turn, ms
(the first token's time in brackets):

| Tokens | bf16 | v0.25.1's routes | SPLIT | SPLIT / v0.25.1's, the median of the rounds' ratios (each round's) | SPLIT / bf16 |
| ---: | ---: | ---: | ---: | :--- | ---: |
| 2048 | 119.2 (126.4) | 150.5 (151.9) | 134.4 (136.8) | **0.893** (0.883, 0.895, 0.893; first token 0.901) | 1.128 |
| 4096 | 245.7 (244.7) | 279.0 (281.0) | 260.4 (262.4) | **0.937** (0.937, 0.946, 0.924; first token 0.933) | 1.060 |
| 8192 | 504.5 (505.7) | 551.4 (553.3) | 518.2 (521.3) | **0.938** (0.938, 0.940, 0.935; first token 0.940) | 1.027 |

SPLIT took 10.7%, 6.3% and 6.2% less time than v0.25.1's routes, every length past the 2% a route has to give
(`summary.txt`'s DECIDES lines: "SPLIT stays"). v0.25.1's routes took 1.263x, 1.136x and 1.093x bf16's time; SPLIT 1.128x,
1.060x and 1.027x. The ratios in the table are each round's SPLIT time over v0.25.1's, the median of the 3, not the ratio
of the medians in the columns beside them.

**Layer 10** (qkv, o, gate_up, down in a prompt's order, 8 layers' Linears over the same packs a pass, the median of 5
passes), against today's route (v0.25.1's) and bf16:

| Tokens | bf16 | v0.25.1's route | SPLIT | SMs for the decode |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 1.912 ms | 2.752 ms (1.439x bf16) | 2.300 ms (1.203x bf16, 0.836x v0.25.1's) | 12 |
| 4096 | 3.870 ms | 4.895 ms (1.265x) | 4.504 ms (1.164x, 0.920x) | 12 |
| 8192 | 7.655 ms | 9.110 ms (1.190x) | 8.400 ms (1.097x, 0.922x) | 4 |

Every Linear of the layer took the route (SPLIT in all four at every length), the ring ran, and the products were within
2.5e-3 to 2.9e-3 of fp32, the same bits pass to pass.

**The GPU was at its power cap** (`summary.txt`, nvidia-smi's samples every 250 ms): in 98% of the samples of SPLIT's e2e phases and
100% of v0.25.1's (SW power cap, 677 and 690 W on average of the 700 W limit), against 82% of bf16's; no thermal
throttling in any sample (60 C at most). The SM clock averaged 1,696 MHz in SPLIT's phases, 1,729 in v0.25.1's and 1,744
in bf16's.

## What this does and does not show

- **Measured:** Qwen3-14B, whose matrices all have O and K at least 5120, at 2048, 4096 and 8192 tokens, on one H100 SXM5
  with 700 W, forced on. Qwen3-32B's matrices (qkv [10,240, 5,120], o [5,120, 8,192], gate_up [51,200, 5,120], down
  [5,120, 25,600]) are such matrices too, so the rule reaches them by their shape; they were not run on this GPU. The
  GH200's settle had them at 0.909-0.952x. Qwen3-8B's matrices (4,096 on a side) are under the rule's
  threshold and were not run here.
- **Not measured:** lengths under 2048 or past 8192 tokens, an H100 SXM with another power limit, an H200, an H100 NVL,
  and the shipped gate itself (its class by name and SM count) on this GPU: the tree had no H100 class, so the route was
  taken by the knob. The gate was checked on the host and on an L4 (`../../release-0.26.0-l4/h100-sxm-route`).

## Files

`check_capi.txt`, `test_split.txt`, `split_stress.txt`, `split_stress_skewed.txt`, `e2e-Qwen3-14B.txt`,
`layer-Qwen3-14B.txt`, `summary.txt` (the page of results, DECIDES lines), `steps.txt`, `windows.txt` (each step's
time window), `smi.csv` and `smi-fields.txt` (nvidia-smi every 250 ms: clocks, power, temperature, throttle reasons),
`machine.txt` and `machine-short.txt`, `env.txt`, `route.txt` and `forced.txt`, `expect.txt`, `split-kernel-ptxas.txt`
(the split decode kernel's registers: 146, no spills), `job.log`, `log/` (the library's build, the model's download,
the environment), `o3_job.sh`.
