# Six-bit block codes (Int6) on an L40 and an A100: exact against the packer, and Qwen3.8-27B's speed and quality (2026-10-09)

Private record. The six-bit block layout of this branch (gpu/glyd_gpu.cu's `Int6`, glyd-gpu's `pack_int6`/`unpack_int6`,
`glyd-engine --layout i6`) built and run on one L40 (Runpod pod `src9jyzrhb274h`, driver 595.91.07, $0.82/hour, 01:41–02:01 UTC
on 2026-10-09: about $0.27): the library by `gpu/build_lib.sh` (CUDA 12.8, every architecture), the engine with its dev features.
Job: `job.sh`, then the small product's fix (`8ddee393`, PR #116 cherry-picked) and the same runs again.

## Exact against the packer (`int6-check3.log`, examples/int6)

- The GPU's decode (`unpack`) of four matrices of a fixed seed, [256, 512] to [6144, 5120], is `unpack_int6`'s bit for bit: all
  131,072 to 31,457,280 weights of each.
- Products (`Library::linear`, 1 to 512 tokens) against the same product of the decoded matrix packed in the tiered layout
  (lossless: the same weights): bit for bit at 33, 64, 65, 200 and 512 tokens, every output. To 16 tokens the tiered layout takes a
  small-product kernel of its own that adds in another order (1 to 9 outputs of thousands differ, by 1 or 2 bf16 places where an output
  is small beside its terms); there every output of both layouts is within one rounding to bf16 (2^-8 of the value) and an f32 sum's
  error (k 2^-24 of its terms' magnitudes) of the product in f64. "all checks passed". The first run's stricter criterion (1 place)
  failed on cancelling outputs of the tiered product too (`int6-check.log`); the criterion was changed, not the kernel.
- The crate's tests (`crate-tests.log`): `int6_round_trips_as_its_runs_say` and `int6_value_is_the_decodes_arithmetic` pass.

## Qwen3.8-27B, one user (`bench --prompt 1024 --new 256`, CUDA graphs)

| On this pod | decode, ms a token | tokens a second | prompt of 1,024, ms | weights, GiB | in use, GiB | loaded in, s |
| :-- | --: | --: | --: | --: | --: | --: |
| Int6 (`--layout i6`) | 35.560 | 28.12 | 491.9 | 20.479 | 22.541 | 51.3 |
| Candidate A, compact (`--layout a`) | 38.924 | 25.69 | 532.2 | 21.160 | 23.240 | 81.0 |
| Int6, with the small product's fix | 32.000 | 31.25 | 481.0 | 20.479 | 22.541 | 52.3 |
| compact, with the fix | 36.454 | 27.43 | 529.2 | 21.160 | 23.240 | 78.4 |

vLLM 0.30.0 with Qwen/Qwen3.8-27B-FP8 on another L40 pod (2026-10-08, the same script and settings: the 0.29.2 measurement) decoded at
39.454 ms (25.35 tokens a second), its first token 503.9 ms, 28.025 GiB of weights: Int6 with the fix decodes 1.23 times as fast with
27% less weight memory (another pod: not one run's comparison). `profile-i6.log` (before the fix): the products at 0.67-0.68 TB/s of the
L40's 0.864 (the MLP's), the ba product 3.95 ms a step (one block, the fix's subject). `generate-i6*.log`: 32 greedy tokens, "red, green,
and blue. When these colours are combined in equal intensities, they produce white light. This principle is fundamental to additive
colour mixing".

## On an A100 80GB PCIe (`job-a100.sh`)

Runpod pod `f708z2r0xze2e9`, $1.79/hour, 02:08-04:08 UTC on 2026-10-09 (about $3.58): the same branch with the small product's fix,
the library for every architecture. The pod's watchdog deleted it at 04:08 before its logs were copied: the figures here are as read
from the pod during the run (the session's record), not files of this directory.

- examples/int6 on Ampere: "all checks passed" (the GPU's decode bit for bit, the products as on the L40).
- `glyd-engine run --lossy` (Int6): loaded in 68 s, 20.5 GiB of weights, 32,768 tokens of context; asked "In two sentences: what is the
  capital of Australia, and why was it chosen?" it answered "The capital of Australia is Canberra. It was chosen as a purpose-built
  compromise between rival claimants Sydney and Melbourne, so the federal government could control its site and development."

| On this pod, Qwen3.8-27B, one user | decode, ms a token | tokens a second | prompt of 1,024, ms | weights, GiB | loaded in, s |
| :-- | --: | --: | --: | --: | --: |
| Int6 (`--layout i6`) | 22.209 | 45.03 | 438.1 | 20.479 | 67.5 |
| Candidate A, compact (`--layout a`) | 43.088 | 23.21 | 720.9 | 21.160 | 98.0 |
| vLLM 0.30.0, Qwen/Qwen3.8-27B-FP8 (glyd-engine/tools/ref_speed.py) | 21.073 | 47.45 | 380.4 | 28.025 | |

On the A100 Int6 decodes at 0.95 of FP8's speed with 27% less weight memory (on the L40, 1.23 times FP8's).

## Quality (the KL study's method, benchmarks/weights/kld-q56-2026-10-09, on the A100 pod)

KL divergence of the next-token distribution from BF16's over the same 253,952 positions of WikiText-2:

| Weights | KL, mean | KL, p99 | top-1 as BF16 | perplexity | bits a weight |
| :-- | --: | --: | --: | --: | --: |
| BF16 | 0 | 0 | 1 | 7.2226 | 16 |
| FP8 (Qwen/Qwen3.8-27B-FP8) | 0.00465 | 0.0362 | 0.9707 | 7.2410 | |
| six-bit runs of 32, fp16 scale (`int:6:32:absmax`) | 0.00486 | 0.0344 | 0.9680 | 7.2607 | 6.198 |
| **this layout: bf16 scale** (`int:6:32:absmax:bf16`, `pack_int6`'s rule) | 0.00513 | 0.0351 | 0.9671 | 7.2665 | 6.198 |
| frac:3 (0.29.2's `--lossy`) | 0.00831 | 0.0665 | 0.9565 | 7.2205 | 6.498 |

The bf16 scale costs 6% over the fp16 one; the layout's KL is 1.10 times FP8's (0.29.2's was 1.79 times). A best-fit bf16 scale
(`int:6:32:mse:bf16`) was scoring when the pod was deleted: not measured. (The study's "bits a weight" counts a scale a run; the
layout's 6.53 counts its block bases too.)
