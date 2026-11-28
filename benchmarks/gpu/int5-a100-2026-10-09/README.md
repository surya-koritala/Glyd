# Five-bit block codes (Int5) on an A100, Qwen3.8-27B's speed and KL; and v0.29.3 installed fresh (2026-10-09)

Private record. One A100-SXM4-80GB (Runpod pod `qkn4obbvrknspw`, driver 580.126.16, $1.79/hour, 12:26-13:16 UTC: about $1.49), branch
`int5-linnet` (the five-bit layout: gpu/glyd_gpu.cu's `Int5`, glyd-gpu's `pack_int5`/`unpack_int5`, `glyd-engine --layout i5`), the
library by `gpu/build_lib.sh` (CUDA 12.8, every architecture) and the engine's release build. Jobs: `job.sh` (build, examples/int5),
`bench.sh` (after the fresh install, on its download of Qwen/Qwen3.8-27B at 1d4bf0f2), `kl.sh` (the KL study). `steps.txt`'s first
bench lines (rc=1, 0 s) are a first attempt that found no checkpoint (the fresh install's first try had failed: PyPI's index had not
yet listed glyd 0.29.3); `bench.sh` ran them again.

## Exact against the packer (`int5-check.log`, examples/int5)

The GPU's decode (`unpack`) of four matrices of a fixed seed, [256, 512] to [6144, 5120], is `unpack_int5`'s bit for bit. Products
(`Library::linear`, 1 to 512 tokens) against the same product of the decoded matrix in the tiered layout (lossless: the same weights):
bit for bit at 33 to 512 tokens; to 16 tokens, where the tiered layout's small products add in another order, every output of both
layouts within one rounding to bf16 and an f32 sum's error of the product in f64. "all checks passed".

## Qwen3.8-27B, one user (`bench --prompt 1024 --new 256`, CUDA graphs)

| On this pod | decode, ms a token | tokens a second | prompt of 1,024, ms | weights, GiB | in use, GiB | loaded in, s |
| :-- | --: | --: | --: | --: | --: | --: |
| Int5 (`--layout i5`) | 20.928 | 47.78 | 420.4 | 17.351 | 19.503 | 51.3 |
| Int6 (`--layout i6`, v0.29.3's `--lossy`) | 21.450 | 46.62 | 421.6 | 20.479 | 22.550 | 49.1 |

Int5 has 15% fewer bytes of weights and decodes 2.5% faster here: on the A100 the products are not yet at the memory's speed (Int6's at
56% of an A100 PCIe's bandwidth, benchmarks/gpu/int6-l40-2026-10-09). In use: 19.5 GiB at 4,096 tokens of context. `generate-i5.log`:
"red, green, and blue. When these colours are combined in equal intensities, they produce white light. This principle is known as additive
colour mixing."

## KL divergence from BF16 (`kl-results.jsonl`, research/lossy-2026/kld_study.py; the same 253,952 positions of WikiText-2)

| Weights | KL, mean | KL, p99 | top-1 as BF16 | perplexity |
| :-- | --: | --: | --: | --: |
| five-bit runs of 32, bf16 scale (`int:5:32:absmax:bf16`: `pack_int5`'s rule) | 0.01408 | 0.1121 | 0.9441 | 7.3058 |
| five-bit, the best of 16 bf16 scales (`int:5:32:mse:bf16`) | 0.01393 | 0.1127 | 0.9433 | 7.2978 |
| six-bit, the best of 16 bf16 scales (`int:6:32:mse:bf16`) | 0.00477 | 0.0335 | 0.9684 | 7.2485 |

For comparison (benchmarks/gpu/int6-l40-2026-10-09, an A100 PCIe pod, the same method): BF16 7.2226; FP8 (Qwen/Qwen3.8-27B-FP8) 0.00465;
six-bit with `pack_int6`'s bf16 scale (v0.29.3) 0.00513. The best of 16 scales (0.70 to 1.0 of the run's largest magnitude over 31)
takes the six-bit layout to 1.03 times FP8's KL, with no change to the layout.

## v0.29.3 installed fresh (`fresh-0.29.3/`)

`fresh3.sh` on this pod: no uv, no glyd, empty caches; the v0.29.3 tag's install.sh (the one getglyd.com serves). Installed in 79 s
(glyd 0.29.3, glyd-gpu 0.29.3); `glyd run Qwen/Qwen3.8-27B --lossy -p "In one sentence, why is the sky blue?"` downloaded 55.6 GB, packed
and answered in 335 s ("Ready in 5m23s"; 20.5 GiB of weights, 32,768 tokens of context): "The sky is blue because sunlight scatters off
molecules in the atmosphere, and shorter blue wavelengths scatter more than longer red ones." A second run (`--no-think`, "Name three
primes above 100.") in 137 s: 101, 103 and 107.
