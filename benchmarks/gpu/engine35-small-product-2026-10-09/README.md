# glyd-engine: Qwen3.5's small product over many blocks (2026-10-09)

Private record. A decode step's profile of Qwen/Qwen3.8-27B (`1d4bf0f2`) on `glyd-engine`'s Qwen3.5 forward pass, on one A100 80GB
PCIe (Runpod pod `zmw74hlhfowkw7`, driver 595.91.07, the pod of benchmarks/weights/kld-q56-2026-10-09), the engine built there (dev
features, CUDA 12.8 nvcc) with glyd-gpu 0.29.2's CUDA 13 library: `glyd-engine bench DIR --layout L --prompt 1024 --new 128 --profile`.

Before (private main `9597e738`), `small_linear` ran as one block of 8 warps for a step's one row: the linear attention's ba product
(in_proj_b and in_proj_a, 96 rows of 5,120) took 11.778 ms of the step's 56.679 ms (eager profile), 48 launches of
about 245 us. After (`48b4ff3e`): a warp a row of W, the rows spread over `ceil(rows / 8)` blocks, 16-byte loads where K and the
rows allow them.

| | Compact (6.8 bits), before | Compact, after | Tiered (lossless), before |
| :-- | --: | --: | --: |
| decode, ms a token (CUDA graph, mean of 128) | 54.171 | 43.131 | 43.784 |
| tokens a second | 18.46 | 23.19 | 22.84 |
| prompt of 1,024, ms | 737.9 | 723.6 | 628.8 |
| weights, GiB | 21.160 | 21.160 | 33.698 |

One step's GPU time by kind (eager, 20 steps' mean, ms):

| Kind of work | Compact, before | Compact, after | Tiered (lossless), before |
| :-- | --: | --: | --: |
| MLP: gate and up product, SiLU | 17.779 | 17.765 | 12.731 |
| linear attention: ba product | 11.778 | 0.621 | 11.819 |
| MLP: down product | 9.190 | 9.184 | 6.789 |
| linear attention: qkvz product | 6.586 | 6.580 | 4.773 |
| linear attention: out product | 3.032 | 3.024 | 2.352 |
| attention: qkv product | 1.972 | 1.970 | 1.517 |
| LM head product | 1.932 | 1.927 | 1.770 |
| norms | 1.305 | 1.292 | 1.311 |
| attention: o product | 1.006 | 1.003 | 0.758 |
| attention: attention | 0.760 | 0.757 | 0.764 |
| linear attention: delta rule | 0.559 | 0.556 | 0.559 |
| linear attention: convolution | 0.313 | 0.307 | 0.314 |
| linear attention: gated norm | 0.299 | 0.290 | 0.291 |
| attention: q/k norm, rotary, cache write | 0.107 | 0.105 | 0.108 |
| argmax | 0.041 | 0.040 | 0.040 |
| embedding row | 0.021 | 0.022 | 0.024 |
| total | 56.679 | 45.444 | 45.919 |

- The fix takes the decode step from 54.2 to 43.1 ms (26% more tokens a second). The Tiered run is before the
  fix: its ba product (11.819 ms) is the same kernel's.
- On this GPU the Tiered layout decodes faster than the compact one (43.8 against 54.2 ms before the fix) though it reads
  1.59 times the bytes: the compact layout's products run at about a quarter of the A100's bandwidth (the MLP's gate and up
  product, 17.78 ms for 64 x 34,816 x 5,120 weights at 6.774 bits a weight, about 9.7 GB, is about 0.55 TB/s), its decode bound by arithmetic
  where the GPU has bandwidth to spare. On an Ada card it is the other way: below.
- `kcheck35.out`: every Qwen3.5 kernel against the CPU reference within its bound after the fix, the small product (b and a of the 0.8B)
  352 of 352 values equal. `gen2.out`: 24 greedy tokens after the fix, "red, green, and blue. When these colours are combined in equal
  intensities, they produce white light."

## The same on an RTX 6000 Ada (Ada, 960 GB/s), against vLLM FP8

`ada/`: one RTX 6000 Ada Generation (Runpod pod `innygdlxrinqi5`, driver 595.91.07, $0.99/hour, 01:04–01:18 UTC: about $0.23; no L40
was free), the engine built on the pod from `9597e738` (before) and `48b4ff3e` (after) with glyd 0.29.2's CUDA 13 library (install.sh),
`bench ... --prompt 1024 --new 256`; vLLM 0.30.0 with Qwen/Qwen3.8-27B-FP8 (`017b9c7a`) by `glyd-engine/tools/ref_speed.py` on the same
pod (`language_model_only`, `max_num_seqs` 16, `VLLM_USE_FLASHINFER_SAMPLER=0`: glyd run's environment), medians of 5.

| | decode, ms a token | tokens a second | prompt of 1,024, ms | weights, GiB |
| :-- | --: | --: | --: | --: |
| vLLM 0.30.0, the FP8 checkpoint | 33.452 | 29.89 | 484.6 | 28.025 |
| glyd-engine compact, before | 33.476 | 29.87 | 497.5 | 21.160 |
| glyd-engine compact, after | 33.144 | 30.17 | 490.7 | 21.160 |
| glyd-engine Tiered (lossless), after | 42.985 | 23.26 | 524.3 | 33.698 |

On this GPU the fix changes the decode step by 1% (33.48 to 33.14 ms): the one-block kernel took 0.43 ms of a step after the fix and
little more before (`profile-after-a.log`: 32.72 ms in all, the products 29.8 ms at 0.57 to 0.79 TB/s of the card's 0.96). The compact
layout decodes 1.30 times as fast as the Tiered one here, where on the A100 it was the slower. Against vLLM's FP8 checkpoint on the same
pod, compact after the fix decodes 0.9% faster with 24.5% less weight memory, and its first token takes 1.3% longer.

Files: `prof-a.*` (compact, before), `prof-mma.*` (Tiered, before), `prof2-a.*` (compact, after), `kcheck35.out`, `gen2.out` (A100);
`ada/` (the RTX 6000 Ada runs, their steps and vLLM's JSON).
