# Packed formats: bits a weight against the cost of unpacking, 2026-09-28

Candidate layouts, measured on the CPU (sizes) and in disassembly (instructions), and the one picked prototyped in the
library's own kernels (`gpu/experimental/format`) and timed on the RTX 4080 SUPER. The picked layout is the 12-bit
layout the release ships. `gpu/experimental/format` and the scripts these logs name are on the branch gpu-format
(d3fd2e6), not in this tree: the release's version is its own code, in gpu/glyd_gpu.cu. Models: the box's cached
Qwen3-0.6B, 1.7B, 4B-Instruct-2507 and 8B and granite-3.1-3b-a800m-instruct (their Linear weights as `gpu/sizes.py`
packs them; lm_head apart).

- `study.py`, `study.txt`: bits a weight of each candidate, per tensor and per model.
- `cols.py`, `cols.txt`: a variant of the candidates on Qwen3-8B.
- `count.txt` (`gpu/experimental/format/count.sh`): each candidate's unpacking alone, in disassembly for sm_80, sm_89 and
  sm_90a.
- `count_lib.txt` (`count_lib.sh`): the library's kernels, the 12-bit layout against the picked one.
- `check.txt`, `selfcheck.txt`: the prototype on every tensor of the five models, bit for bit, and one layer's
  products against fp32 and against the 12-bit layout's.
- `layer-*-run*.txt`: one layer's products (q,k,v and gate,up merged) at 1, 8, 32, 256 and 1024 tokens, cuBLAS against
  both layouts, two runs a model.
- `h100-sxm5/`, `a100-sxm4-40gb/`: the same on an H100 SXM5 and an A100 SXM4 40 GB (Lambda, 2026-09-28), run by the
  unattended job in `cloud/` (`format_job.sh`, its launchers and its summary; the sources `git archive` of 47c95d5's
  gpu/glyd_gpu.cu and .h, gpu/experimental/format and bindings/python/glyd): the self-check, layer 10 of Qwen3-8B, 14B
  and 32B at 1-512 tokens (H100) and 1-768 (A100), two runs each, and check.py on the whole Qwen3-8B.

## Bits a weight

| Layout | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B-Instruct-2507 | granite-3.1-3b-a800m | Qwen3-8B |
| :-- | :-- | :-- | :-- | :-- | :-- |
| tiered (`mma`) | 10.82 | 10.77 | 10.85 | 10.80 | 10.87 |
| 12-bit (`mma12`), before and after | 12.04 | 12.04 | 12.04 | 12.07 | 12.04 |

## One layer, RTX 4080 SUPER (`layer-*`; time against cuBLAS, run 1 / run 2)

| Model | M | cuBLAS us | 12-bit | 12-bit, revised |
| :-- | --: | --: | :-- | :-- |
| Qwen3-8B | 1 | 671 | 0.888 / 0.887 | 0.887 / 0.885 |
| | 8 | 751 | 0.797 / 0.798 | 0.797 / 0.797 |
| | 32 | 863 / 873 | 0.709 / 0.702 | 0.708 / 0.701 |
| | 256 | 1173 / 1165 | 0.918 / 0.917 | 0.913 / 0.912 |
| | 1024 | 4033 / 4031 | 0.983 / 0.985 | 0.979 / 0.979 |
| Qwen3-4B-Instruct-2507 | 1 | 394 / 395 | 0.840 / 0.837 | 0.838 / 0.837 |
| | 8 | 435 / 436 | 0.762 / 0.765 | 0.767 / 0.766 |
| | 32 | 462 / 461 | 0.745 / 0.747 | 0.747 / 0.744 |
| | 256 | 618 | 0.973 / 0.975 | 0.967 / 0.967 |
| | 1024 | 2054 / 2085 | 1.040 / 1.040 | 1.036 / 1.036 |

The same bits out (`check.txt`). On this GPU the revised layout is within 0.7% of the 12-bit layout before it at 1-32
tokens and 0.4-0.8% faster at 256-1024. These runs were made one at a time.

## H100 SXM5 and A100 SXM4 40 GB (`h100-sxm5/`, `a100-sxm4-40gb/`)

H100 80GB HBM3 (1980 MHz, 700 W) and A100-SXM4-40GB (1410 MHz, 400 W), driver 580.126.20, PyTorch 2.14.0 (CUDA 13.0),
nvcc 13.0. The revised layout's layer time against the 12-bit layout's, each of the 2 runs of the 3 models (Qwen3-8B,
14B, 32B):

| GPU | Tokens | Revised / 12-bit | 12-bit / cuBLAS | Revised / cuBLAS |
| :-- | :-- | :-- | :-- | :-- |
| H100 | 1-16 | 0.983-0.998 | 0.86-0.99 | 0.85-0.98 |
| H100 | 32-64 | 0.948-0.962 | 0.92-0.99 | 0.88-0.95 |
| H100 | 128-512 | 0.955-0.969 | 1.11-1.66 | 1.06-1.59 |
| A100 | 1-16 | 0.976-0.999 | 0.79-0.93 | 0.79-0.91 |
| A100 | 256-768 | 0.969-0.981 | 1.31-1.50 | 1.27-1.47 |

On both, every tensor of Qwen3-8B unpacked bit for bit in both layouts (253), and every product the 12-bit layout's
bits, within 1e-2 of fp32 and the same on a second call (63 on the H100; 35 on the A100).
