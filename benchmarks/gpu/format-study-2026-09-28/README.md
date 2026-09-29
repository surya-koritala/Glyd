# Packed formats: bits a weight against the decode's instructions, 2026-09-28

Candidates for a layout about as small as the tiered one with a decode about a third of the 12-bit one's, measured
on the CPU (sizes, exceptions) and in SASS (instructions), and the one picked (split byte, 12 bits) prototyped in the
library's own kernels (`gpu/experimental/format`) and timed on the RTX 4080 SUPER. `gpu/experimental/format` and the
scripts these logs name are on the branch gpu-format (d3fd2e6), not in this tree: the release's split byte is its own
code, in gpu/glyd_gpu.cu. Models: the box's cached Qwen3-0.6B, 1.7B, 4B-Instruct-2507 and 8B and
granite-3.1-3b-a800m-instruct (their Linear weights as `gpu/sizes.py` packs them; lm_head apart).

- `study.py`, `study.txt`: bits a weight and exceptions of each candidate, per tensor and per model, and how the
  exceptions fall in the kernels' steps and stages.
- `cols.py`, `cols.txt`: the 3-bit codes' escapes with a window a row or a column instead of one a tensor (Qwen3-8B).
- `count.txt` (`gpu/experimental/format/count.sh`): each candidate's decode alone, in SASS for sm_80, sm_89 and sm_90a.
- `count_lib.txt` (`count_lib.sh`): the library's step, mid, prompt and Hopper TMA kernels, 12-bit against split byte.
- `check.txt`, `selfcheck.txt`: the prototype on every tensor of the five models, bit for bit, and one layer's
  products against fp32 and against the 12-bit layout's.
- `layer-*-run*.txt`: one layer's products (q,k,v and gate,up merged) at 1, 8, 32, 256 and 1024 tokens, cuBLAS against
  both layouts in the same kernels, two runs a model.
- `h100-sxm5/`, `a100-sxm4-40gb/`: the same on an H100 SXM5 and an A100 SXM4 40 GB (Lambda, 2026-09-28), run by the
  unattended job in `cloud/` (`format_job.sh`, its launchers and its summary; the sources `git archive` of 47c95d5's
  gpu/glyd_gpu.cu and .h, gpu/experimental/format and bindings/python/glyd): the self-check, layer 10 of Qwen3-8B, 14B
  and 32B at 1-512 tokens (H100) and 1-768 (A100), two runs each, and check.py on the whole Qwen3-8B.

## Bits a weight (and exceptions, share of weights)

| Candidate | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B-Instruct-2507 | granite-3.1-3b-a800m | Qwen3-8B |
| :-- | :-- | :-- | :-- | :-- | :-- |
| floor: 8 + the exponents' entropy | 10.64 | 10.61 | 10.63 | 10.63 | 10.64 |
| tiered (`mma`) | 10.82 | 10.77 | 10.85 | 10.80 | 10.87 |
| 12-bit (`mma12`) | 12.04, 0.03% | 12.04, 0.02% | 12.04, 0.03% | 12.07, 0.13% | 12.04, 0.04% |
| split byte, 12 bits (`sb12`) | 12.04, 0.02% | 12.04, 0.02% | 12.04, 0.02% | 12.07, 0.12% | 12.04, 0.03% |
| fast (3-bit, 7 commonest) | 11.30, 3.3% | 11.25, 2.7% | 11.30, 3.3% | 11.32, 3.4% | 11.37, 4.2% |
| 3-bit by value, 7 exponents (`v11`) | 11.30, 3.3% | 11.25, 2.7% | 11.30, 3.3% | 11.32, 3.4% | 11.38, 4.3% |
| the same, a window a row (`v11r`) | 11.27, 2.9% | 11.24, 2.6% | 11.26, 2.8% | 11.28, 2.7% | 11.28, 3.1% |
| split byte, 11 bits (`sb11`, 24-bit list) | 11.46, 1.8% | 11.39, 1.5% | 11.57, 2.3% | 11.78, 3.1% | 11.67, 2.7% |
| the same, a window a row (`sb11r`) | 11.44, 1.7% | 11.38, 1.4% | 11.49, 1.9% | 11.64, 2.5% | 11.47, 1.8% |

A code of fixed width takes at least 3 bits for the exponent (11 a weight), and the 7 or 8 exponents 3 bits hold leave
1.4-4.3% of the weights out on every model (up to 37% of a few early layers' matrices: Qwen3-8B's layers 1-3 MLP,
Qwen3-4B's layer 2 gate, granite's layers 0-1 attention); a window a row or a column does not narrow it (the rows' and
the columns' mean exponents spread by a standard deviation of 0.07-0.42). Their escapes: 15-44 a step (1024 weights)
on average, and no step without; the 12-bit layouts' 0.2-1.4, and 75-86% of steps without.

## Decode instructions, a k-block (8 weights a lane), from SASS (`count.txt`)

| | integer pipe | FMA pipe | integer instructions against the 12-bit layout's |
| :-- | --: | --: | --: |
| 12-bit (today: a 16-entry table, a permute and a rotate a pair) | 22.0 | 3.3 | 1 |
| 12-bit by value (no table) | 12.3 | 0.8 | 0.56 |
| split byte, 12 bits | 8.5 | 0.5 | 0.39 |
| split byte, 12 bits, the base OR'd in (the product scaled back after) | 6.8 (sm_90a) | 0.5 | 0.31 |
| split byte, 11 bits, no escapes | 8.3 | 0.8 | 0.38 |
| the same with its escapes through a warp's scratch (15-27 a step) | about 16.5 | 1.5 | about 0.75 |
| 3-bit by value (7 exponents), no escapes | 12.3 | 0.8 | 0.56 |

A stage as the mid and Hopper kernels decode it (4 k-blocks of a warp's 16 rows): 81 integer instructions and 10-20 on
the FMA pipe (12-bit), 26-30 and 4-6 (split byte). In the library's Hopper TMA kernel (sm_90a) its consumers' loop
goes from 65.4 to 51.6 integer instructions a k-block with the split-byte decode (`count_lib.txt`).

## One layer, RTX 4080 SUPER (`layer-*`; time against cuBLAS, run 1 / run 2)

| Model | M | cuBLAS us | 12-bit | split byte | kernel |
| :-- | --: | --: | :-- | :-- | :-- |
| Qwen3-8B | 1 | 671 | 0.888 / 0.887 | 0.887 / 0.885 | step |
| | 8 | 751 | 0.797 / 0.798 | 0.797 / 0.797 | step |
| | 32 | 863 / 873 | 0.709 / 0.702 | 0.708 / 0.701 | mid (step: 0.720 / 0.712 both) |
| | 256 | 1173 / 1165 | 0.918 / 0.917 | 0.913 / 0.912 | prompt |
| | 1024 | 4033 / 4031 | 0.983 / 0.985 | 0.979 / 0.979 | prompt |
| Qwen3-4B-Instruct-2507 | 1 | 394 / 395 | 0.840 / 0.837 | 0.838 / 0.837 | step |
| | 8 | 435 / 436 | 0.762 / 0.765 | 0.767 / 0.766 | step |
| | 32 | 462 / 461 | 0.745 / 0.747 | 0.747 / 0.744 | mid (step: 0.790 / 0.790, 0.777 / 0.781) |
| | 256 | 618 | 0.973 / 0.975 | 0.967 / 0.967 | prompt |
| | 1024 | 2054 / 2085 | 1.040 / 1.040 | 1.036 / 1.036 | prompt |

The same kernels either way, and the same bits out (`check.txt`): on this GPU the decode is hidden (steps bound by
memory, the prompt kernel's decode on producer warps, fp32-accumulating GeForce tensor cores), and the split byte is
within 0.7% of the 12-bit layout at 1-32 tokens (1.1-1.6% faster in the step kernel at 32 on Qwen3-4B) and 0.4-0.8%
faster at 256-1024. The box reset at 17:35 EDT, before any GPU job of this study (its CPU study had ended at
17:34:30); these runs were made after it, one at a time under the box's lock.

## H100 SXM5 and A100 SXM4 40 GB (`h100-sxm5/`, `a100-sxm4-40gb/`)

H100 80GB HBM3 (1980 MHz, 700 W) and A100-SXM4-40GB (1410 MHz, 400 W), driver 580.126.20, PyTorch 2.14.0 (CUDA 13.0),
nvcc 13.0. The split byte's layer time against the 12-bit layout's in the same kernel, each of the 2 runs of the 3
models (Qwen3-8B, 14B, 32B), on the kernel main runs there:

| GPU | Tokens | Kernel | Split byte / 12-bit | 12-bit / cuBLAS | Split byte / cuBLAS |
| :-- | :-- | :-- | :-- | :-- | :-- |
| H100 | 1-16 | step | 0.983-0.998 | 0.86-0.99 | 0.85-0.98 |
| H100 | 32-64 | TMA | 0.948-0.962 | 0.92-0.99 | 0.88-0.95 |
| H100 | 128-512 | TMA | 0.955-0.969 | 1.11-1.66 | 1.06-1.59 |
| A100 | 1-16 | step | 0.976-0.999 | 0.79-0.93 | 0.79-0.91 |
| A100 | 256-768 | prompt (variant 3) | 0.969-0.981 | 1.31-1.50 | 1.27-1.47 |

Also timed, not main's route there: on the H100 the step kernel at 32-64 tokens, 0.967-0.988; on the A100 the step
kernel at 32 tokens, 0.981-1.010, at 64, 0.974-0.983, and the prompt kernel at 128, 0.949-0.975 (main runs the A100's
own kernel at 17-128, which the prototype does not have).

On both, every tensor of Qwen3-8B unpacked bit for bit in both layouts (253), and every product the 12-bit layout's
bits, within 1e-2 of fp32 and the same on a second call (63 on the H100, the TMA kernel's among them; 35 on the A100).

**Against the go criteria set before the run** (the H100 at most 0.95x the 12-bit layout's time at 128-512 tokens and
1.00x at 1-64; the A100 at most 1.00x at 1-16 and 129-768; the same products): the H100's 128-512 criterion is not met
(0.955-0.969); the other three are.

**Where the projection was wrong.** It projected 5-9% less layer time on the H100 at 256-1024 tokens, from gpu-hopper2's
ablation builds of its own wgp kernel (the decode's 21 integer instructions a k-block against 11). Main's TMA kernel,
measured here at 128-512 tokens, gains 3.1-4.5%. Its consumers' loop holds 65 integer instructions a k-block
(`count_lib.txt`) and the split byte removes 14 of them, a fifth; the loop's addresses, bounds and exception check stay.
The A100's was projected at 0-3% and measured 1.9-3.1% at 256-768 tokens.
