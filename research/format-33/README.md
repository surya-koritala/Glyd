# format-33: a lossless bf16 code at 10.72 bits a weight (33.0%) as fast as the tiered layout?

Private research (glyd-gpu-private), 2026-10-06. A sampled compression study and static compiled-decoder count; no production kernel or format change.
Tensor samples from eight 2026 bf16 models were read by HTTP range on a Runpod pod. Exact constructed packs
are identified separately from size-model estimates. Whole-model rows are sample-weighted projections, not
complete checkpoint encodings or resident GPU measurements. Entropy estimates describe the sampled
distributions and conditioning schemes; they are not universal lossless-compression ceilings.

## Result

1. **The sampled memoryless coding limit exceeds 10.72 bits for one tensor.** gemma-4-12B-it's `embed_audio.embedding_projection` has an
   order-0 floor of 10.848 bits (32.2%); the measured marginal distribution does not support that target for memoryless coding. Conditioning can
   change the estimate (the best context measured here: 10.571); other structures were not ruled out.
2. **Among the fixed-rate variants searched, the smaller projections add a fourth tier and more static instructions.** The tiered layout as built is
   10.756 bits pooled (32.78%), with whole-model projections of 10.694-10.766 (32.7-33.2%). Of 2,700 fixed-rate codes searched (5 symbol
   choices, up to 4 tiers of 1-4-bit digits, 5 table scopes), the only ones at or under 10.72 pooled are the tiered layout plus a
   fourth tier: tier 3 gets an escape to a dense tier 4 (`E 2-2-2-3d`: 10.712 pooled, 33.05%; whole models 32.98-33.36%, one of
   eight (Qwen3.5-27B) still under 33.0%). Its straightforward decoder, written on the library's own decode and compiled for
   sm_89, has **+31% instructions in the hot path** of a step (506 against 386); the 2-bit tier-4 form (`2-2-2-2d`, 10.718,
   33.01%) +24% (479). These counts suggest additional decode cost; GPU timings of these variants were not measured, so their runtime ordering is unproven.
3. **The other searched combinations did not improve the pooled size estimate.** Joint codes of the exponent with the top mantissa bit (Em6), with the sign (SE), with two mantissa bits
   (Em65), the 12-bit layout's base trick under tiers (Eh), per-step / per-row-block / per-column-strip exponent offsets, and
   per-step adaptive tables are all at or above the plain exponent's best (table below). The sign carries no information about the
   exponent (H(S|E) = 1.000); the top mantissa bit 0.053 bit (H(m6|E) = 0.947), which digit codes cannot collect.
4. **The floors.** Order-0 (the whole 16-bit word, a table a tensor): 10.498 bits, **34.4%**; the best context measured (the
   exponent given its left neighbour and its row's mean exponent): 10.474, **34.5%**. A static rANS of the exponent reaches 10.588
   (33.8%), 10.650 (33.4%) with 32 lane states written every 16 steps; of exponent + top mantissa bit 10.600 (33.7%). Its decoder
   has 982 instructions a step (2.5x the tiered layout's) and decodes a strip of 16 steps in order.

**Disposition: retain these results as research; they do not justify changing the default.** Within the searched
three-tier structures, the best modeled size is 10.741 bits per weight (32.87%). The variants reaching
10.72 bits add static instructions, but static instruction counts are not measured execution speed.
The T4b pooled projection improves saving by about 0.27 percentage points, from 32.78% to 33.05%,
with 31% more static hot-path instructions. Device correctness, speed and full-checkpoint size remain
unmeasured for these variants; this study does not rule out other implementations or coding methods.

## Fixed-rate codes (pooled over 100 samples, 392 M weights)

Bits a weight with every header; "Worst": the worst sample; "<= 10.72": samples at or under the bar. Decode cost: warp
instructions in the hot path of one step (1,024 weights; loads, decode, the xor-fold of the 16 output words) of a decode-only
kernel, static SASS for sm_89 (`results/decode/count_89.txt`), and per value (/ 32 lanes' values... a lane decodes 32 weights).

| Code | Tiers (bits a digit) | Bits | Saving | Worst | <= 10.72 | Decode, hot path a step (sm_89) | vs tiered |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| Tiered layout as built (E) | 2-2-2, tier 3 dense, list | 10.756 | 32.78% | 11.160 | 23 | 386 (12.1 a value) | 1.00 |
| E, 3-bit dense tier 3 | 2-2-3 | 10.744* | 32.85% | | | not compiled (one PRMT pair a group more than tiered: ~1.02, estimate) | |
| **T4a: E, tier 3 escapes to a dense 2-bit tier 4** | 2-2-2-2 | 10.718* | 33.01% | 11.069 | 61 | 479 (15.0) | **1.24** |
| **T4b: E, tier 3 escapes to a dense 3-bit tier 4** | 2-2-2-3 | 10.712* | 33.05% | 11.059 | 64 | 506 (15.8) | **1.31** |
| T4b with an exponent offset a 64-row block / a step | 2-2-2-3 | 10.712* / 10.714* | 33.05% | 11.059 / 11.004 | 64 / 61 | T4b + about 1 IADD a tier word a step (estimate) | ~1.31 |
| T4a, the step's own ranking of the tensor's 16 commonest | 2-2-2-2 | 10.753* | 32.79% | 11.037 | 17 | not compiled (3 table words loaded a step) | |
| Em6: exponent + top mantissa bit | 3-3-3-4 | 10.736* | 32.90% | 11.179 | 47 | not compiled (3-bit tier 1 over 7-bit raw fields: more than T4b) | |
| SE: sign + exponent | 3-3-3-4 | 10.781* | 32.62% | 11.193 | 0 | not compiled | |
| Em65: exponent + 2 mantissa bits | 4-4-4-4 | 10.809* | 32.44% | 11.325 | 0 | not compiled | |
| Eh: the 12-bit layout's high byte under tiers | 1-1-1-2 | 10.843* | 32.23% | 11.132 | 0 | not compiled | |
| rANS of E, states every 16 steps (entropy-coded reference) | | 10.650 | 33.44% | | | 982 (30.7) | 2.54 |

\* The size model (f33.py) gives a literal a digit in every tier past the first; the library gives it a placeholder tier-2 digit
and none after. For the tiered layout the exact count (the model less 2 x P(rank >= 10)) matches the real packs to 0.0001
(below); for the other rows the surplus is 0.0003-0.007 on the five packed tensors, so they are upper bounds by that much.
Registers (ptxas, sm_89): 48 for every hot-path kernel; the full decode-only kernels 48 (tiered) and 56 (T4a, T4b).

Exact packs (`f33_pack.py`, `results/decode/pack.txt`), bits a weight:

| Tensor | Tiered (v2b) | T4a | T4b | rANS(E), states every 16 steps |
| :-- | --: | --: | --: | --: |
| Qwen3.5-9B layers.8.mlp.gate_proj | 10.7512 | 10.7108 | 10.7108 | 10.6185 |
| Qwen3.5-9B embed_tokens | 10.7294 | 10.6923 | 10.6923 | 10.6166 |
| gemma-4-12B-it embed_audio.embedding_projection | 11.1603 | 11.0618 | 11.0580 | 10.9546 |
| GLM-5.3-BF16 layers.10.mlp.experts.168.gate_proj | 10.7015 | 10.6734 | 10.6686 | 10.6037 |
| Qwen3.5-9B layers.31.self_attn.o_proj | 10.8332 | 10.7687 | 10.7673 | 10.7117 |

## Whole-model projections (sample means weighted by checkpoint matrix bytes)

| Model | Tiered as built | T4a | T4b | Order-0 floor | Best context |
| :-- | --: | --: | --: | --: | --: |
| Qwen/Qwen3.5-9B | 10.757 (32.77%) | 10.719 (33.01%) | 10.714 (33.04%) | 10.499 (34.38%) | 10.484 (34.48%) |
| Qwen/Qwen3.5-27B | 10.766 (32.71%) | 10.727 (32.96%) | 10.724 (32.98%) | 10.488 (34.45%) | 10.477 (34.52%) |
| Qwen/Qwen3.8-27B | 10.758 (32.76%) | 10.720 (33.00%) | 10.716 (33.02%) | 10.485 (34.47%) | 10.478 (34.52%) |
| google/gemma-4-12B-it | 10.761 (32.74%) | 10.726 (32.96%) | 10.719 (33.01%) | 10.524 (34.23%) | 10.498 (34.39%) |
| google/gemma-4-31B-it | 10.736 (32.90%) | 10.700 (33.12%) | 10.693 (33.17%) | 10.504 (34.35%) | 10.489 (34.44%) |
| ibm-granite/granite-4.2-8b | 10.749 (32.82%) | 10.711 (33.05%) | 10.708 (33.07%) | 10.443 (34.73%) | 10.429 (34.82%) |
| meta-models/Muse-Glimmer-30B | 10.740 (32.88%) | 10.704 (33.10%) | 10.701 (33.12%) | 10.487 (34.46%) | 10.477 (34.52%) |
| zai-org/GLM-5.3-BF16 (97.5% experts) | 10.694 (33.16%) | 10.669 (33.32%) | 10.663 (33.36%) | 10.469 (34.57%) | 10.466 (34.59%) |

These projections weight each sampled kind by its BF16 matrix bytes in the checkpoint, including embeddings,
LM heads, vision towers and MTP heads. They are not complete encoded checkpoints or serving-engine memory measurements. By kind (pooled, tiered exact / T4b): attention 10.774 / 10.725, linear attention 10.716 / 10.680, MLP
10.754 / 10.712, experts 10.692 / 10.661, embeddings 10.734 / 10.690, LM heads 10.726 / 10.687, vision 10.784 / 10.736, MTP
10.786 / 10.735. Every sample: `results/tables.txt`.

## Floors (pooled)

| What | Bits | Saving |
| :-- | --: | --: |
| H(16-bit word), a table a tensor (order-0 floor) | 10.498 | 34.4% |
| Best context (exponent given left neighbour's and row mean; the rest given the exponent) | 10.474 | 34.5% |
| H(E) + 8 raw (a perfect exponent code, nothing else) | 10.584 | 33.8% |
| H(E, m6) + 7 raw | 10.531 | 34.2% |
| H(E, m6, m5) + 6 raw | 10.516 | 34.3% |
| H(E) | 2.584 | |
| H(sign, mantissa \| E) | 7.913 | |
| H(sign \| E) / H(m6 \| E) / H(mantissa \| E) | 1.000 / 0.947 / 6.913 | |
| H(E \| left) / (E \| row) / (E \| column) / (E \| left, row) | 2.580 / 2.568 / 2.572 / 2.565 | |

Where the tiered layout's 0.26 bit over the order-0 floor goes: 0.17 is the exponent code (2-bit digits against H(E) = 2.584:
ranks 3-5 pay 4 bits, ranks past 9 a 22-bit literal), 0.03 the step offsets, 0.09 the mantissa and sign information no
per-weight code collects (H16 vs H(E) + 8).

## Samples

`samples.json` (`plan.py`, seed 0, from research/inventory-2026's headers): 100 slices of whole rows (64-row multiples, about
4 M weights each; 0.78 GB read), 12-13 a model: Qwen/Qwen3.5-9B, Qwen3.5-27B, Qwen3.8-27B (attention, Gated DeltaNet projections,
MLP, embedding, LM head, vision tower, MTP), google/gemma-4-12B-it, gemma-4-31B-it (attention, MLP, embedding, vision/audio
projections), ibm-granite/granite-4.2-8b, zai-org/GLM-5.3-BF16 (MLA attention, indexer, shared and routed experts, embedding, LM
head, MTP), meta-models/Muse-Glimmer-30B (attention incl. its gate, MLP, embedding, LM head, vision tower). All released in 2026.

## What was not done, and why

- **The decoders were not run on a GPU.** The second pod (RTX 2000 Ada, `f33_job2.sh`) packed the five tensors and built the
  harness for sm_89; it was gone 4.5 minutes in (01:28 EDT, during the sm_80 build; not deleted by this job's launcher), before
  the bit-for-bit unpack check, the decode-only timings, the sm_80 / sm_90a SASS and the ncu instruction count. A new pod was
  then refused by the launcher ("bf16 GPU work is paused while the quantized sessions run"), so it was not retried. The T4 and
  rANS decoders are therefore compiled but **not verified bit for bit**, and the decode costs above are static counts (all four
  groups of every warp-uniform tier loop counted), not dynamic counts or times.
- **The size model was not rerun with the exact literal rule** (same refusal); `tables.py` applies the exact rule to the tiered
  layout and states the bound for the rest (footnote above).
- Per-tensor codes that need context (the 0.02-0.28 bit between the order-0 floor and the best context) were measured as floors
  only.

## Files and how to reproduce

- `plan.py` (the samples), `f33.py` (sizes and floors; `--selftest` checks the tiered count against v2bits' formula),
  `tables.py RESULTS_DIR [HEADERS_DIR]` (the tables), `f33_pack.py` (the tiered layout byte for byte, T4a/T4b, rANS packers),
  `f33_dec.cu` (the decoders on the library's own code; `k_hot_*` the hot paths counted, `k_dec_*` / `k_unp_*` the decode-only
  and unpack kernels), `count.py` (static SASS counts with gpu/experimental/tiered_fast/sassstat.py).
- `results/sizes/NNN.json` (one a sample), `results/sizes-run.log`, `results/selftest.txt`, `results/env.txt`, `results/tables.txt`;
  `results/decode/` (`pack.txt`: exact bits of each pack, `count_89.txt`, `regs_89.txt`, `env.txt`).
- Pods: `f33a` (RTX 2000 Ada, 64 vCPU, 8.5 min, sizes) and `f33b` (RTX 2000 Ada, 4.6 min, packs and the sm_89 build); $0.05.
  Jobs: `python3 f33.py --selftest; PROCS=24 python3 f33.py samples.json OUT`; `python3 f33_pack.py samples.json 5,8,47,81,2 PACKS`,
  `nvcc -O3 -lineinfo -std=c++20 --expt-relaxed-constexpr -gencode arch=compute_89,code=sm_89 -I../../gpu -Xptxas -v -cudart static
  f33_dec.cu -o f33dec_89`, `cuobjdump -sass f33dec_89 > s; python3 count.py s`, `./f33dec_89 PACKS`.
