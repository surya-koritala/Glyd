# Nemotron 3 Super expert-offload probe (2026-10-10)

`research/lossy-2026/nemotron_probe.py` (v030 8e0dd8c8) on four A100-SXM4-80GB, 16 minutes of job (about $3). Model:
nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 at 2dc98e2a, bf16 as released, nothing rounded. Text: glm_skew.py's held-out set, 24
sequences of 2,048 tokens over six domains (43,253,760 routed token-expert pairs over the 40 MoE layers), and 1,260 greedy decode steps
(1,108,800 pairs). Linear GLY-130. The report is probe.md (with its "how to read it"); probe.json, prefill.npz and decode.npz hold the
numbers.

| measure | held-out text | decode steps |
|---|---|---|
| an expert's 2,688 neurons exactly 0 after relu² | 91.8% (p10 83.7%, p90 98.5%) | 91.6% (p10 83.4%, p90 98.5%) |
| columns of down kept for the output within 1% / 0.1% (largest activations first) | 4.4% / 5.6% | 4.5% / 5.7% |
| bytes of a 4-bit expert read if down is read for those columns only | 52% of 2,752,512 B | 52% |
| the same if up were read for the active neurons only too | 4.4% to 5.6% | 4.5% to 5.7% |
| active neurons named from the calibration profile, 25% of them (672) | recall 52.7%, output error 59.8% | |
| next MoE layer's experts from its own router one block early (d = 1) | recall 74.5%; naming 44: 91.1% | |
| the same at the previous MoE layer's input (d = 2, 3) | recall 68.7% (weighted 74.5%) | |
| a cold expert's (least-used 50%) contribution when used, against a hot one's (most-used 13%) | 1.00 as the mean of the layers' ratios; 0.82 pooled over the layers (8.81 against 10.71) | 1.17 as the mean of the layers' ratios |

How to read it:
- **Sparsity is exact and large.** relu² makes 92% of an expert's activations exactly 0 for a token, and the rest are concentrated: 4.4%
  of the columns of down carry the output to 1%. The zeros need no column of down at all, so an expert's output can be computed from up in
  full and down for the active neurons only, the same result up to the order of a float sum. Down is half of an expert's bytes.
- **The active set is not predictable from a profile.** Each token picks its own neurons: the calibration profile's most frequent 25%
  hold half of them and leave 60% of the output out. Up has to be computed (or a better predictor found) before down's columns are known.
- **Look-ahead is partial.** The next MoE layer's router on the stream one block early names 74.5% of its experts, and 91% when 44 are
  named: enough to prefetch most, not all.
- **Cold experts matter when used.** The least-used half contributes about as much per use as the most-used experts, so holding them at
  fewer bits trades quality in proportion to their use (the quality job's mix13 and cold50 rows), not for free.
- **Checks:** the layer-by-layer pass reproduces the model's forward (KL 0, top-1 100%); the router at d = 0 names every chosen expert;
  the bisection that counts columns agrees with a full scan on 1,270 of 1,280 counts (at most 6 columns apart).

Files: probe.md, probe.json, meta.json, prefill.npz, decode.npz, steps.txt, versions.txt, run.log, selftest.log, cpus.txt, gpus.txt,
prep.json.
