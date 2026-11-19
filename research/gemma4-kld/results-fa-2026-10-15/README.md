# Gemma 4 31B QAT (i4x): the prompt attention on the tensor cores vs g4_attn_t, against transformers (2026-10-15)

Lambda gpu_1x_a100_sxm4 (A100 80 GB, Ubuntu 22.04), engine e2d7265f; reference transformers 5.19.0 bf16 eager, whole model on the GPU.
Wikitext-2 test, 32 x 512 + BOS (kld_tokens.py), positions 256..510; g4score (prompt n/2 + 1 tokens, then decode steps).

```
                   mean KLD   median    99.9%    top-1     PPL ratio
ref vs old         0.0225     0.0036    1.789    95.49%    1.0018      (g4_attn_t, GLYD_G4_FA=0)
ref vs new         0.0192     0.0030    1.488    95.82%    1.0003      (g4_fa256 + g4_fa512b)
old vs new         0.0217     0.0033    1.672    95.72%
28,672-token prompt: old 56.1 s (511 tokens/s), new 28.3 s (1,013 tokens/s)
```
The tensor-core kernels are closer to transformers and twice as fast on a long prompt: they are the default from here.
On the 12B (i6, RTX 4080 SUPER): 16K 9.55 -> 6.15 s, 28K 22.66 -> 13.12 s. Random-input checks (g4fatest): largest difference 0.0002
at score scale 0.5, 0.008-0.016 at scale 4-8 (bf16 rounding of the scores in another order).
