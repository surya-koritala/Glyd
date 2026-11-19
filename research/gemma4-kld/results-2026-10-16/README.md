# Gemma 4 12B QAT: the engine against transformers, each vs float32 (2026-10-16, the owner's PC, RTX 4080 SUPER)

Wikitext-2 test, 64 chunks of 512 (kld_tokens.py, BOS 2), in four batches of 16; logits at positions n/2 .. n-2. References: transformers 5.18
(eager), float32 and bf16, the model split between the GPU and the computer's memory. Ours: the wren pack (Google's QAT weights bit for bit,
glyd-engine ship), `g4score --ctx 1024`, branch gemma-penguin-run; "precise" is the same with GLYD_G4_PRECISE=1 (the residual in fp32).

| vs fp32 | batch 0 | batch 1 | batch 2 | batch 3 | mean of 4 | median (mean of 4) | same top-1 |
|---|--:|--:|--:|--:|--:|--:|--:|
| transformers bf16 | 0.114 | 0.087 | 0.062 | 0.144 | **0.102** | 0.0092 | 90.0% |
| ours | 0.076 | 0.137 | 0.082 | 0.139 | **0.109** | 0.0102 | 89.6% |
| ours, fp32 residual | 0.092 | 0.065 | 0.084 | 0.140 | **0.095** | 0.0082 | 90.3% |

The batches differ by about ±0.03 for the same engine: the -it model on raw wikitext is sensitive to any rounding pattern. Over 64 chunks
ours is at transformers' bf16 level (+6%, inside that spread); the fp32 residual is -7% (also inside it). The 20-25% gap read from 32 chunks of
the 12B penguin (0.170 vs 0.132-0.143) is not borne out. Cards: KLD vs fp32 beside transformers' own bf16 on the same tokens, 64+ chunks.
raw.txt: each batch's numbers.
