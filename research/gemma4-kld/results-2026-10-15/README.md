# Gemma 4 31B-it QAT at i4x: KL divergence vs bf16, and the long-prompt check (2026-10-15)

Machine: Lambda gpu_1x_h100_sxm5 (H100 80 GB, driver 580, CUDA 12.8 at /usr/local/cuda), Ubuntu 24.04, ~25 min of use.
Engine: 3c9743be (g4score, decode path, ctx 1024); pack `glyd-engine pack m/qat qat-i4x --layout i4x` (411 packs, 242 s).
Reference: transformers 5.19.0, torch 2.7.0, Gemma4ForConditionalGeneration, bf16, eager attention, whole model on the GPU
(kld_ref_g4.py). Models: google/gemma-4-31B-it-qat-q4_0-unquantized (shards sha256 8ad3c678..., a373e714...) and google/gemma-4-31B-it.

Method (as llama.cpp's perplexity tool): wikitext-2-raw-v1 test (text sha256 bbf94c53a05abe9e..., 292,281 tokens with Gemma's
tokenizer), chunks of 512 from the start, the first 32, each chunk's first token replaced by BOS (2); logits at positions 256..510
scored (8,160). kld_compare.py: full 262,144-token rows, float64.

```
                                   ours vs QAT bf16    ours vs original bf16   QAT bf16 vs original bf16
mean KLD (± SE)                    0.0182 ± 0.0012     1.258 ± 0.016           1.258 ± 0.016
median KLD                         0.0030              0.735                   0.735
99% / 99.9% KLD                    0.325 / 1.47        6.37 / 10.43            6.26 / 10.28
top-1 agreement                    95.9%               58.1%                   58.3%
PPL (ref / ours)                   1185.4 / 1192.0     6799.7 / 1192.0         6799.7 / 1185.4
```

- Our weights are Google's QAT weights bit for bit (i4x); the 0.018 is the engine's arithmetic (bf16 rounding points, attention order,
  the decode path) against transformers', on text where the model is unsure (flat distributions amplify small differences). On chat
  prompts it is ~0.0005 (below).
- Google's QAT model is not the original with rounded weights: on this text it differs from gemma-4-31B-it by KLD 1.26 and scores a
  much lower perplexity (1185 vs 6800). So "vs the original" mostly measures Google's QAT itself; ours adds 0.018 on top of it.
- The instruction-tuned models score oddly high perplexity on raw wikitext (its " @-@ " and " , " spacing): transformers alone gives
  PPL 1147 on chunk 0's first 200 tokens and 3.2 on a plain English paragraph (sanity check on the same machine). The numbers are real
  but this text is a poor yardstick for -it models; a chat-style corpus would say more.

Long prompt (g4check, ref from ref_gemma4.py on the H100 with the QAT model, 6 prompts incl. one of 3,999 tokens, ctx 8192):
top-1 agrees 124/124, mean KL over the reference's top 8 0.00049; the 3,999-token prompt 24/24, KL 0.00061, read in 1.87 s.

The raw logs and the f32 logit files were not kept (the copy failed and the machine was terminated); these numbers are from the run's
printed output.
