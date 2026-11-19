# KL divergence vs bf16, Qwen3.8-Flash-Next, 2026-10-15

Machine: owner's PC, RTX 4080 SUPER 16 GB, Ryzen 9 7950X3D, 60 GiB RAM, Ubuntu 26.04. Engine: qwen-users 34c100cf, dev build for sm_89
(sha256 83f77febd942...), pack precise_x (`--precise`) and fast_x, model dir qwen/slim (the release's: n-gram table at 4 bits).
Reference: the BF16 checkpoint (Qwen/Qwen3.8-Flash-Next @ de4b8e4d), research/qwen-data/qwen_ref.py layer at a time, transformers 5.19,
torch 2.11, sdpa, LM head in fp32.

## Method (llama.cpp's perplexity tool's way of cutting the text)

- Text: wikitext-2-raw-v1 test, Hugging Face `Salesforce/wikitext`, `wikitext-2-raw-v1/test-00000-of-00001.parquet`
  (sha256 of the parquet 5f1bea067869d048...), rows joined with "" (text sha256 bbf94c53a05abe9e..., 297,053 tokens).
- Tokens: the model's tokenizer.json, no special tokens added. Chunks of 512 consecutive tokens from the start; the first 32.
- Scored: in each chunk, the logits at positions 256..510 (they predict tokens 257..511), 255 a chunk, 8,160 in all. Each chunk is
  read fresh (no state carried between chunks). Ours: the first 257 tokens read as a prompt, the rest teacher-forced one at a time.
- KLD per position = sum over the vocabulary of p_ref * (log p_ref - log p_ours), full 248,320-token rows, float64.
  PPL = exp(mean NLL of the true next token). Top-1 = argmax equal.

## Numbers

```
wikitext-2 test, 32 x 512, 8,160 positions     precise            fast
mean KLD (± SE)                                 0.404 ± 0.010      0.526 ± 0.012
median KLD                                      0.061              0.113
99% KLD                                         4.27               5.10
99.9% KLD                                       8.88               8.69
PPL bf16 / ours / ratio                         2.242 / 2.831 / 1.263   2.242 / 3.035 / 1.354
top-1 agreement                                 82.2%              79.1%
```

A second text (ref_long.json: the 5,923-token mixed prompt behind the engine's earlier gates, 11 x 512, 2,805 positions):
precise mean KLD 0.106 ± 0.009 (median 0.00045, top-1 96.4%, PPL ratio 1.22); fast 0.192 ± 0.013 (median 0.0028, top-1 94.1%).

## What the big numbers are

The mean is driven by a tail: the median is 0.06 (precise) but 1% of positions are above 4.3. The largest ones are text the bf16 model
knows by heart: wikitext's first article (Robert Boulter) where bf16 predicts the next word of a title it has never been shown in the
context ("theatre production of" -> " How", p 0.993, also with every later token cut off, so not a leak) and ours predicts " the"
(p 0.998). On the two positions checked, our generation path gives the same answer as the scoring path, and so does the full
bf16 n-gram table (model dir flash-next instead of slim), so it isn't the scoring code or the n-gram table there. Which of the
quantized parts (experts at ~2.7/2.1 bits, always-on matrices at 4 bits, attention and LM head at 5) costs how much isn't split out
yet. Wikitext is full of memorized facts, which seem to be what the quantization loses most. Only compare with numbers made the same way on the same model (other models' or other
texts' KLD tables don't compare).

## Rerun

On the PC (~/glyd-scratch/qwen-users/kld), with research/ copied there:

```
venv/bin/python research/qwen-kld/kld_tokens.py wikitext-2-raw-v1/test-00000-of-00001.parquet $Q/flash-next/tokenizer.json chunks64.json --chunks 64
# chunks32.json = the first 32 of chunks64.json
bash chain.sh     # reference (lock per layer), then precise and fast through the engine (lock per run), then kld_compare.py
```

Time on the 4080: the reference about 2.5 min per 16 chunks (disk-bound: 336 GB read per pass), the engine about 4 min per 32 chunks.
The f32 logit files (8 GB each for 32 chunks) were deleted after the comparison; the per-run metadata is kept here (`*.f32.json`).
