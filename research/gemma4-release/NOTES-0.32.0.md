Glyd 0.32.0

- Gemma 4 12B and 31B at `:penguin`, lossless: `glyd run glyd/gemma-4-12B-penguin` downloads `glyd/gemma-4-12B-penguin`
  (16.1 GB, plus Google's 0.88 GB draft model); the 31B is `glyd/gemma-4-31B-penguin` (41.3 GB + 0.97 GB). Every weight is
  Google's bf16 weight bit for bit (`glyd verify` checks it), at about 10.8 bits per weight: the 12B takes 15.0 GiB on the GPU in
  place of 22.2 GiB and runs on a 24 GB card; the 31B takes 38.4 GiB in place of 57.2 GiB and runs on a 48 GB card.
  Measured, one user, greedy, with the draft model (without it in brackets):
  - RTX 4090, 12B: 89.8 tokens a second in chat (50.8); a 28K-token prompt read in 7.6 s.
  - RTX A6000 48 GB, 31B: 34.3 in chat (13.3); a 24K-token prompt read in 35.5 s.
  - H100 PCIe, 31B: 61.8 in chat (30.0); the 24K-token prompt in 23.0 s.
  - GSM8K 72/75 and MMLU-Pro 67/75, both sizes. KLD against transformers' float32 run (wikitext-2, 32 x 512), 12B: 0.170,
    where transformers' own bf16 runs are at 0.132 and 0.143. The cards have the details.
- `glyd run google/gemma-4-12B-it` (and 31B) with no level runs penguin on Glyd's engine where the GPU has room, and downloads
  the penguin repo above. Before, the engine's check said it runs Qwen3 models only and handed Gemma 4 to vLLM.
- `glyd serve`: a request that fails has its reason in the server's log, not only in the reply to the client.

Known limits
- Linux + NVIDIA RTX 30 series or newer only. The 12B penguin needs 24 GB, the 31B 48 GB.
- Text only; one request at a time.
- On H100 the Gemma engine is not tuned yet: prompt reading runs well below the card's peak, and the 31B without the draft model
  writes at about 62% of what the card's memory bandwidth allows.

Install: pip install glyd==0.32.0 (or brew upgrade glyd)
