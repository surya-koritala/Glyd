# Quantized checkpoints of 2026, losslessly: how much VRAM is left to save

Research from the former private repo glyd-gpu-private (now part of this repository). A measurement, no product code: 17 quantized checkpoints released in 2026 (NVFP4, MXFP4,
FP8 with a scale a tensor, a row or a block, INT4 from GPTQ, AWQ and compressed-tensors, INT8 W8A8), their real tensors read by
HTTP range from the public Hugging Face repos on a Runpod pod's CPU (2026-10-06). For each: the order-0 floor and the best of a few
contexts, for codes and scales apart; a fixed-rate layout (a few integer operations a value to decode); and the whole checkpoint
projected (codes + scales + the 16-bit parts every one of them keeps). Lossless means bit for bit throughout: every number below
is of the stored bits, nothing is requantized.

## Result

Over the 17 checkpoints (2.75 TB stored), by format, each format's own bytes:

| Format | Checkpoints with it | Their downloads (all time) | Its bytes in them | Saved at a fixed rate | of which the codes | Best contexts |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| MXFP4 | 3 | 6,750,453 | 1,963.6 GB | 87.3 GB (4.4%) | 0.0 GB | 240.9 GB (12.3%) |
| 16-bit | 17 | 148,274,317 | 230.9 GB | 74.3 GB (32.2%) | 74.3 GB | 82.8 GB (35.9%) |
| FP8 | 8 | 103,363,802 | 421.1 GB | 69.9 GB (16.6%) | 65.3 GB | 84.4 GB (20.1%) |
| INT4 | 5 | 25,754,920 | 66.0 GB | 4.5 GB (6.9%) | 3.1 GB | 8.1 GB (12.2%) |
| NVFP4 | 4 | 52,986,780 | 57.5 GB | 3.1 GB (5.3%) | 0.0 GB | 4.4 GB (7.7%) |
| INT8 | 1 | 240,567 | 5.3 GB | 0.5 GB (9.4%) | 0.5 GB | 0.7 GB (12.4%) |

("Checkpoints with it": FP8 also counts the FP8 attention of three NVFP4 checkpoints and DeepSeek-V4.1-Flash's FP8 parts.
Downloads: all time, the Hub API's `downloadsAllTime`, 2026-10-06; the last 30 days' counts are in `results/hub.json` too.
Each format's parts are projected group by group, each (format, part, attention / MLP / experts / ...) group at its own samples'
ratio: see Method. "Saved at a fixed rate" is built and checked for FP8's codes only; for the others it is computed from the samples,
not built. The 16-bit row counts cyankiwi's fp16 embedding and LM head at the tiered code over fp16's exponent, 13.7 bits, a layout
Glyd does not have.)

- **FP8 has 17-20% to give, and a fixed-rate layout takes 16.6% of its bytes.** The lossless FP8 format as built (`fp8ref` layout 2,
  fp8-lossless@42e6999f, the same file on private main) is **6.717 bits a value on Qwen3.8-27B-FP8** here, against 6.719 for layout 2 in
  `benchmarks/gpu/fp8-linear-2026-10-05/report.md` on that repo's 10 samples of its own (6.722 over its five checkpoints; main's
  `gpu/fp8/README.md` still gives layout 1's 6.685). It is 6.69-6.83 on every other FP8 checkpoint except one: Gemma 4's experts in
  RedHatAI's FP8-dynamic (704 columns, not a multiple of the 128-column patch) take 7.28-7.30 (6.94 over that checkpoint). Every
  FP8 sample went through `pack` and `unpack` and came back bit for bit.
- **NVFP4 and MXFP4 codes have almost nothing a fixed-rate code can take.** NVFP4's nibbles are 3.93-3.97 bits order-0 (1-2%);
  MXFP4's 3.72-3.89 (3-7%, more in Kimi K3 and MiniMax-M3 than in DeepSeek-V4.1-Flash). Their room is in the scales: MXFP4's E8M0
  scales are 0.5-1.0 bit order-0 (of 8) and 1.8-2.0 at a fixed rate; NVFP4's E4M3 scales 3.6-4.0 bits order-0 and 4.1-4.3 at a fixed
  rate (half of their bytes, 2-4% of the checkpoint).
- **INT4 depends on the quantizer.** Symmetric min-max groups (Qwen's GPTQ, RedHatAI's INT4) leave codes of 3.43-3.46 bits (a fixed
  rate takes 3.58-3.61: 10%); AWQ and MSE-fitted asymmetric groups 3.67-3.89 (a fixed rate 3.94-4.00: nothing); Google's QAT
  INT4 3.77 (nothing). Their 16-bit group scales code to 8.9-11.9 bits of 16.
- **The 16-bit parts are the biggest cheap win.** Every quantized checkpoint keeps embeddings, LM heads, vision towers, MTP heads
  and often the attention in bf16 (0.8% to 64% of the file). Their bf16 parts (226 GB over the 17) code as bf16 checkpoints do: the
  tiered code without its tile heads takes 32.6% (74 GB), Glyd's 12-bit layout 25% (56 GB). 5.1 GB of them is fp16 (cyankiwi's
  embedding and LM head), which Glyd has no layout for. Plan item 1 has the table by layout.
- **Whole checkpoints: 6.5% to 24.2% at a fixed rate, 12% to 26% at the best contexts measured.** The checkpoint table below.

## The checkpoints

| Checkpoint | Format | Downloads (all time) | Stored GB | Quantized codes / scales / 16-bit | Fixed-rate layouts | Order-0 floor | Best contexts | GB saved (fixed rate) |
| :--- | :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| [RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic) | FP8, a scale a row | 5,630,973 | 28.6 | 85.5% / 0.1% / 14.4% | 15.2% | 21.0% | 21.0% | 4.3 |
| [mistralai/Mistral-Small-4-119B-2603](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) | FP8, a scale a tensor | 670,091 | 120.9 | 97.5% / 0.0% / 2.5% | 16.8% | 19.4% | 19.4% | 20.3 |
| [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) | FP8, block scales | 44,802,569 | 37.5 | 92.0% / 0.0% / 8.0% | 17.5% | 19.7% | 19.9% | 6.5 |
| [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | FP8, block scales | 11,437,964 | 30.9 | 80.0% / 0.0% / 20.0% | 19.4% | 21.3% | 21.7% | 6.0 |
| [cyankiwi/Qwen3.6-27B-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.6-27B-AWQ-INT4) | INT4 (compressed-tensors) | 9,271,261 | 20.4 | 60.4% / 9.4% / 30.1% | 7.8% | 15.2% | 16.3% | 1.6 |
| [google/gemma-4-12B-it-qat-w4a16-ct](https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct) | INT4 (compressed-tensors) | 6,199,967 | 10.3 | 53.1% / 6.6% / 40.2% | 17.0% | 23.4% | 23.6% | 1.7 |
| [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) | INT4 (compressed-tensors) | 740,121 | 19.5 | 62.5% / 2.0% / 35.5% | 19.3% | 22.2% | 22.4% | 3.8 |
| [QuantTrio/Qwen3.6-35B-A3B-AWQ](https://huggingface.co/QuantTrio/Qwen3.6-35B-A3B-AWQ) | INT4 AWQ | 4,538,080 | 25.4 | 61.7% / 2.4% / 35.5% | 13.4% | 19.3% | 20.1% | 3.4 |
| [Qwen/Qwen3.5-35B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4) | INT4 GPTQ | 5,005,491 | 24.4 | 66.0% / 2.6% / 30.6% | 17.6% | 20.5% | 20.7% | 4.3 |
| [RedHatAI/Qwen3.5-9B-quantized.w8a8](https://huggingface.co/RedHatAI/Qwen3.5-9B-quantized.w8a8) | INT8 W8A8 | 240,567 | 14.0 | 37.9% / 0.0% / 62.1% | 24.2% | 26.0% | 26.0% | 3.4 |
| [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | MXFP4 | 5,271,523 | 1,560.9 | 87.2% / 5.5% / 7.3% | 6.5% | 12.9% | 14.1% | 100.8 |
| [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | MXFP4 (+ FP8) | 869,321 | 510.3 | 94.6% / 4.6% / 0.8% | 10.0% | 13.0% | 15.2% | 51.0 |
| [amd/MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4) | MXFP4 | 609,609 | 242.7 | 85.8% / 5.4% / 8.8% | 7.0% | 15.6% | 16.3% | 17.1 |
| [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) | NVFP4 (+ FP8 attention) | 37,345,625 | 23.4 | 75.6% / 8.8% / 15.6% | 10.2% | 11.7% | 12.1% | 2.4 |
| [nvidia/Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) | NVFP4 (bf16 attention) | 13,033,896 | 32.6 | 31.9% / 4.0% / 64.1% | 23.2% | 24.4% | 24.5% | 7.6 |
| [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) | NVFP4 (+ FP8 mixer) | 2,016,556 | 21.6 | 74.1% / 8.8% / 17.0% | 10.4% | 12.3% | 13.1% | 2.2 |
| [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) | NVFP4 (+ FP8 attention) | 590,703 | 21.9 | 74.8% / 5.2% / 19.9% | 14.4% | 16.2% | 16.7% | 3.1 |

"Quantized codes / scales / 16-bit": shares of the stored bytes (the rest: norms, biases, per-tensor scalars, `g_idx`, left as
stored). Mistral-Small-4's repo holds the weights twice (Mistral's `consolidated-*` and the Hugging Face `model-*` files): only the
latter, 120.9 of its 241.9 GB, is counted. DeepSeek-V4.1-Flash's FP8 includes its two `engram.embed` lookup tables (196.6 GB, 38.5% of
the file, coded at 6.75 and 6.73 bits); whether an engine keeps them in GPU memory is the engine's choice. Its two `engram.wkv`
projections (0.3 GB) are sampled apart (6.96 and 6.87 bits); its FP8 attention, MLP and MTP weights (7.1 GB, 1.4% of the file) were not
sampled and are projected at the ratio of its four FP8 samples.

Where each checkpoint's fixed-rate saving comes from (GB, and share of the checkpoint):

| Checkpoint | the quantized codes | their scales and zero points | the 16-bit parts | Total |
| :--- | ---: | ---: | ---: | ---: |
| RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic | 3.0 (10.4%) | 0.0 (0.1%) | 1.4 (4.8%) | 4.3 (15.2%) |
| mistralai/Mistral-Small-4-119B-2603 | 19.3 (16.0%) | 0.0 (0.0%) | 1.0 (0.8%) | 20.3 (16.8%) |
| Qwen/Qwen3.6-35B-A3B-FP8 | 5.6 (14.8%) | 0.0 (0.0%) | 1.0 (2.7%) | 6.5 (17.5%) |
| Qwen/Qwen3.8-27B-FP8 | 4.0 (12.8%) | 0.0 (0.0%) | 2.0 (6.6%) | 6.0 (19.4%) |
| cyankiwi/Qwen3.6-27B-AWQ-INT4 | 0.0 (0.0%) | 0.5 (2.5%) | 1.1 (5.2%) | 1.6 (7.8%) |
| google/gemma-4-12B-it-qat-w4a16-ct | 0.0 (0.0%) | 0.3 (2.9%) | 1.4 (14.1%) | 1.7 (17.0%) |
| RedHatAI/Qwen3.8-27B-INT4 | 1.3 (6.7%) | 0.2 (0.9%) | 2.3 (11.8%) | 3.8 (19.3%) |
| QuantTrio/Qwen3.6-35B-A3B-AWQ | 0.2 (0.9%) | 0.2 (0.8%) | 3.0 (11.7%) | 3.4 (13.4%) |
| Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 | 1.6 (6.5%) | 0.2 (0.9%) | 2.5 (10.1%) | 4.3 (17.6%) |
| RedHatAI/Qwen3.5-9B-quantized.w8a8 | 0.5 (3.6%) | 0.0 (0.0%) | 2.9 (20.7%) | 3.4 (24.2%) |
| moonshotai/Kimi-K3 | 0.0 (0.0%) | 64.2 (4.1%) | 36.6 (2.3%) | 100.8 (6.5%) |
| deepseek-ai/DeepSeek-V4.1-Flash | 32.0 (6.3%) | 17.7 (3.5%) | 1.3 (0.3%) | 51.0 (10.0%) |
| amd/MiniMax-M3-MXFP4 | 0.0 (0.0%) | 10.1 (4.1%) | 7.0 (2.9%) | 17.1 (7.0%) |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | 0.2 (0.8%) | 1.0 (4.3%) | 1.2 (5.1%) | 2.4 (10.2%) |
| nvidia/Gemma-4-31B-IT-NVFP4 | 0.0 (0.0%) | 0.6 (1.9%) | 6.9 (21.2%) | 7.6 (23.2%) |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | 0.1 (0.6%) | 0.9 (4.2%) | 1.2 (5.6%) | 2.2 (10.4%) |
| nvidia/Qwen3.8-27B-NVFP4 | 1.2 (5.3%) | 0.5 (2.4%) | 1.4 (6.6%) | 3.1 (14.4%) |

(In the NVFP4 rows "codes" is their FP8 attention or mixer; DeepSeek's is its FP8 part, most of it the engram tables.)

## Codes and scales, bits

Pooled over every sample of a format (weighted by values; scales of different stored widths apart: FP8's 8-bit ones are
DeepSeek-V4.1-Flash's E8M0, its 16-bit ones the BF16 scales of the other FP8 checkpoints). "Fixed-rate layout": FP8's is the format as
built and checked (`fp8ref.pack`, then `unpack` bit for bit); every other one is computed from the samples' histograms, not built (see
Method).

| Format (repos) | Part | Stored bits | Order-0 floor | Best contexts | Fixed-rate layout | Values sampled |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| NVFP4 (4) | codes | 4 | 3.957 | 3.926 | 4.000 | 449.1 M |
| NVFP4 (4) | scales | 8 | 3.766 | 3.743 | 4.165 | 28.1 M |
| MXFP4 (3) | codes | 4 | 3.763 | 3.707 | 4.000 | 369.6 M |
| MXFP4 (3) | scales | 8 | 0.641 | 0.640 | 1.897 | 11.6 M |
| FP8 (8) | codes | 8 | 6.545 | 6.517 | 6.745 | 386.4 M |
| FP8 (8) | scales (8-bit) | 8 | 0.930 | 0.912 | 2.001 | 0.8 M |
| FP8 (8) | scales (16-bit) | 16 | 7.511 | 7.488 | 8.470 | 26 k |
| INT4 (5) | codes | 4 | 3.691 | 3.660 | 3.855 | 706.7 M |
| INT4 (5) | scales | 16 | 9.298 | 9.226 | 10.306 | 16.2 M |
| INT8 (1) | codes | 8 | 7.024 | 7.011 | 7.245 | 125.8 M |
| INT8 (1) | scales | 16 | 7.405 | 7.405 | 8.522 | 29 k |
| 16-bit parts (17) | weights | 16 | 10.282 | 10.282 | 10.817 | 291 M |

By checkpoint and format:

| Checkpoint | Format | Codes: order-0 / best contexts / fixed rate (bits) | Scales: stored / order-0 / fixed rate (bits) | Zero points: order-0 / fixed |
| :--- | :--- | :--- | :--- | :--- |
| RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic | FP8 | 6.551 / 6.540 / 6.944 | 16 / 7.34 / 8.35 |  |
| mistralai/Mistral-Small-4-119B-2603 | FP8 | 6.480 / 6.479 / 6.690 | one a tensor |  |
| Qwen/Qwen3.6-35B-A3B-FP8 | FP8 | 6.624 / 6.564 / 6.751 | 16 / 7.90 / 8.65 |  |
| Qwen/Qwen3.8-27B-FP8 | FP8 | 6.552 / 6.518 / 6.717 | 16 / 7.90 / 8.78 |  |
| cyankiwi/Qwen3.6-27B-AWQ-INT4 | INT4 | 3.886 / 3.813 / 4.000 | 16 / 10.88 / 11.91 | 2.58 / 2.72 |
| google/gemma-4-12B-it-qat-w4a16-ct | INT4 | 3.769 / 3.759 / 3.997 | 16 / 8.07 / 9.03 |  |
| RedHatAI/Qwen3.8-27B-INT4 | INT4 | 3.429 / 3.419 / 3.575 | 16 / 7.78 / 8.90 |  |
| QuantTrio/Qwen3.6-35B-A3B-AWQ | INT4 | 3.673 / 3.622 / 3.943 | 16 / 10.64 / 11.29 | 1.85 / 2.02 |
| Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 | INT4 | 3.459 / 3.449 / 3.605 | 16 / 10.97 / 11.70 | 0.00 / 1.00 |
| RedHatAI/Qwen3.5-9B-quantized.w8a8 | INT8 | 7.024 / 7.011 / 7.245 | 16 / 7.41 / 8.52 |  |
| moonshotai/Kimi-K3 | MXFP4 | 3.754 / 3.699 / 4.000 | 8 / 0.62 / 1.96 |  |
| deepseek-ai/DeepSeek-V4.1-Flash | FP8 | 6.663 / 6.558 / 6.827 | 8 / 0.93 / 2.00 |  |
| deepseek-ai/DeepSeek-V4.1-Flash | MXFP4 | 3.893 / 3.766 / 4.000 | 8 / 0.99 / 2.00 |  |
| amd/MiniMax-M3-MXFP4 | MXFP4 | 3.721 / 3.689 / 4.000 | 8 / 0.53 / 1.82 |  |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | FP8 | 6.601 / 6.596 / 6.828 | one a tensor |  |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | NVFP4 | 3.968 / 3.947 / 4.000 | 8 / 3.57 / 4.10 |  |
| nvidia/Gemma-4-31B-IT-NVFP4 | NVFP4 | 3.970 / 3.948 / 4.000 | 8 / 3.64 / 4.10 |  |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | FP8 | 6.559 / 6.557 / 6.779 | one a tensor |  |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | NVFP4 | 3.934 / 3.891 / 4.000 | 8 / 3.78 / 4.17 |  |
| nvidia/Qwen3.8-27B-NVFP4 | FP8 | 6.511 / 6.510 / 6.704 | one a tensor |  |
| nvidia/Qwen3.8-27B-NVFP4 | NVFP4 | 3.942 / 3.897 / 4.000 | 8 / 3.99 / 4.28 |  |

Agreement with earlier measurements: Kimi K3's MXFP4 experts, 4 whole layers in `research/lowbit` (lowbit-study branch), were
3.756 bits order-0 and its scales 0.55-0.90 with 2.000 at a fixed rate; here, from 10 samples, 3.754, 0.62 and 1.96. The whole
checkpoint at a fixed rate was 6.3% off there (scales in 2 bits, the bf16 parts at their floor + 0.2 bit) and is 6.5% here (the
bf16 parts at the tiered code's measured 10.75 bits).

Two 16-bit findings, not pursued: amd/MiniMax-M3-MXFP4 keeps attention and dense MLP weights as bf16 that hold few distinct values
(6.5 and 6.8 bits order-0 against about 10.5 for trained bf16; FP8 values stored as bf16, by the look of it): an order-0 code
would take 59% of them, the tiered code only 33%. cyankiwi's AWQ checkpoint keeps its embedding and LM head in fp16 (24.8% of the
file), which the tiered code over fp16's 5-bit exponent takes to 13.7 bits only; Glyd has no fp16 layout.

## The plan: which format next, by VRAM saved on checkpoints people download

1. **The bf16 parts of quantized checkpoints (no new kernel).** 226 GB over these 17 checkpoints. The layout `best_layout` picks
   decides the saving: the tiered one (Ada; 32.6% here, the code without its tile heads) saves 74 GB, the 12-bit one (A10, A100,
   H100, Blackwell; 25% in `gpu/README.md`) 56 GB. By checkpoint, as a share of the whole file: 16-21% of Gemma-4-31B-IT-NVFP4 (its
   attention is bf16), 15-21% of the W8A8 one, 8-14% of the INT4 ones but cyankiwi's (1-2%: its big 16-bit parts are fp16), 4-7% of the NVFP4 MoEs, 2-3% of Kimi K3 and MiniMax-M3. The
   fp16 parts (5.1 GB, cyankiwi's embedding and LM head) are not in these figures: Glyd has no fp16 layout. Glyd's bf16 layouts and
   kernels exist; the work is loading them beside vLLM's quantized methods (a quantized checkpoint's bf16 Linears, embeddings and LM
   head packed while its NVFP4/FP8/INT4 layers stay vLLM's). The biggest shapes are LM heads (248320 x 5120 in the Qwen checkpoints)
   through Glyd's bf16 kernels; it becomes a default only where it is measured at least as fast as vLLM at 1, 8, 32 and 64 sequences
   and uses less memory, and stays opt-in elsewhere. Not measured yet.

| Checkpoint | bf16 parts GB | Tiered (Ada) GB saved | 12-bit (A10, A100, H100, Blackwell) GB saved | fp16 parts GB (no layout) |
| :--- | ---: | ---: | ---: | ---: |
| RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic | 4.1 | 1.4 (4.8%) | 1.0 (3.6%) | 0.0 |
| mistralai/Mistral-Small-4-119B-2603 | 3.0 | 1.0 (0.8%) | 0.8 (0.6%) | 0.0 |
| Qwen/Qwen3.6-35B-A3B-FP8 | 3.0 | 1.0 (2.7%) | 0.7 (2.0%) | 0.0 |
| Qwen/Qwen3.8-27B-FP8 | 6.2 | 2.0 (6.6%) | 1.5 (5.0%) | 0.0 |
| cyankiwi/Qwen3.6-27B-AWQ-INT4 | 1.0 | 0.3 (1.7%) | 0.3 (1.3%) | 5.1 |
| google/gemma-4-12B-it-qat-w4a16-ct | 4.1 | 1.4 (14.1%) | 1.0 (10.1%) | 0.0 |
| RedHatAI/Qwen3.8-27B-INT4 | 6.9 | 2.3 (11.8%) | 1.7 (8.9%) | 0.0 |
| QuantTrio/Qwen3.6-35B-A3B-AWQ | 9.0 | 3.0 (11.7%) | 2.3 (8.9%) | 0.0 |
| Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 | 7.5 | 2.5 (10.1%) | 1.9 (7.7%) | 0.0 |
| RedHatAI/Qwen3.5-9B-quantized.w8a8 | 8.7 | 2.9 (20.7%) | 2.2 (15.5%) | 0.0 |
| moonshotai/Kimi-K3 | 114.4 | 36.6 (2.3%) | 28.6 (1.8%) | 0.0 |
| deepseek-ai/DeepSeek-V4.1-Flash | 4.0 | 1.3 (0.3%) | 1.0 (0.2%) | 0.0 |
| amd/MiniMax-M3-MXFP4 | 21.3 | 7.0 (2.9%) | 5.3 (2.2%) | 0.0 |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | 3.6 | 1.2 (5.1%) | 0.9 (3.9%) | 0.0 |
| nvidia/Gemma-4-31B-IT-NVFP4 | 20.9 | 6.9 (21.2%) | 5.2 (16.0%) | 0.0 |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | 3.7 | 1.2 (5.6%) | 0.9 (4.2%) | 0.0 |
| nvidia/Qwen3.8-27B-NVFP4 | 4.4 | 1.4 (6.6%) | 1.1 (5.0%) | 0.0 |
| all 17 | 225.8 | 73.6 (32.6% of the bf16 parts) | 56.4 (25.0% of them) | 5.1 |

2. **FP8 codes (kernel exists, too slow).** 16.6% of 421 GB of FP8 at the fixed rate as built (65 GB from the codes); from the codes
   13-15% of the two FP8 checkpoints people download most (Qwen3.6-35B-A3B-FP8, 44.8 M downloads, and Qwen3.8-27B-FP8, 11.4 M), 17.5-19.4%
   of those whole checkpoints with their bf16 parts. The pack, unpack and fused `f8_linear` are on private main (off by default) and
   failed "nothing slower": 0.94-2.34x vLLM's FP8 GEMM time on an RTX 4090 and 1.45-5.71x on an H100, geometric means by model and
   tokens a step (`benchmarks/gpu/fp8-linear-2026-10-05/report.md`).
   The next kernel work with the most VRAM behind it is a cheaper FP8 decode, or the memory-only release of the 16% with the
   slower step (the owner's call, already on the table).
3. **MXFP4 and NVFP4 scales (a small kernel change).** Leaving the nibbles as stored and coding only the scales: MXFP4's to 1.8-2.0
   bits (4.1% of Kimi K3 and MiniMax-M3, 3.5% of DeepSeek-V4.1-Flash: 64, 10 and 18 GB), NVFP4's to 4.1-4.3 bits (2-4% of the NVFP4
   checkpoints, 0.5-1.0 GB each). The decode is a scale a block of 16 or 32 values (about 0.25 integer instruction a weight for
   MXFP4's, as counted in `research/lowbit`), inside whatever FP4 GEMM reads the nibbles today. Worth it for the trillion-parameter
   MXFP4 MoEs (Kimi K3 is 1.6% over one 8xB200 node as stored; the lowbit study's fit tables), small for NVFP4.
4. **INT4 codes, symmetric quantizers only.** A fixed rate takes 10% of GPTQ's and RedHatAI's codes (6.5-6.6% of those
   checkpoints, 1.3-1.6 GB), nothing of AWQ/MSE-asymmetric or Google's QAT codes. A per-value tier decode in front of an INT4 GEMM;
   less VRAM than 1-3 for the work.
5. **Entropy-coded FP4 codes and INT8 last.** MXFP4 codes have 3-7% more at the best contexts (Kimi K3's ceiling 14.1% of the
   checkpoint against 6.5% at a fixed rate) but need a prefix-code decode of about 10 instructions a weight: capacity only (a
   trillion-parameter MoE fitting a node). INT8 W8A8: one 2026 checkpoint with 55 k downloads, 9.4% of its codes.

## Method

- **Checkpoints**: 2026 releases (Hub `createdAt` 2026-01 to 2026-09), first-party where the maker publishes the format (NVIDIA's
  NVFP4, Moonshot's and DeepSeek's FP4, Qwen's FP8 and GPTQ, Mistral's FP8, Google's QAT INT4), else the most downloaded 2026
  re-quantization (RedHatAI, cyankiwi, QuantTrio, AMD). Ids and downloads in `results/hub.json`. No token: every repo is public
  and ungated.
- **Census**: every tensor's bytes from the safetensors headers (`hdr.py`, HTTP range reads of the headers, 2 s apart), classified
  by format (the weight's dtype and its scale tensor's dtype and shape) and part: codes, scales, zero points, 16-bit, other.
- **Samples** (`quant.py`, seed 0): up to 10 quantized weight tensors a repo, spread over the (format, attention / MLP / experts / LM
  head / embedding) groups by bytes (each group with 3% of the bytes gets one), and up to 4 16-bit ones (fewer where a repo has fewer groups or tensors: 13 samples in Qwen3.6-35B-A3B-FP8, Qwen3.8-27B-FP8,
  gemma-4-26B-A4B-it-FP8-dynamic, cyankiwi's AWQ, Mistral-Small-4 and Qwen3.8-27B-NVFP4, 12 in gemma-4-12B QAT, 14 elsewhere); each a slice of whole stored rows
  (a multiple of 128, about 12 MB; small expert matrices whole), with the rows of its scales and zero points, read by HTTP range.
  230 samples, 2.33 G values. `engram.embed` alone counts as an embedding (the census and the samples' groups were recomputed so,
  `quant.py --reproject`; the run had filed DeepSeek's `engram.wkv` with it). Unpacked to codes [output rows, input columns]: modelopt and MX nibbles low first, compressed-tensors
  int32 words value k at bits 4k, GPTQ along the input dimension, AWQ with its column order (0, 2, 4, 6, 1, 3, 5, 7); `--selftest`
  checks the unpacking against hand-packed data.
- **Entropies** (`ent.py`, from `research/lowbit`): plug-in with the Miller-Madow correction, a table a tensor (not counted). Contexts
  for codes: the left and upper neighbours, the block's scale (the byte for 8-bit scales, the exponent for 16-bit ones), the group's
  zero point, and for FP4 the block's largest magnitude sent as side information (its cost counted). A context counts only with at
  least 10 values a nonzero cell of its joint histogram. Scales of 16 and 32 bits: sign and exponent modelled (order 0, left,
  above), the mantissa counted as raw bits (their full-width histograms are too sparse to estimate on these samples).
- **Fixed-rate layouts**: only FP8's is built and checked; the others are computed from the samples' histograms and not built
  (their tables not counted, a scale escape with no tile pointer, the 16-bit one without tile heads). (The best a tensor of these, each with a table of the tensor's own and, where a part varies in size, a
  32-bit pointer a 64 x 128 tile): 4-bit codes as stored, a k-bit index into the commonest values with an escape to the value raw,
  or tiers of 1-3-bit digits; FP8: `fp8ref.pack` as built (heads, parts, padding, the shape all counted) and `unpack` checked equal;
  INT8: the high nibble by tiers with the low nibble raw, or sign + bit-length class by tiers + the bits below the leading one;
  scales: a k-bit window a row (its base counted) or a k-bit index into the tensor's commonest, an escape to the value raw, low
  mantissa bits raw; 16-bit weights: 3 tiers of 2-bit digits over the exponent (Glyd's tiered code without its tile heads).
- **Projection**: each (format, part, category) group of the census at the ratio its own samples measured (pooled by values within
  the group), a group with no sample at the ratio of the repo's samples of that format and part, a part with no sample at all (norms,
  biases, per-tensor scalars, `g_idx`) as stored. Weighting by group bytes rather than by values sampled moved two checkpoints by
  about a point (gemma-4-26B-A4B-it-FP8-dynamic 16.1% to 15.2%, Qwen3.6-35B-A3B-FP8 17.0% to 17.5%). Within a group the samples are
  not stratified (gate/up against down, which differ in width in Gemma 4's experts): take each checkpoint's figure as about +-1 point.
- **Not measured**: GGUF and other llama.cpp files, E5M2, KV caches, the decode speed of any layout (only FP8's kernel exists; the
  others' costs are the lowbit study's instruction counts or not estimated).

## Records

```
python hdr.py                                    # results/hub.json, then headers/ + config.json of each repo (headers only; not in git)
python hdr.py hub                                # results/hub.json only
python quant.py --plan headers OUT               # the census and the samples it would read, from the headers alone
python quant.py --selftest && python quant.py headers OUT     # on the pod (fp8ref.py from glyd-gpu-py/glyd_gpu on the fp8 branches beside it)
python quant.py --reproject headers results      # the census and the projection again from the samples' numbers (no tensor read)
python tables.py results                         # the tables above
```

`results/<org>__<name>.json`: census, every sample's numbers (contexts, layouts, scale block), the projection; `results/run.log`
and `results/env.txt` from the pod (a Runpod RTX 2000 Ada pod used as a CPU box, 64 vCPUs, numpy 2.1.2, `fp8ref.py` layout 2 from
fp8-lossless@42e6999f, blob aafce648, which `quant.py` asserts; two runs of 5 and 4
minutes, the second after the sparse-context and 16-bit-scale corrections above; these are the second's). Spend: about $0.04.
