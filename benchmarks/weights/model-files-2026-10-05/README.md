# Model files: glyd --max against zstd -19, xz -9 and v0.28.0 (2026-10-05)

`glyd --max` on safetensors and GGUF files of 2026 models, before and after the model-file mode (`src/weights`),
against zstd -19, zstd -19 --long=27 and xz -9, single-threaded, on this Mac (`machine.txt`). Every glyd decode was compared
byte for byte with its input; so was every zstd and xz decode (`tools/bench.py`).

## The corpus (not committed)

22 files, 3.9 GB, built by `tools/build_corpus.py` from public, ungated Hugging Face repos by HTTP range reads (no token, two
requests a second at most): real tensors of each source, kept in whole rows at their own bytes, in files of their format with the
header rewritten to match (`tools/check_corpus.py` checks each file). A GGUF file keeps its source's key-value section byte for
byte, tokenizer included, so its header (8 to 16 MB) is 4 to 9% of a file of 175 to 217 MB; in the full files it is 0.03%
or less. Each kind of tensor of a source (a name with its layer and expert numbers taken out) gets a share of the file's bytes
in proportion to its share of the source's.

| file | bytes | header bytes | tensors | bytes by type |
| :--- | ---: | ---: | ---: | :--- |
| st-qwen3.8-27b-bf16.safetensors | 134,778,712 | 12,472 | 109 | BF16 134.8 MB |
| st-gemma-4-26b-a4b-bf16.safetensors | 134,515,928 | 13,520 | 105 | BF16 134.5 MB |
| st-glm-5.3-flash-bf16.safetensors | 202,519,968 | 14,720 | 125 | F32 0.1 MB, BF16 202.4 MB |
| st-minimax-m3-bf16.safetensors | 201,551,064 | 14,512 | 109 | BF16 201.4 MB, F32 0.1 MB |
| st-glm-5.3-flash-fp8.safetensors | 99,749,276 | 18,704 | 153 | F32 0.2 MB, BF16 23.7 MB, F8_E4M3 75.9 MB |
| st-qwen3.8-27b-fp8.safetensors | 134,964,976 | 12,720 | 106 | BF16 102.6 MB, F8_E4M3 32.4 MB |
| st-mistral-small-4-fp8.safetensors | 134,577,232 | 20,392 | 154 | BF16 3.7 MB, F8_E4M3 130.8 MB |
| gguf-qwen3.8-27b-bf16.gguf | 212,908,064 | 10,947,104 | 57 | F32 0.9 MB, BF16 201.1 MB |
| gguf-qwen3.8-27b-q8_0.gguf | 181,393,984 | 10,947,264 | 60 | Q8_0 169.6 MB, F32 0.9 MB |
| gguf-qwen3.8-27b-q4_k_m.gguf | 181,285,856 | 10,947,392 | 62 | Q6_K 12.9 MB, F32 0.9 MB, Q4_K 91.5 MB, Q8_0 65.1 MB |
| gguf-gemma-4-26b-a4b-bf16.gguf | 217,406,944 | 15,788,256 | 74 | F32 0.6 MB, BF16 201.1 MB |
| gguf-gemma-4-26b-a4b-q8_0.gguf | 183,875,200 | 15,788,160 | 73 | F32 0.6 MB, Q8_0 167.4 MB |
| gguf-gemma-4-26b-a4b-ud-q4_k_m.gguf | 183,900,544 | 15,788,800 | 72 | F32 0.8 MB, Q8_0 27.8 MB, Q4_K 84.9 MB, Q5_1 54.7 MB |
| gguf-glm-5.3-flash-bf16.gguf | 213,280,480 | 9,437,408 | 124 | BF16 201.8 MB, F32 2.1 MB |
| gguf-glm-5.3-flash-q8_0.gguf | 179,878,752 | 9,437,440 | 121 | Q8_0 168.3 MB, F32 2.1 MB |
| gguf-mistral-small-4-bf16.gguf | 209,315,552 | 7,878,368 | 46 | BF16 201.2 MB, F32 0.2 MB |
| gguf-mistral-small-4-q8_0.gguf | 175,822,400 | 7,869,760 | 43 | Q8_0 167.7 MB, F32 0.3 MB |
| gguf-mistral-small-4-q4_k_m.gguf | 175,779,200 | 7,869,888 | 45 | Q6_K 38.8 MB, F32 0.3 MB, Q4_K 126.2 MB, Q8_0 1.3 MB, Q5_K 1.2 MB |
| gguf-minimax-m3-bf16.gguf | 209,652,608 | 8,246,144 | 52 | BF16 201.1 MB, F32 0.3 MB |
| gguf-minimax-m3-q8_0.gguf | 176,178,656 | 8,246,176 | 49 | Q8_0 167.6 MB, F32 0.3 MB |
| gguf-minimax-m3-q4_k_m.gguf | 176,142,144 | 8,247,008 | 61 | Q6_K 39.3 MB, F32 0.9 MB, Q4_K 125.1 MB, Q8_0 1.4 MB, Q5_K 1.3 MB |
| gguf-gpt-oss-20b-f16.gguf | 180,927,648 | 12,984,224 | 55 | MXFP4 123.6 MB, F32 0.6 MB, F16 43.7 MB |

Sources: Qwen/Qwen3.8-27B (BF16 and FP8, safetensors; ggml-org GGUF BF16, Q8_0, Q4_K_M), google/gemma-4-26B-A4B-it
(safetensors BF16; ggml-org GGUF BF16, Q8_0; unsloth UD-Q4_K_M), zai-org/GLM-5.3-Flash (BF16 and FP8 safetensors; unsloth GGUF BF16,
Q8_0), MiniMaxAI/MiniMax-M3 (safetensors BF16; unsloth GGUF BF16, Q8_0; bartowski Q4_K_M), mistralai/Mistral-Small-4-119B-2603 (FP8
safetensors; unsloth GGUF BF16; bartowski Q8_0, Q4_K_M), unsloth/gpt-oss-20b-GGUF (F16).

## Results

Saved bytes, as a percentage of the file; the "vs" columns in percentage points. Speeds are the process's CPU time (user + sys),
one thread, in MB/s of the original file: `glyd --max -s`, v0.28.0 against this change.

| file | size MB | zstd -19 | zstd -19 --long=27 | xz -9 | glyd --max, v0.28.0 | **glyd --max, now** | vs zstd -19 | vs xz -9 | now: compress MB/s | now: decompress MB/s | v0.28.0: compress MB/s | v0.28.0: decompress MB/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| st-gemma-4-26b-a4b-bf16.safetensors | 134.5 | 24.11% | 23.91% | 29.73% | 32.70% | **34.29%** | +10.18 | +4.56 | 130 | 487 | 115 | 825 |
| st-glm-5.3-flash-bf16.safetensors | 202.5 | 45.90% | 45.94% | 49.35% | 47.12% | **53.78%** | +7.88 | +4.44 | 115 | 439 | 72 | 716 |
| st-glm-5.3-flash-fp8.safetensors | 99.7 | 20.46% | 20.54% | 21.75% | 22.60% | **23.62%** | +3.16 | +1.88 | 216 | 616 | 179 | 1187 |
| st-minimax-m3-bf16.safetensors | 201.6 | 51.46% | 51.38% | 54.07% | 55.78% | **57.45%** | +5.99 | +3.38 | 128 | 488 | 65 | 690 |
| st-mistral-small-4-fp8.safetensors | 134.6 | 17.88% | 18.04% | 18.01% | 18.27% | **19.08%** | +1.20 | +1.08 | 286 | 666 | 189 | 1103 |
| st-qwen3.8-27b-bf16.safetensors | 134.8 | 24.30% | 24.67% | 29.91% | 32.93% | **34.37%** | +10.06 | +4.45 | 130 | 487 | 119 | 827 |
| st-qwen3.8-27b-fp8.safetensors | 135.0 | 22.63% | 22.18% | 26.86% | 29.16% | **30.34%** | +7.71 | +3.48 | 159 | 517 | 125 | 771 |
| gguf-gemma-4-26b-a4b-bf16.gguf | 217.4 | 28.55% | 28.31% | 33.57% | 25.39% | **37.05%** | +8.50 | +3.48 | 132 | 509 | 293 | 1510 |
| gguf-gemma-4-26b-a4b-q8_0.gguf | 183.9 | 10.51% | 10.52% | 9.91% | 9.83% | **13.04%** | +2.53 | +3.13 | 135 | 422 | 601 | 1520 |
| gguf-gemma-4-26b-a4b-ud-q4_k_m.gguf | 183.9 | 8.98% | 8.98% | 9.00% | 7.98% | **12.83%** | +3.86 | +3.84 | 117 | 340 | 654 | 1916 |
| gguf-glm-5.3-flash-bf16.gguf | 213.3 | 49.82% | 49.47% | 53.31% | 41.81% | **57.18%** | +7.37 | +3.88 | 118 | 441 | 148 | 908 |
| gguf-glm-5.3-flash-q8_0.gguf | 179.9 | 13.44% | 13.43% | 13.95% | 11.65% | **28.86%** | +15.42 | +14.90 | 184 | 548 | 454 | 1551 |
| gguf-gpt-oss-20b-f16.gguf | 180.9 | 13.79% | 13.78% | 14.56% | 12.94% | **18.38%** | +4.59 | +3.82 | 219 | 565 | 482 | 1546 |
| gguf-minimax-m3-bf16.gguf | 209.7 | 54.57% | 54.56% | 56.80% | 48.07% | **59.60%** | +5.03 | +2.81 | 128 | 488 | 137 | 794 |
| gguf-minimax-m3-q4_k_m.gguf | 176.1 | 5.62% | 5.59% | 5.41% | 5.11% | **9.13%** | +3.51 | +3.72 | 119 | 335 | 683 | 1835 |
| gguf-minimax-m3-q8_0.gguf | 176.2 | 12.97% | 12.94% | 13.24% | 11.30% | **28.16%** | +15.19 | +14.92 | 190 | 578 | 448 | 1532 |
| gguf-mistral-small-4-bf16.gguf | 209.3 | 54.22% | 54.17% | 56.44% | 48.29% | **59.67%** | +5.44 | +3.23 | 133 | 501 | 134 | 834 |
| gguf-mistral-small-4-q4_k_m.gguf | 175.8 | 5.50% | 5.48% | 5.35% | 4.94% | **8.96%** | +3.46 | +3.61 | 123 | 341 | 546 | 1690 |
| gguf-mistral-small-4-q8_0.gguf | 175.8 | 12.38% | 12.39% | 12.22% | 11.10% | **28.05%** | +15.67 | +15.83 | 191 | 578 | 423 | 1516 |
| gguf-qwen3.8-27b-bf16.gguf | 212.9 | 26.74% | 26.76% | 32.55% | 24.77% | **36.39%** | +9.65 | +3.84 | 138 | 507 | 298 | 1543 |
| gguf-qwen3.8-27b-q4_k_m.gguf | 181.3 | 7.47% | 7.47% | 7.18% | 7.03% | **10.60%** | +3.13 | +3.42 | 129 | 364 | 621 | 1619 |
| gguf-qwen3.8-27b-q8_0.gguf | 181.4 | 8.51% | 8.51% | 7.78% | 8.15% | **11.48%** | +2.98 | +3.70 | 138 | 423 | 553 | 1364 |

```
files 22 total GB 3.900405188 aggregate compress MB/s (CPU) 141.1758067178225 decompress 462.5169201944741
compress MB/s min/max (115.19907167235495, 'st-glm-5.3-flash-bf16.safetensors') (285.7266072186837, 'st-mistral-small-4-fp8.safetensors')
decompress MB/s min/max (334.8709961977186, 'gguf-minimax-m3-q4_k_m.gguf') (666.2239207920792, 'st-mistral-small-4-fp8.safetensors')
whole corpus saved: zstd19 24.82% xz9 26.71% glyd-before 24.19% glyd-now 31.81%
```

**Against zstd -19:** 19 of the 22 files are 3.0 points or more ahead (the range is +1.20 to +15.67). The three that are not:
`st-mistral-small-4-fp8` (+1.20: 97% of its bytes are FP8, whose entropy is about 6.45 of 8 bits, so every coder is within a
few points of the same floor; zstd -19 and xz -9 already hold 17.9% and 18.0% of the 19.4% there is to take),
`gguf-gemma-4-26b-a4b-q8_0` (+2.53) and `gguf-qwen3.8-27b-q8_0` (+2.98): their tensors are 3.6 points ahead of zstd -19 (the next
table), and the tokenizer header these cut-down files keep, 8.6% and 6.0% of their bytes, takes the max level 1.3 and
0.7 MB more than zstd -19 takes (below). **Against xz -9:** ahead on every file (+1.08 to +15.83); 20 of 22 by 2.8 or more.

### The tensors alone

The GGUF headers (tokenizer vocabularies and merge lists) go through the caller's level as they always did, and at `--max` that is
30 to 40% bigger than zstd -19 on this text (`header_sizes.tsv`; glyd --ultra reads 3.34 MB against zstd -19's 3.30 on the Gemma 4
header, in 6.6 s). In a file of tens of gigabytes the header is a few hundredths of a percent, so here are the GGUF files with the
header's bytes taken out of both sides (the file's compressed size less its header compressed alone, over the file's size less
the header):

| GGUF file | header MB | header: zstd -19 / glyd v0.28.0 / xz -9 (MB) | payload only: zstd -19 | payload only: glyd --max now | difference |
| :--- | ---: | :--- | ---: | ---: | ---: |
| gguf-gemma-4-26b-a4b-bf16.gguf | 15.8 | 3.29 / 4.64 / 3.04 | 24.59% | 34.42% | +9.83 |
| gguf-gemma-4-26b-a4b-q8_0.gguf | 15.8 | 3.30 / 4.64 / 3.05 | 4.07% | 7.63% | +3.56 |
| gguf-gemma-4-26b-a4b-ud-q4_k_m.gguf | 15.8 | 3.30 / 4.64 / 3.05 | 2.39% | 7.41% | +5.02 |
| gguf-glm-5.3-flash-bf16.gguf | 9.4 | 1.92 / 2.58 / 1.80 | 48.43% | 56.47% | +8.03 |
| gguf-glm-5.3-flash-q8_0.gguf | 9.4 | 1.92 / 2.58 / 1.80 | 9.77% | 26.43% | +16.66 |
| gguf-gpt-oss-20b-f16.gguf | 13.0 | 2.71 / 3.88 / 2.53 | 8.74% | 14.38% | +5.63 |
| gguf-minimax-m3-bf16.gguf | 8.2 | 2.01 / 2.64 / 1.94 | 53.71% | 59.26% | +5.55 |
| gguf-minimax-m3-q4_k_m.gguf | 8.2 | 2.01 / 2.64 / 1.94 | 2.18% | 6.24% | +4.06 |
| gguf-minimax-m3-q8_0.gguf | 8.2 | 2.01 / 2.64 / 1.94 | 9.90% | 26.21% | +16.31 |
| gguf-mistral-small-4-bf16.gguf | 7.9 | 1.69 / 2.35 / 1.59 | 53.27% | 59.26% | +5.99 |
| gguf-mistral-small-4-q4_k_m.gguf | 7.9 | 1.69 / 2.35 / 1.59 | 2.08% | 6.09% | +4.01 |
| gguf-mistral-small-4-q8_0.gguf | 7.9 | 1.69 / 2.35 / 1.59 | 9.28% | 26.08% | +16.80 |
| gguf-qwen3.8-27b-bf16.gguf | 10.9 | 2.42 / 3.13 / 2.32 | 23.96% | 34.49% | +10.53 |
| gguf-qwen3.8-27b-q4_k_m.gguf | 10.9 | 2.42 / 3.13 / 2.32 | 2.94% | 6.69% | +3.75 |
| gguf-qwen3.8-27b-q8_0.gguf | 10.9 | 2.42 / 3.13 / 2.32 | 4.05% | 7.64% | +3.59 |

## What was not done in the time given

- `--ultra`, `--cold` and brotli -q 11 -w 24 on every file: `baseline.tsv` has them for 8, 6 and 6 of the 22 files (the run was
  stopped to leave the machine to the final measurements); the default, `--fast` and `--turbo` levels, zstd -19 (with and without
  `--long=27`) and xz -9 are all there. The levels above `--max` use the new mode too (they go through the same hook, with their
  own level for the bytes the mode keeps); their numbers after the change are not measured.
- The header, as above: a transform of the GGUF string arrays (lengths apart from text) took 4.64 MB to about 4.0 MB on the Gemma 4
  header in a trial, 0.3 points of that file; a better parse for short strings is what the other 0.7 MB asks for.
- A bf16 tensor that is an FP8 tensor times block scales (GLM-5.3-Flash BF16 is, with a little noise on the small values) could
  be coded as the FP8 tensor and its scales: an estimate from one expert tensor and its FP8 twin puts that at about 6.7 bits an
  element, against 7.0 to 7.4 for the file now. Not built.

## Reproduce

```
CODECW_SCRATCH=. python tools/build_corpus.py build           # needs gguf/gguflib.py beside it (a GGUF header reader over range reads)
python tools/check_corpus.py
GLYD=./glyd python tools/bench.py after.tsv 2 glyd-max corpus/*.safetensors corpus/*.gguf
GLYD=./glyd-v0.28.0 python tools/bench.py baseline.tsv 4 zstd19,zstd19long,xz9,glyd,glyd-fast,glyd-turbo,glyd-max corpus/*
ls corpus/*.gguf | xargs -P 4 -n 1 tools/header_sizes.sh > header_sizes.tsv
python tools/make_table.py baseline.tsv after.tsv header_sizes.tsv
GLYD_WEIGHTS_CORPUS=corpus cargo test --release --test weights the_corpus_comes_back     # every file back, five ways
```
