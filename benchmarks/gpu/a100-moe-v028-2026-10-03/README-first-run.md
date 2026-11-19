# Qwen3-16B-A3B on an A100: MoE serving against bf16, and the layer sweep of Qwen3-30B-A3B's layer (2026-10-03)

*The first run's README, kept as it was (its numbers are the tree before the epilogue change; `moea100b/`). The final numbers are `README.md`, from the second run (`moea100c/`).*

One NVIDIA A100-SXM4-40GB (compute capability 8.0, 108 SMs, 40 GB, 400 W; driver 580.126.20; 30 CPUs; Lambda `gpu_1x_a100_sxm4`), vLLM 0.30.0, torch 2.13 with CUDA 13.0.
The tree is moe-a100 9267523e on private/main ddaefb8a (v0.27.0 and the MoE route of the H100's record: nothing in it is the A100's own but the layer's loader and a development value for the attention products). The job is `job/moe_job.sh`, run once, with the A100's variants.

**The serve model is `kalomaze/Qwen3-16B-A3B`**: Qwen3-30B-A3B with half its experts (64 of 128 a layer pruned by their routing statistics; Apache-2.0): the same 48 layers, hidden 2,048, intermediate 768, 8 experts a token and the same attention, 16.03B parameters, 29.87 GiB in bf16. Qwen3-30B-A3B itself (57 GiB) does not fit this GPU in bf16. The model is not the H100 record's: its experts are 64 a layer, so a step's tokens give each expert twice as many pairs.
**The layer sweep is on Qwen3-30B-A3B's own layer 0** (128 experts), as the H100's record has it.
**The baseline is vLLM's own bf16 path with its default, untuned fused-MoE configuration**: its log says `Using default MoE config. Performance might be sub-optimal!` and names the file it did not find, `E=64,N=768,device_name=NVIDIA_A100-SXM4-40GB.json` (`moea100b/serve/bf16/serve-bf16.txt.gz`). A tuned bf16 would be faster than the baseline here; that was not measured. Name the baseline wherever these numbers are quoted.

## What to know

- **The run is comparable** (the job's guard, `moea100b/serve/clocks.txt`): every saturated pass at a median SM clock of 1,410 MHz (the A100's top; the floor the guard asks is 1,213), at most 61 C, no thermal slowdown from the driver, and the two bf16 passes, the first and the last of four, 0.5% apart (4.02 and 4.00 requests a second).
- **Saturated, 256 requests at once, 1,024 tokens in and 256 out**: bf16 4.02 requests a second, **Glyd with the lossless KV cache 5.32 (1.32x)**, **and with the attention products left bf16 5.44 (1.35x)**. The first token comes after 27,299, 16,435 and 15,933 ms (0.60x and 0.58x); each token takes 31.8, 72.0 and 67.5 ms (2.26x and 2.12x as long). The gain is room: the weights take 29.87, 23.59 and 23.97 GiB, the KV cache holds 40,288, 140,336 and 138,336 tokens (3.48x and 3.43x), and a decode step carries 32, 100 and 95 sequences on average. A request takes Glyd 2.1 to 2.3 times as long once it runs, which the layer sweep below explains.
- **One request a second** (64 requests): the same 0.94 to 0.95 requests a second, each token 14.3, 12.5 and 12.3 ms (0.87x and 0.86x), but **the first token 102, 125 and 119 ms: 1.23x and 1.17x later**. A prompt of 1,024 tokens is one step of about 1,030, where the layer sweep has Glyd's grouped products at 1.85x to 2.0x bf16's time.
- **Every check passes**: 101 of 101 (the real layer against an fp32 product at 1 to 1,152 tokens a step, the library's variants, uneven loads, a layer with exceptions by the dozen).
- **The server of an A100 cuts a step at 2,048 tokens** (vLLM's default for this GPU; the H100's is 8,192), so no prompt step here has more: they take 15.9 s of bf16's 63.3 s saturated pass and 23.1 to 24.2 s of Glyd's 46.5 to 47.6 s, 52 us a prompt token against 76 to 80 (1.46x to 1.54x), and the decode steps 784 us a token against 465 to 467.
- **The layer sweep, Glyd's grouped products against vLLM's bf16 layer, built-in settings** (tables below): 0.80x at 16 tokens, 1.1x to 1.5x from 24 to 512 (where the rows kernel, off by default on this GPU, takes 11% to 24% off: its variants `rows17`, `rows33` and `rows65` are 1.0x to 1.2x bf16's), 1.85x at 768, **2.01x at 1,152**, 1.83x at 2,048, 1.96x at 4,096 and 2.02x at 8,192 tokens; the decoded route (the experts unpacked for vLLM's kernel, what the plugin takes from 1,152 tokens on this GPU) takes 2.48x, 2.18x, 1.75x and 1.44x at 1,152, 2,048, 4,096 and 8,192.
- **The tiled GEMM runs the gate and up at 25% of the A100's peak at 8,192 tokens (77 TFLOP/s of 312) and the down at 19% (60), against bf16's 45% (139)**; it is also what takes over from 768 tokens (48 pairs an expert): for the gate and up the other kernels are the faster up to about 1,152 (1,352 us against 1,647 at 1,152), for the down the tiled one is the faster of the two measured (753 against 1,315 us at 1,152; the rows kernel there is not measured).

## Serving, each mode against bf16 (`vllm bench serve`, random dataset, 1,024 tokens in, 256 out)

`glyd-picked+kv` is Glyd's weights with the lossless KV cache (its tables for the model made at the first start, 111 s), `glyd-picked+kv+bf16-attn` the same with the attention products left bf16 (`GLYD_MOE_DENSE=bf16`), `bf16-2` the bf16 server started again at the end.

| Rate (req/s) | Mode | Requests/s | vs bf16 | First token (ms) | vs bf16 | Each token (ms) | vs bf16 | SM clock, median (MHz) | Highest temperature (C) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 0.94 | 1.00x | 102 | 1.00x | 14.3 | 1.00x | 1410 | 50 |
| 1 | glyd-picked+kv | 0.95 | 1.01x | 125 | 1.23x | 12.5 | 0.87x | 1410 | 49 |
| 1 | glyd-picked+kv+bf16-attn | 0.95 | 1.01x | 119 | 1.17x | 12.3 | 0.86x | 1410 | 48 |
| 1 | bf16-2 | 0.94 | 1.00x | 102 | 1.00x | 14.6 | 1.02x | 1410 | 50 |
| inf | bf16 | 4.02 | 1.00x | 27299 | 1.00x | 31.8 | 1.00x | 1410 | 61 |
| inf | glyd-picked+kv | 5.32 | 1.32x | 16435 | 0.60x | 72.0 | 2.26x | 1410 | 58 |
| inf | glyd-picked+kv+bf16-attn | 5.44 | 1.35x | 15933 | 0.58x | 67.5 | 2.12x | 1410 | 58 |
| inf | bf16-2 | 4.00 | 1.00x | 27477 | 1.01x | 31.8 | 1.00x | 1410 | 61 |

## The clocks of each pass (COMPARABLE: bf16 1410 MHz, 61 C; bf16-2 1410 MHz, 61 C; glyd-picked+kv 1410 MHz, 58 C; glyd-picked+kv+bf16-attn 1410 MHz, 58 C)

| Mode | Pass | Requests/s | SM clock median (MHz) | lowest | Highest temperature (C) | Power median (W) | Seconds |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 1 |  | 1410 | 1095 | 50 | 207 | 92 |
| bf16 | inf | 4.02 | 1410 | 1395 | 61 | 272 | 63 |
| bf16-2 | 1 |  | 1410 | 1095 | 50 | 206 | 90 |
| bf16-2 | inf | 4.00 | 1410 | 1395 | 61 | 274 | 63 |
| glyd-picked+kv | 1 |  | 1410 | 1095 | 49 | 195 | 89 |
| glyd-picked+kv | inf | 5.32 | 1410 | 1410 | 58 | 267 | 47 |
| glyd-picked+kv+bf16-attn | 1 |  | 1410 | 1095 | 48 | 203 | 90 |
| glyd-picked+kv+bf16-attn | inf | 5.44 | 1410 | 1410 | 58 | 264 | 46 |

## Where the saturated pass's time went (the servers' iteration lines)

This table is from the fixed `job/sweep_summary.py`. The run's own `moea100b/summary.txt` has a wrong row for `glyd-picked+kv`: its window began in the rate pass before it, where two prompts of 1,024 tokens met in one step, and counted 4,610 steps and 120 s (and 71 s in the clock table) for the saturated pass, which has 786 and 47.6. The ratios of the serving table above are the benchmark's own and stand.

| Mode | Steps | Seconds | Prompt steps | s | ms a step | us a token | Decode steps | s | ms a step | us a token | Decode steps' mean sequences | Highest KV usage |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 2077 | 63.3 | 174 | 15.9 | 91 | 52 | 1903 | 47.4 | 24.9 | 784 | 32 | 100% |
| glyd-picked+kv | 786 | 47.6 | 164 | 24.2 | 147 | 80 | 622 | 23.4 | 37.7 | 467 | 100 | 100% |
| glyd-picked+kv+bf16-attn | 790 | 46.5 | 165 | 23.1 | 140 | 76 | 625 | 23.4 | 37.5 | 465 | 95 | 100% |
| bf16-2 | 2077 | 63.5 | 173 | 15.9 | 92 | 52 | 1904 | 47.7 | 25.0 | 789 | 32 | 100% |

## The experts' layer by tokens a step, GPU time in a CUDA graph (Qwen3-30B-A3B's layer 0, microseconds)

vLLM's own bf16 layer (its default fused-MoE configuration) against the built-in settings (`new`) and, in the second table, each variant of them: `rowsN` the rows kernel (the down product, a warp a row block over the whole K) from N tokens a step, `bigN` the tiled GEMM from N pairs an expert, `na1` the m-tiles with no pairs left out, `pb1` the exceptions of a step loaded together, `pf4` the L2 prefetch.
The last column of the first table is the plugin's decoded route.

| Tokens a step | bf16 (vLLM's layer) | v027 | new | best variant | best / bf16 | the plugin's decoded route* / bf16 |
| ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| 16 | 617 us | - | 493 us | rows65 463 us | 0.75x | 2.91x |
| 24 | 694 us | - | 784 us | rows17 685 us | 0.99x | 3.08x |
| 32 | 762 us | - | 857 us | rows17 763 us | 1.00x | 3.07x |
| 48 | 830 us | - | 1,194 us | rows33 967 us | 1.17x | 3.06x |
| 64 | 854 us | - | 1,222 us | rows33 982 us | 1.15x | 3.05x |
| 128 | 919 us | - | 1,285 us | rows65 1,022 us | 1.11x | 2.93x |
| 232 | 946 us | - | 1,341 us | rows33 1,053 us | 1.11x | 2.88x |
| 384 | 976 us | - | 1,426 us | rows17 1,101 us | 1.13x | 2.83x |
| 512 | 997 us | - | 1,501 us | rows65 1,143 us | 1.15x | 2.79x |
| 768 | 1,157 us | - | 2,135 us | big96 1,658 us | 1.43x | 2.55x |
| 1,152 | 1,214 us | - | 2,440 us | big24 2,440 us | 2.01x | 2.48x |
| 2,048 | 1,540 us | - | 2,815 us | pf4 2,815 us | 1.83x | 2.18x |
| 4,096 | 2,482 us | - | 4,865 us | new 4,865 us | 1.96x | 1.75x |
| 8,192 | 4,450 us | - | 8,983 us | rows65 8,980 us | 2.02x | 1.44x |

What the built-in settings spend, and the rates they reach (9.44 MFLOP a pair; the A100's bf16 peak is 312 TFLOP/s):

| Tokens a step | bf16 us | Glyd us | gate+up us (TFLOP/s) | down us (TFLOP/s) | route us | bf16 TFLOP/s | Glyd us a token | bf16 us a token | unpacked for vLLM's kernel us |
| ---: | ---: | ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 16 | 617 | 493 | 301 (3) | 185 (2) | 7 | 2 | 30.83 | 38.59 | 1,174 |
| 24 | 694 | 784 | 451 (3) | 326 (2) | 7 | 3 | 32.65 | 28.92 | 1,436 |
| 32 | 762 | 857 | 491 (3) | 359 (2) | 8 | 3 | 26.79 | 23.80 | 1,572 |
| 48 | 830 | 1,194 | 652 (4) | 531 (2) | 11 | 4 | 24.87 | 17.28 | 1,697 |
| 64 | 854 | 1,222 | 665 (5) | 549 (3) | 8 | 6 | 19.10 | 13.35 | 1,740 |
| 128 | 919 | 1,285 | 693 (9) | 583 (6) | 10 | 11 | 10.04 | 7.18 | 1,765 |
| 232 | 946 | 1,341 | 707 (17) | 620 (9) | 12 | 19 | 5.78 | 4.08 | 1,767 |
| 384 | 976 | 1,426 | 730 (26) | 680 (14) | 16 | 30 | 3.71 | 2.54 | 1,767 |
| 512 | 997 | 1,501 | 750 (34) | 731 (18) | 19 | 39 | 2.93 | 1.95 | 1,768 |
| 768 | 1,157 | 2,135 | 1,423 (27) | 677 (29) | 24 | 50 | 2.78 | 1.51 | 1,768 |
| 1,152 | 1,214 | 2,440 | 1,647 (35) | 753 (38) | 34 | 72 | 2.12 | 1.05 | 1,767 |
| 2,048 | 1,540 | 2,815 | 1,693 (61) | 1,059 (49) | 54 | 100 | 1.37 | 0.75 | 1,767 |
| 4,096 | 2,482 | 4,865 | 2,897 (71) | 1,855 (56) | 101 | 125 | 1.19 | 0.61 | 1,768 |
| 8,192 | 4,450 | 8,983 | 5,327 (77) | 3,437 (60) | 196 | 139 | 1.10 | 0.54 | 1,768 |

All three together (route, gate+up, down with the sum) by variant, microseconds:

| Tokens | new | big24 | big96 | na1 | pb1 | pf4 | rows17 | rows33 | rows65 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 493 | 463 (0.94) | 464 (0.94) | 463 (0.94) | 463 (0.94) | 487 (0.99) | 464 (0.94) | 464 (0.94) | 463 (0.94) |
| 24 | 784 | 784 (1.00) | 784 (1.00) | 759 (0.97) | 784 (1.00) | 753 (0.96) | 685 (0.87) | 784 (1.00) | 784 (1.00) |
| 32 | 857 | 858 (1.00) | 858 (1.00) | 832 (0.97) | 857 (1.00) | 826 (0.96) | 763 (0.89) | 858 (1.00) | 858 (1.00) |
| 48 | 1,194 | 1,194 (1.00) | 1,194 (1.00) | 1,090 (0.91) | 1,194 (1.00) | 1,168 (0.98) | 967 (0.81) | 967 (0.81) | 1,194 (1.00) |
| 64 | 1,222 | 1,222 (1.00) | 1,222 (1.00) | 1,116 (0.91) | 1,222 (1.00) | 1,201 (0.98) | 982 (0.80) | 982 (0.80) | 1,222 (1.00) |
| 128 | 1,285 | 1,286 (1.00) | 1,286 (1.00) | 1,168 (0.91) | 1,285 (1.00) | 1,284 (1.00) | 1,022 (0.79) | 1,022 (0.80) | 1,022 (0.79) |
| 232 | 1,341 | 1,340 (1.00) | 1,340 (1.00) | 1,235 (0.92) | 1,340 (1.00) | 1,361 (1.02) | 1,053 (0.79) | 1,053 (0.79) | 1,053 (0.79) |
| 384 | 1,426 | 1,622 (1.14) | 1,426 (1.00) | 1,347 (0.94) | 1,426 (1.00) | 1,460 (1.02) | 1,101 (0.77) | 1,101 (0.77) | 1,101 (0.77) |
| 512 | 1,501 | 1,813 (1.21) | 1,501 (1.00) | 1,451 (0.97) | 1,501 (1.00) | 1,545 (1.03) | 1,144 (0.76) | 1,144 (0.76) | 1,143 (0.76) |
| 768 | 2,135 | 2,133 (1.00) | 1,658 (0.78) | 2,132 (1.00) | 2,134 (1.00) | 2,134 (1.00) | 2,134 (1.00) | 2,133 (1.00) | 2,133 (1.00) |
| 1,152 | 2,440 | 2,440 (1.00) | 2,704 (1.11) | 2,441 (1.00) | 2,440 (1.00) | 2,440 (1.00) | 2,441 (1.00) | 2,440 (1.00) | 2,441 (1.00) |
| 2,048 | 2,815 | 2,817 (1.00) | 2,817 (1.00) | 2,816 (1.00) | 2,816 (1.00) | 2,815 (1.00) | 2,817 (1.00) | 2,815 (1.00) | 2,817 (1.00) |
| 4,096 | 4,865 | 4,867 (1.00) | 4,867 (1.00) | 4,867 (1.00) | 4,866 (1.00) | 4,867 (1.00) | 4,866 (1.00) | 4,865 (1.00) | 4,868 (1.00) |
| 8,192 | 8,983 | 8,983 (1.00) | 8,984 (1.00) | 8,984 (1.00) | 8,985 (1.00) | 8,983 (1.00) | 8,983 (1.00) | 8,982 (1.00) | 8,980 (1.00) |

Gate+up and down apart, around the tiled GEMM's threshold (768 tokens is 48 pairs an expert, the settings' own):

| Tokens | variant | gate+up us | down us |
| ---: | :--- | ---: | ---: |
| 384 | new | 730 | 680 |
| 384 | rows17 | 730 | 355 |
| 384 | big96 | 730 | 679 |
| 384 | big24 | 1,018 | 576 |
| 512 | new | 750 | 731 |
| 512 | rows17 | 750 | 374 |
| 512 | big96 | 750 | 731 |
| 512 | big24 | 1,166 | 610 |
| 768 | new | 1,423 | 677 |
| 768 | rows17 | 1,426 | 677 |
| 768 | big96 | 797 | 835 |
| 768 | big24 | 1,426 | 677 |
| 1,152 | new | 1,647 | 753 |
| 1,152 | rows17 | 1,647 | 753 |
| 1,152 | big96 | 1,352 | 1,315 |
| 1,152 | big24 | 1,646 | 753 |
| 2,048 | new | 1,693 | 1,059 |
| 2,048 | rows17 | 1,694 | 1,059 |
| 2,048 | big96 | 1,692 | 1,059 |
| 2,048 | big24 | 1,695 | 1,059 |

## Not measured here

- The job's dense part (22 s were left of its 30 minutes), so the attention products' routes on this GPU are not in the record; the serve above has the A/B of their being packed or not.
- A tuned bf16 baseline; an A100 of 80 GB, where Qwen3-30B-A3B itself runs; tensor parallel; other prompt and output lengths.
- The tiled GEMM at 1,152 to 2,048 tokens with the gate and up on the other kernels, and the rows kernel for the down at 768 tokens and over (the tiled GEMM takes those first).

## After this record (same day)

The A100's two open numbers (first token at low load, each token when saturated) ride on the tiled GEMM, and on an L4, which runs it as it is, its epilogue was about two fifths of a tile's time (`l4-tiled/`): shrunk, it takes 12 to 25% off the kernel's products on a compute-bound layer and 6 to 11% off the real layer at its DRAM floor. The tree after this record (`gpu/glyd_gpu.cu`, commit 7820a5ea) also has what this run's sweep measured built in: the tiled GEMM from 84 pairs an expert for the gate and up and 48 for the down, the rows kernel from 17 tokens, the decode kernels' m-tile skipping (`na1`), and m-tiles with no pairs skipped in the tiled kernel; the plugin keeps an A100's attention products bf16 and takes the decoded route from 3,072 tokens. The second run (`job-next/moe_job.sh`, `moea100c/`) measured it on an A100: `README.md`. The numbers above are this run's, as they were.

## Files

- `l4-tiled/`: three captures of the tiled kernel on an L4 and the epilogue change (README there).
- `job-next/`: the job for the tree after this record (`moe_job.sh`, its tables' scripts, `make_tar.sh` for its source tar).

- `job/`: the job as run (`moe_job.sh`, `bash ~/moe_job.sh` with its source tar, the tree's `gpu/` and `glyd-gpu-py/` at 9267523e and `job/`) and the scripts of its tables (`sweep_summary.py`: the fixed one; `tile_verdict.py`, `ncu_summary.py`).
- `moea100b/summary.txt`: the job's summary as it came; `steps.txt` its steps with times; `machine.txt`, `env.txt`, `COMMIT`, `job.log`.
- `moea100b/serve/`: a directory a mode with each pass's `vllm bench serve` output and result, the server's log (`serve-*.txt.gz`: gunzip it for `job/sweep_summary.py steps`), the nvidia-smi samples of each pass and of the whole server start (`smi-reasons.csv`); `clocks.txt`, `steps.txt` and `kv-notes.txt` beside them (as the run made them).
- `moea100b/check/check.txt`: the 101 checks; `moea100b/sweep/`: the layer sweep (`layer.json`, `layer.txt`, `summary.txt`); `moea100b/log/`: the downloads, the environment and the thermal preflight (43 s of matmuls on the idle GPU: 64 C, 369 W, no thermal slowdown).
