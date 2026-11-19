# Qwen3-16B-A3B on an A100: MoE serving against bf16, and the layer sweep of Qwen3-30B-A3B's layer (2026-10-03, the final numbers)

One NVIDIA A100-SXM4-40GB (compute capability 8.0, 108 SMs, 40 GB, 400 W; driver 580.126.20; 30 CPUs, AMD EPYC 7J13; Lambda `gpu_1x_a100_sxm4`), vLLM 0.30.0, torch 2.13 with CUDA 13.0: the machine type of the first run.
These are the **second run's** numbers (`moea100c/`). The tree is moe-a100 48ab4599 on private/main ddaefb8a (merged into private main with PR #5, 5b539d0c): the first run's tree with the tiled MoE kernel's epilogues shrunk and its empty m-tiles skipped, an A100's dispatch per product and the decode kernels' m-tile skipping built in (`../l4-tiled/`). The job is `job-next/moe_job.sh` (sha256 88c63e57746a208485a9b5948ae61998f7da34a4a725d37329f7deb200e5e9e8) with its source tar made by `job-next/make_tar.sh` at 48ab4599 (sha256 b235cd1d1749da21561e59e548c6876756b61b04e56459b29f0f8beb57ebe0a8), run once, 1,410 s, all five parts (check, serve, sweep, Nsight, dense). **The first run** (`moea100b/`, `README-first-run.md`, the tree before the epilogue change) is kept as history: its numbers are compared below and its README is as it was.

**The serve model is `kalomaze/Qwen3-16B-A3B`**: Qwen3-30B-A3B with half its experts (64 of 128 a layer pruned by their routing statistics; Apache-2.0): the same 48 layers, hidden 2,048, intermediate 768, 8 experts a token and the same attention, 16.03B parameters, 29.87 GiB in bf16. Qwen3-30B-A3B itself (57 GiB) does not fit this GPU in bf16. The model is not the H100 record's: its experts are 64 a layer, so a step's tokens give each expert twice as many pairs.
**The layer sweep is on Qwen3-30B-A3B's own layer 0** (128 experts), as the H100's record has it.
**The baseline is vLLM's own bf16 path with its default, untuned fused-MoE configuration**: its log says `Using default MoE config. Performance might be sub-optimal!` and names the file it did not find, `E=64,N=768,device_name=NVIDIA_A100-SXM4-40GB.json` (`moea100c/serve/bf16/serve-bf16.txt.gz`). A tuned bf16 would be faster than the baseline here; that was not measured. Name the baseline wherever these numbers are quoted.

## What to know

- **The run is comparable** (the job's guard, `moea100c/serve/clocks.txt`): every saturated pass at a median SM clock of 1,410 MHz (the A100's top; the floor the guard asks is 1,213), at most 64 C, no slowdown from the driver in any server's 324 to 425 s, and the two bf16 passes, the first and the last of three, 4.4% apart (4.12 and 3.94 requests a second; the guard's limit is 5%).
- **Saturated, 256 requests at once, 1,024 tokens in and 256 out**: bf16 4.12 requests a second (3.94 the second time), **Glyd with the lossless KV cache 6.18 (1.50x; 1.57x against the second bf16 pass)**. The first token comes after 27,100 and 14,233 ms (0.53x); each token takes 30.9 and 62.9 ms (2.04x as long). The gain is room: the weights take 29.87 and 23.97 GiB, the KV cache holds 40,288 and 144,896 tokens (3.60x), and a decode step carries 32 and 103 sequences on average. Glyd's attention products are bf16 (the A100's default since ruling 1; the weights, 23.97 GiB, are the first run's bf16-attention mode's).
- **One request a second** (64 requests): 0.94 and 0.95 requests a second, each token 13.9 and 12.3 ms (0.88x), and **the first token 102 and 110 ms: 1.08x later** (p99 401 and 175 ms). A prompt of 1,024 tokens is one step of about 1,030, which for this model's 64 experts is as many pairs an expert as 2,048 tokens of Qwen3-30B-A3B's layer, 1.58x bf16's time in the sweep below.
- **Every check passes**: 162 of 162 (the real layer against an fp32 product at 1 to 4,096 tokens a step in 13 settings, uneven loads, a layer with exceptions by the dozen).
- **The server of an A100 cuts a step at 2,048 tokens** (vLLM's default for this GPU; the H100's is 8,192), so no prompt step here has more: they take 15.7 s of bf16's 61.6 s saturated pass and 22.0 s of Glyd's 40.9 s (133 ms a step, 72 us a prompt token against 52: 1.38x), and Glyd's decode steps 30.5 ms with 103 sequences, 380 us a token against bf16's 758.
- **The layer sweep, Glyd's grouped products against vLLM's bf16 layer, built-in settings** (tables below): 0.75x at 16 tokens, 0.96x to 1.12x from 32 to 512 (1.01x at 232), 1.22x at 768, **1.67x at 1,152**, 1.40x at 1,536, 1.58x at 2,048, 1.71x, 1.70x and 1.75x at 3,072, 4,096 and 8,192 tokens; the first run's 1.42x at 232 and 1.83x to 2.02x from 768 on. The decoded route (the experts unpacked for vLLM's kernel) takes 2.49x, 2.18x, 1.92x, 1.75x and 1.44x at 1,152, 2,048, 3,072, 4,096 and 8,192.
- **The code change alone** (the variant `old`: the first run's settings, the rows kernel off, the tiled GEMM from 48 pairs an expert, no m-tile skipped, with this tree's kernels, against the first run's own `new`, the same settings): the layer -23% at 768, -29% at 1,152, -9% at 2,048, -11% at 4,096 and -12% at 8,192 tokens (the gate and up -12% to -15% from 2,048 and -31% to -39% at 768-1,152, the down -5% to -7%), nothing at 512 and below, where the decode kernels run. Skipping the empty m-tiles takes 1% to 5% off from 768 (`noskip`). The built-in dispatch against `old`: -15% to -28% from 32 to 512 tokens, -14% at 768, **+16% at 1,152**, -1% to -5% above.
- **The tiled GEMM** runs the gate and up at 93 TFLOP/s at 8,192 tokens (30% of the A100's 312; 25% in the first run) and the down at 64 (21%; 19%), against bf16's 139 (45%); with a tile of 256 tokens 121 and 75 (39% and 24%).
- **The variants** (second table below) say the thresholds built in from the first run (the tiled gate and up from 84 pairs an expert, the down from 48) are now off, the tiled kernel being faster: the tiled gate and up from 64 pairs (`gu64`) takes 16% off at 1,152 tokens, the rows kernel for the down (`dn96`) 11% off at 768 and costs 4% at 1,152; 64 for both would give each (1.09x and 1.40x bf16, from the variants' parts, not run together). A tile of 256 tokens (`tm256`) takes 22%, 13% and 19% off at 3,072, 4,096 and 8,192 tokens (1.34x, 1.48x and 1.43x bf16, ahead of the decoded route) and costs 1% to 32% from 768 to 2,048; three stages (`nb3`) are worse than two. Neither is built in; the A100's server does not reach them by default.
- **Nsight Compute on the A100's own tiled kernel** (below): 25.5% of the tensor peak, 28% of the issue slots, 25% of DRAM; the producers wait for the consumers and are done at about 60% of a consumer warp's life, and the epilogue is still about a quarter of it.
- **The dense part ran this time** (below): Glyd's best route for the attention products is slower than cuBLAS at every size but one, which is the case for leaving them bf16 on this GPU.

## Serving, each mode against bf16 (`vllm bench serve`, random dataset, 1,024 tokens in, 256 out)

`glyd-picked+kv` is Glyd's weights with the lossless KV cache and the built-in settings (its tables for the model made at the first start, 107 s), `bf16-2` the bf16 server started again at the end, to show the machine's drift. 64 prompts at 1 a second, then 256 at once.

| Rate (req/s) | Mode | Requests/s | vs bf16 | First token (ms) | vs bf16 | Each token (ms) | vs bf16 | SM clock, median (MHz) | Highest temperature (C) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 0.94 | 1.00x | 102 | 1.00x | 13.9 | 1.00x | 1410 | 49 |
| 1 | glyd-picked+kv | 0.95 | 1.01x | 110 | 1.08x | 12.3 | 0.88x | 1410 | 50 |
| 1 | bf16-2 | 0.94 | 1.00x | 103 | 1.01x | 14.1 | 1.01x | 1410 | 50 |
| inf | bf16 | 4.12 | 1.00x | 27100 | 1.00x | 30.9 | 1.00x | 1410 | 64 |
| inf | glyd-picked+kv | 6.18 | 1.50x | 14233 | 0.53x | 62.9 | 2.04x | 1410 | 56 |
| inf | bf16-2 | 3.94 | 0.96x | 28024 | 1.03x | 32.2 | 1.04x | 1410 | 64 |

## The clocks of each pass (COMPARABLE: bf16 1410 MHz, 64 C; bf16-2 1410 MHz, 64 C; glyd-picked+kv 1410 MHz, 56 C)

| Mode | Pass | Requests/s | SM clock median (MHz) | lowest | Highest temperature (C) | Power median (W) | Seconds |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 1 |  | 1410 | 1095 | 49 | 200 | 92 |
| bf16 | inf | 4.12 | 1410 | 1410 | 64 | 264 | 61 |
| bf16-2 | 1 |  | 1410 | 1095 | 50 | 200 | 90 |
| bf16-2 | inf | 3.94 | 1410 | 1410 | 64 | 265 | 65 |
| glyd-picked+kv | 1 |  | 1410 | 1095 | 50 | 196 | 89 |
| glyd-picked+kv | inf | 6.18 | 1410 | 1410 | 56 | 284 | 41 |

bf16-2: the driver's slowdown reasons, seconds active of 324: hardware thermal slowdown 0, software thermal slowdown 0, software power cap 0, hardware power brake 0
bf16: the driver's slowdown reasons, seconds active of 351: hardware thermal slowdown 0, software thermal slowdown 0, software power cap 0, hardware power brake 0
glyd-picked+kv: the driver's slowdown reasons, seconds active of 425: hardware thermal slowdown 0, software thermal slowdown 0, software power cap 0, hardware power brake 0

## Where the saturated pass's time went (the servers' iteration lines)

| Mode | Steps | Seconds | Prompt steps | s | ms a step | us a token | Decode steps | s | ms a step | us a token | Decode steps' mean sequences | Highest KV usage |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 2076 | 61.6 | 171 | 15.7 | 92 | 52 | 1905 | 45.9 | 24.1 | 758 | 32 | 100% |
| glyd-picked+kv | 784 | 40.9 | 166 | 22.0 | 133 | 72 | 618 | 18.9 | 30.5 | 380 | 103 | 100% |
| bf16-2 | 2077 | 64.6 | 173 | 16.0 | 92 | 52 | 1904 | 48.7 | 25.6 | 805 | 32 | 100% |

| Mode | Steps | Count | Seconds | Mean |
| :--- | :--- | ---: | ---: | ---: |
| bf16 | decode steps of <64 sequences | 1905 | 45.9 s | 24.1 ms a step |
| glyd-picked+kv | decode steps of <64 sequences | 230 | 3.3 s | 14.3 ms a step |
| glyd-picked+kv | decode steps of 64-127 sequences | 273 | 10.9 s | 39.9 ms a step |
| glyd-picked+kv | decode steps of 128-191 sequences | 115 | 4.7 s | 40.9 ms a step |
| bf16-2 | decode steps of <64 sequences | 1904 | 48.7 s | 25.6 ms a step |

## The experts' layer by tokens a step, GPU time in a CUDA graph (Qwen3-30B-A3B's layer 0, microseconds)

vLLM's own bf16 layer (its default fused-MoE configuration) against the built-in settings (`new`) and the first run's, and in the later tables each variant of the built-in settings: new: the built-in settings (the rows kernel from 17 tokens, the tiled GEMM from 84 pairs an expert for the gate and up and 48 for the down, in tiles of 128 tokens, empty m-tiles skipped, the decode kernels' too); old: the settings of the job before this one (the rows kernel off, the tiled GEMM from 48 for both, every m-tile multiplied); the built-in epilogue slimming is in every column; tm256: tiles of 256 tokens (eight consumer warps, 2 stages); nb3: the same with 3 stages; noskip: every m-tile multiplied in the tiled kernel; guN, dnN: the tiled GEMM from N pairs an expert for the gate and up, for the down. The last column of the first table is the plugin's decoded route (the route, the experts unpacked, and vLLM's layer on them).

| Tokens a step | bf16 (vLLM's layer) | new | new / bf16 | the first run's | / bf16 | best variant | best / bf16 | the plugin's decoded route* / bf16 |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| 16 | 616 us | 464 us | 0.75x | 493 us | 0.80x | gu64 463 us | 0.75x | 2.92x |
| 32 | 763 us | 732 us | 0.96x | 857 us | 1.13x | noskip 732 us | 0.96x | 3.07x |
| 64 | 857 us | 890 us | 1.04x | 1,222 us | 1.43x | tm256 889 us | 1.04x | 3.05x |
| 128 | 919 us | 920 us | 1.00x | 1,285 us | 1.40x | tm256 920 us | 1.00x | 2.93x |
| 232 | 947 us | 960 us | 1.01x | 1,341 us | 1.42x | nb3 960 us | 1.01x | 2.88x |
| 384 | 975 us | 1,033 us | 1.06x | 1,426 us | 1.46x | dn192 1,032 us | 1.06x | 2.83x |
| 512 | 996 us | 1,111 us | 1.12x | 1,501 us | 1.51x | new 1,111 us | 1.12x | 2.79x |
| 768 | 1,155 us | 1,413 us | 1.22x | 2,135 us | 1.84x | dn192 1,260 us | 1.09x | 2.55x |
| 1,152 | 1,209 us | 2,014 us | 1.67x | 2,440 us | 2.01x | gu64 1,689 us | 1.40x | 2.49x |
| 1,536 | 1,266 us | 1,776 us | 1.40x | - | - | dn96 1,776 us | 1.40x | 2.43x |
| 2,048 | 1,544 us | 2,435 us | 1.58x | 2,815 us | 1.83x | new 2,435 us | 1.58x | 2.18x |
| 3,072 | 1,998 us | 3,413 us | 1.71x | - | - | tm256 2,677 us | 1.34x | 1.92x |
| 4,096 | 2,477 us | 4,200 us | 1.70x | 4,865 us | 1.96x | tm256 3,666 us | 1.48x | 1.75x |
| 8,192 | 4,456 us | 7,816 us | 1.75x | 8,983 us | 2.02x | tm256 6,362 us | 1.43x | 1.44x |

*route + the experts decoded + vLLM's own layer on them (the decoded route's GPU time, vLLM's kernels taken as the bf16 layer's).

Each variant, us a layer (all = route, gate+up, down with the sum), against new:
| Tokens | new | dn192 | dn96 | gu112 | gu64 | nb3 | noskip | old | tm256 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 464 | 463 (1.00) | 463 (1.00) | 464 (1.00) | 463 (1.00) | 463 (1.00) | 464 (1.00) | 464 (1.00) | 465 (1.00) |
| 32 | 732 | 733 (1.00) | 732 (1.00) | 733 (1.00) | 733 (1.00) | 733 (1.00) | 732 (1.00) | 860 (1.17) | 732 (1.00) |
| 64 | 890 | 890 (1.00) | 890 (1.00) | 890 (1.00) | 890 (1.00) | 890 (1.00) | 890 (1.00) | 1,228 (1.38) | 889 (1.00) |
| 128 | 920 | 920 (1.00) | 920 (1.00) | 920 (1.00) | 920 (1.00) | 920 (1.00) | 920 (1.00) | 1,287 (1.40) | 920 (1.00) |
| 232 | 960 | 960 (1.00) | 960 (1.00) | 960 (1.00) | 960 (1.00) | 960 (1.00) | 960 (1.00) | 1,342 (1.40) | 961 (1.00) |
| 384 | 1,033 | 1,032 (1.00) | 1,033 (1.00) | 1,033 (1.00) | 1,032 (1.00) | 1,033 (1.00) | 1,032 (1.00) | 1,427 (1.38) | 1,033 (1.00) |
| 512 | 1,111 | 1,111 (1.00) | 1,111 (1.00) | 1,111 (1.00) | 1,111 (1.00) | 1,111 (1.00) | 1,111 (1.00) | 1,500 (1.35) | 1,111 (1.00) |
| 768 | 1,413 | 1,260 (0.89) | 1,261 (0.89) | 1,413 (1.00) | 1,413 (1.00) | 1,752 (1.24) | 1,445 (1.02) | 1,645 (1.16) | 1,649 (1.17) |
| 1,152 | 2,014 | 2,098 (1.04) | 2,098 (1.04) | 2,014 (1.00) | 1,689 (0.84) | 2,365 (1.17) | 2,028 (1.01) | 1,733 (0.86) | 2,257 (1.12) |
| 1,536 | 1,776 | 1,917 (1.08) | 1,776 (1.00) | 2,256 (1.27) | 1,776 (1.00) | 2,606 (1.47) | 1,811 (1.02) | 1,811 (1.02) | 2,344 (1.32) |
| 2,048 | 2,435 | 2,583 (1.06) | 2,436 (1.00) | 2,436 (1.00) | 2,435 (1.00) | 2,730 (1.12) | 2,563 (1.05) | 2,563 (1.05) | 2,465 (1.01) |
| 3,072 | 3,413 | 3,413 (1.00) | 3,413 (1.00) | 3,413 (1.00) | 3,414 (1.00) | 2,940 (0.86) | 3,478 (1.02) | 3,478 (1.02) | 2,677 (0.78) |
| 4,096 | 4,200 | 4,200 (1.00) | 4,200 (1.00) | 4,199 (1.00) | 4,200 (1.00) | 4,009 (0.95) | 4,338 (1.03) | 4,338 (1.03) | 3,666 (0.87) |
| 8,192 | 7,816 | 7,817 (1.00) | 7,803 (1.00) | 7,816 (1.00) | 7,803 (1.00) | 6,892 (0.88) | 7,951 (1.02) | 7,909 (1.01) | 6,362 (0.81) |

gate+up and down alone (us), by variant:
gate_up_us:
| Tokens | new | dn192 | dn96 | gu112 | gu64 | nb3 | noskip | old | tm256 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 301 | 292 | 292 | 292 | 292 | 292 | 292 | 292 | 292 |
| 32 | 477 | 477 | 477 | 477 | 478 | 477 | 477 | 493 | 477 |
| 64 | 604 | 603 | 603 | 603 | 603 | 603 | 604 | 669 | 603 |
| 128 | 619 | 619 | 619 | 619 | 619 | 619 | 619 | 693 | 619 |
| 232 | 637 | 637 | 637 | 637 | 637 | 637 | 637 | 707 | 637 |
| 384 | 671 | 671 | 671 | 671 | 671 | 671 | 671 | 730 | 671 |
| 512 | 709 | 709 | 709 | 709 | 709 | 709 | 709 | 748 | 709 |
| 768 | 787 | 786 | 787 | 787 | 787 | 787 | 787 | 985 | 787 |
| 1,152 | 1,309 | 1,309 | 1,309 | 1,309 | 981 | 1,309 | 1,309 | 1,012 | 1,310 |
| 1,536 | 1,015 | 1,014 | 1,014 | 1,497 | 1,015 | 1,494 | 1,039 | 1,039 | 1,338 |
| 2,048 | 1,404 | 1,404 | 1,404 | 1,403 | 1,404 | 1,535 | 1,491 | 1,491 | 1,375 |
| 3,072 | 1,953 | 1,953 | 1,953 | 1,953 | 1,953 | 1,590 | 1,999 | 1,999 | 1,430 |
| 4,096 | 2,391 | 2,394 | 2,394 | 2,394 | 2,394 | 2,190 | 2,489 | 2,489 | 1,997 |
| 8,192 | 4,414 | 4,414 | 4,414 | 4,414 | 4,414 | 3,699 | 4,507 | 4,508 | 3,408 |

down_us:
| Tokens | new | dn192 | dn96 | gu112 | gu64 | nb3 | noskip | old | tm256 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 186 | 167 | 167 | 167 | 167 | 167 | 167 | 167 | 167 |
| 32 | 248 | 247 | 247 | 247 | 248 | 248 | 247 | 359 | 248 |
| 64 | 279 | 279 | 279 | 279 | 279 | 280 | 279 | 552 | 279 |
| 128 | 291 | 291 | 291 | 291 | 291 | 291 | 291 | 583 | 291 |
| 232 | 311 | 311 | 311 | 311 | 311 | 311 | 311 | 622 | 311 |
| 384 | 345 | 346 | 346 | 346 | 346 | 345 | 345 | 680 | 345 |
| 512 | 383 | 383 | 383 | 383 | 382 | 383 | 382 | 732 | 383 |
| 768 | 591 | 448 | 448 | 591 | 591 | 932 | 623 | 623 | 829 |
| 1,152 | 661 | 754 | 754 | 661 | 661 | 1,013 | 673 | 674 | 904 |
| 1,536 | 706 | 854 | 706 | 706 | 706 | 1,058 | 716 | 716 | 950 |
| 2,048 | 965 | 1,116 | 965 | 965 | 965 | 1,129 | 1,003 | 1,003 | 1,022 |
| 3,072 | 1,367 | 1,368 | 1,368 | 1,368 | 1,368 | 1,264 | 1,386 | 1,386 | 1,159 |
| 4,096 | 1,693 | 1,693 | 1,693 | 1,693 | 1,693 | 1,705 | 1,731 | 1,731 | 1,555 |
| 8,192 | 3,197 | 3,198 | 3,198 | 3,197 | 3,198 | 2,985 | 3,235 | 3,226 | 2,749 |

route_us:
| Tokens | new | dn192 | dn96 | gu112 | gu64 | nb3 | noskip | old | tm256 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 7 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| 32 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| 64 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| 128 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| 232 | 13 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 13 |
| 384 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| 512 | 19 | 19 | 19 | 19 | 19 | 19 | 19 | 19 | 19 |
| 768 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 |
| 1,152 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 |
| 1,536 | 42 | 42 | 42 | 42 | 42 | 42 | 42 | 42 | 42 |
| 2,048 | 54 | 54 | 54 | 54 | 54 | 54 | 54 | 54 | 54 |
| 3,072 | 79 | 78 | 78 | 78 | 78 | 78 | 78 | 78 | 78 |
| 4,096 | 102 | 101 | 101 | 101 | 101 | 101 | 101 | 101 | 101 |
| 8,192 | 195 | 193 | 193 | 194 | 193 | 194 | 193 | 193 | 193 |

## Nsight Compute on the A100's tiled kernel (Qwen3-30B-A3B's layer 0, 2,048 tokens a step, the real layer)

The job's prof part: `mma_gemm_big_kernel`, the first two launches of one call of the layer (the check's calls made once, so that they are the gate and up product and the down), at ncu's locked clock (its times are longer than the sweep's: 1.77 ms against 1,404 us for the gate and up). `moea100c/prof/summary.txt` has the metrics, `details-2048.txt` ncu's own report, `raw-2048.csv.gz` every metric, `sass-2048.csv.gz` the stall samples by instruction (the report itself, 21 MB, is not kept).

| | Gate and up | Down |
| :--- | ---: | ---: |
| Duration | 1.77 ms | 1.07 ms |
| Tensor pipe; tensor bf16 ops, % of peak | 27.1%; 25.5% | 22.8%; 21.1% |
| Issue slots active | 27.8% | 25.1% |
| DRAM throughput; bytes | 24.9%; 659 MB | 27.3%; 329 MB |
| L2 hit rate | 58.6% | 65.2% |
| Warp instructions | 204.6 M | 116.2 M |
| Registers a thread; shared memory | 255; 98.3 KB | 255; 98.3 KB |
| Warps active | 11.1% | 10.9% |
| Stalls (of those sampled) | wait 37%, barrier 17%, selected 15%, no instructions 9%, short scoreboard 8%, long scoreboard 7% | wait 36%, selected 15%, long scoreboard 13%, barrier 12%, no instructions 9%, short scoreboard 7% |

By instruction (`moea100c/prof/sass-regions.txt`, from `../l4-tiled/scripts/sassreg.py`: the consumers' loop up to the last HMMA, the producers' loop, and the epilogue and the paths a step rarely takes between; four producer warps and four consumer warps a block):

```
sass-2048.csv.gz, launch 0: 17992 static instructions, 204.6 M executed
  consumers' main loop     static    629  executed   80.57 M (39.4%)  samples  47.2%  [wait 49%, barrier 16%, selected 12%, short_sb 12%]
  epilogue and rare code   static  15784  executed   10.57 M ( 5.2%)  samples  15.1%  [wait 32%, no_inst 29%, selected 15%, long_sb 8%]
  producers                static   1579  executed  113.45 M (55.5%)  samples  37.8%  [wait 24%, barrier 23%, selected 19%, long_sb 13%]
  producers' share of a consumer warp's life 61%; the epilogue's share of it 24%; the producers' instructions: MOV 15%, IADD3 14%, LOP3 11%, ISETP 9%, IMAD 9%, PRMT 8%
sass-2048.csv.gz, launch 1: 17992 static instructions, 116.2 M executed
  consumers' main loop     static    629  executed   49.18 M (42.3%)  samples  48.8%  [wait 48%, barrier 15%, selected 13%, short_sb 12%]
  epilogue and rare code   static  15784  executed    9.84 M ( 8.5%)  samples  18.1%  [long_sb 29%, wait 25%, no_inst 23%, selected 12%]
  producers                static   1579  executed   57.15 M (49.2%)  samples  33.1%  [wait 24%, selected 20%, long_sb 19%, barrier 15%]
  producers' share of a consumer warp's life 49%; the epilogue's share of it 27%; the producers' instructions: MOV 15%, IADD3 14%, LOP3 12%, ISETP 9%, IMAD 8%, PRMT 8%
```

The same code on an L4 executes the same instructions (205.7 M for the gate and up). The kernel is neither memory-bound nor tensor-bound: 25% of DRAM, 25-27% of the tensor peak, 11% of the warps. The consumers' loop spends half its samples in `wait` (the tensor pipe's issue gaps) and a sixth at the barrier for the producers; the producers spend 23% (gate and up) and 15% (down) of theirs at the barrier for the consumers and are done at about 60% and 50% of a consumer warp's life, so they are not the bound; the epilogue and the paths between are 24% and 27% of a consumer warp's life, instruction fetch still 29% and 23% of their stalls (9% of all stalls; an L4's, 5%). The producers' 56% of the instructions are mostly address arithmetic and moves (MOV 15%, IADD3 14%, LOP3 11%, ISETP 9%, IMAD 9%; the decode's byte permutes 8%).

## The dense products (Qwen3-30B-A3B's qkv, o and router shapes, 1 to 512 tokens a step)

Each route of the library against cuBLAS: microseconds, and in brackets the ratio to cuBLAS's time; the last column is the best route's.

| Matrix (O x K) | Tokens a step | bf16 (cuBLAS) | decode | default | gemm | best / bf16 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 x 2048 | 1 | 3.9 us | 6.9 (1.79) | 8.7 (2.25) | 8.7 (2.25) | 1.79x |
| 128 x 2048 | 8 | 6.7 us | 9.8 (1.45) | 9.5 (1.42) | 9.5 (1.42) | 1.42x |
| 128 x 2048 | 16 | 7.0 us | 10.0 (1.44) | 10.8 (1.54) | 10.8 (1.55) | 1.44x |
| 128 x 2048 | 32 | 7.1 us | 10.3 (1.44) | 14.1 (1.98) | 14.9 (2.10) | 1.44x |
| 128 x 2048 | 64 | 7.5 us | 10.7 (1.43) | 17.4 (2.31) | 23.4 (3.10) | 1.43x |
| 128 x 2048 | 128 | 6.7 us | 9.9 (1.47) | 26.5 (3.93) | - | 1.47x |
| 128 x 2048 | 232 | 7.4 us | 10.5 (1.43) | 18.1 (2.46) | - | 1.43x |
| 128 x 2048 | 512 | 8.7 us | 11.9 (1.37) | 19.0 (2.19) | - | 1.37x |
| 2048 x 4096 | 1 | 20.6 us | 39.5 (1.92) | 19.6 (0.95) | 19.7 (0.96) | 0.95x |
| 2048 x 4096 | 8 | 18.5 us | 37.6 (2.03) | 21.2 (1.15) | 21.1 (1.14) | 1.14x |
| 2048 x 4096 | 16 | 18.9 us | 38.3 (2.03) | 23.6 (1.25) | 23.5 (1.24) | 1.24x |
| 2048 x 4096 | 32 | 19.3 us | 38.9 (2.01) | 21.3 (1.10) | 29.4 (1.52) | 1.10x |
| 2048 x 4096 | 64 | 19.8 us | 40.4 (2.04) | 26.6 (1.34) | 43.8 (2.21) | 1.34x |
| 2048 x 4096 | 128 | 24.0 us | 47.7 (1.99) | 41.2 (1.71) | - | 1.71x |
| 2048 x 4096 | 232 | 29.4 us | 56.1 (1.91) | 40.6 (1.38) | - | 1.38x |
| 2048 x 4096 | 512 | 54.4 us | 80.8 (1.49) | 66.8 (1.23) | - | 1.23x |
| 5120 x 2048 | 1 | 21.0 us | 56.8 (2.70) | 24.8 (1.18) | 24.8 (1.18) | 1.18x |
| 5120 x 2048 | 8 | 21.1 us | 57.1 (2.71) | 26.3 (1.25) | 26.2 (1.24) | 1.24x |
| 5120 x 2048 | 16 | 21.3 us | 58.3 (2.74) | 28.4 (1.33) | 28.3 (1.33) | 1.33x |
| 5120 x 2048 | 32 | 21.3 us | 59.1 (2.78) | 28.2 (1.33) | 38.8 (1.83) | 1.33x |
| 5120 x 2048 | 64 | 21.6 us | 61.8 (2.86) | 32.9 (1.53) | 57.3 (2.65) | 1.53x |
| 5120 x 2048 | 128 | 26.1 us | 68.1 (2.61) | 49.5 (1.90) | - | 1.90x |
| 5120 x 2048 | 232 | 39.7 us | 82.7 (2.08) | 82.4 (2.08) | - | 2.08x |
| 5120 x 2048 | 512 | 72.1 us | 115.2 (1.60) | 148.0 (2.05) | - | 1.60x |

Glyd's best route is slower than cuBLAS at every size but one (o at 1 token, 0.95x): 1.18x to 2.08x for qkv (5120 x 2048), 0.95x to 1.71x for o (2048 x 4096), 1.37x to 1.79x for the router (128 x 2048). That is the case for the A100's default, the attention products and the router left bf16.

## The first run, kept as history (`moea100b/`, `README-first-run.md`)

The tree before the epilogue change (9267523e), the A100's variants as variants. Against it, like for like (the first run's mode with the attention products left bf16; its default then packed them, 5.32 requests a second):

| | First run | This run |
| :--- | ---: | ---: |
| bf16, saturated, two passes (req/s) | 4.02, 4.00 | 4.12, 3.94 |
| Glyd with the lossless KV cache, saturated (req/s; vs bf16) | 5.44 (1.35x) | 6.18 (1.50x) |
| First token, saturated (ms; vs bf16) | 15,933 (0.58x) | 14,233 (0.53x) |
| Each token, saturated (ms; vs bf16) | 67.5 (2.12x) | 62.9 (2.04x) |
| First token at one request a second (ms; vs bf16) | 119 (1.17x) | 110 (1.08x) |
| Each token at one request a second (ms; vs bf16) | 12.3 (0.86x) | 12.3 (0.88x) |
| Prompt step (ms; us a token) | 140; 76 | 133; 72 |
| Decode step (ms; sequences) | 37.5; 95 | 30.5; 103 |
| Qwen3-30B-A3B's layer at 232, 1,152, 2,048, 8,192 tokens (us; vs bf16) | 1,341 (1.42x), 2,440 (2.01x), 2,815 (1.83x), 8,983 (2.02x) | 960 (1.01x), 2,014 (1.67x), 2,435 (1.58x), 7,816 (1.75x) |
| Checks | 101 | 162 |

The first run's own numbers, tables and notes are in `README-first-run.md`; its job as run is `job/`.

## Not measured here

- A tuned bf16 baseline; an A100 of 80 GB, where Qwen3-30B-A3B itself runs; tensor parallel; other prompt and output lengths; the H100 (its record is `../h100-moe-v028-2026-10-03`).
- The thresholds 64 and 64 together, and the tile of 256 as the default above 2,048 tokens: the variants were run one at a time.
- A Glyd mode with the attention products packed (the first run has it: 5.32 requests a second against 5.44 with them bf16).

## Files

- `job-next/`: the second run's job as run (`moe_job.sh`, the scripts of its tables, `make_tar.sh` for its source tar); `job/` the first run's.
- `moea100c/summary.txt`: the job's summary as it came; `steps.txt` its steps with times; `machine.txt`, `env.txt`, `COMMIT`, `job.log`.
- `moea100c/serve/`: a directory a mode (`bf16`, `glyd-picked+kv`, `bf16-2`) with each pass's `vllm bench serve` output and result, the server's log (`serve-*.txt.gz`: gunzip it for `job-next/sweep_summary.py steps`), the nvidia-smi samples of each pass and of the whole server start (`smi-reasons.csv`); `clocks.txt`, `steps.txt` and `kv-notes.txt` beside them.
- `moea100c/check/check.txt`: the 162 checks; `moea100c/sweep/`: the layer sweep (`layer.json`, `layer.txt`, `summary.txt`); `moea100c/prof/`: the Nsight capture; `moea100c/dense/`: the dense products; `moea100c/log/`: the downloads, the environment and the thermal preflight (43 s of matmuls on the idle GPU: 66 C, 362 W, no thermal slowdown).
- `moea100b/`: the first run's, laid out the same; `README-first-run.md` its README. `../l4-tiled/`: the L4's three captures and the epilogue change.
