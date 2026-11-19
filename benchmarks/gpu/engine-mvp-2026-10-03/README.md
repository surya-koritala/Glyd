# glyd-engine's MVP on an L4: a first run from nothing, its own KV tables, its own dense products, a server and a chat (2026-10-03)

Qwen3-8B (and 0.6B) on the NVIDIA L4 of the first engine record (22 GiB, Ada, compute capability 8.9; AWS g6.4xlarge, driver 595.91), `glyd-engine` (Rust, branch `engine-mvp`, the commit in `commit.txt`) against vLLM 0.30.0 (torch 2.13, CUDA 13.0) in its default mode (`glyd@1`: Glyd's weights with vLLM's own cache).
The engine of the first record decoded one prompt at a time from weights packed by a separate step, with KV tables made by the vLLM plugin's Python and a prompt kernel that was slow past a thousand tokens. This record is what the MVP's first items changed: the prompt kernel, the tables made by the engine itself, a model made ready on its first use (download, packing, tables), the chat template and sampler, an OpenAI-compatible server and a chat, `glyd run` starting the engine, and the engine's own dense bf16 products for long prompts.

## What to know

- **An empty machine reaches an answer in 15.8 s for Qwen3-0.6B and in 2 min 28 s for Qwen3-8B, through the `glyd` program** (`glyd run Qwen/Qwen3-8B --prompt ...`): the engine downloads the checkpoint itself (8 connections, resumable, checked against the Hub's SHA-256, into the cache `huggingface_hub` keeps), packs it for the GPU, makes the KV tables on its own forward pass and answers. 8B: download 16.4 GB 1 min 07 s (246 MB/s), packing 22 s (12.4 GB written), tables and load 1 min 03 s. **The second start takes 3.2 s** (0.6B: 1.1 s). The program is 9.7 MB (the kernels for Ampere, Ada and Hopper compressed inside it) and the library 28.5 MB; nothing else is installed (no Python, no CUDA toolkit; cuBLAS is used if the machine has it).
- **Prompt time against vLLM+Glyd, alternated in one hold of the box (three rounds, medians of the rounds): 1,024 tokens 394.2 ms against 412.9 (0.955x), 3,500 tokens 1,151.2 ms against 1,159.9 (0.992x)** with the engine's own dense products; with cuBLAS 420.7 and 1,183.8 ms (1.019x and 1.021x). An earlier build gave 0.950x and 0.991x (own) and 1.005x and 1.018x (cuBLAS) in the same kind of run.
- **The engine's own dense products match cuBLAS.** Over the four products of a layer, at 512 to 4,096 rows, their time is 0.98 to 1.06 times cuBLAS's with the engine's choice of tile (0.98 to 1.05 with the best tile for each product), and the gate-and-up product with the SiLU and the multiplication by `up` in its epilogue is 0.86 to 0.98 times the time of cuBLAS's product and the SiLU kernel together, with the same bits as the separate steps (0 of 43 million values differ). In the engine, alternated with cuBLAS in one hold: a 1,024-token prompt 383.4 ms with its own products against 397.3 with cuBLAS, a 3,500-token prompt 1,127.3 against 1,128.6. The seven tiles tried (the six it chooses among and one more) give the same bits as one another on every size and product tried (`gemm/tile-bits.out`), so a reply does not depend on the length of the prompt around it.
- **The KV tables the engine makes are as good as the plugin's.** Qwen3-8B at 8,192 positions: 56 s of calibration (67 s with the load; for Qwen3-0.6B the plugin's Python takes 85 s against the engine's 5 s). Against the plugin's tables for the same model the windows are the same in 98.8% (keys) and 96.8% (values) of 36,864 each, every difference one or two steps (the difference between two of the plugin's own runs is of the same size); the rate of values left outside the windows is 0.0131% and 0.0187% against 0.0133% and 0.0185%; no token of the vocabulary needs more than the cache holds. With them, the 8 short prompts give the same 64 tokens as with the plugin's tables (8 of 8), the teacher-forced top-1 agreement with vLLM+Glyd is 98.63% (98.50% with the plugin's), and nothing is lost in any run.
- **Tokens against vLLM+Glyd are where they were**: 6 of 8 short prompts the same in all 64 tokens (457 of 512 positions) in every configuration below, every difference at a near tie; the teacher-forced top-1 agreement over 1,535 positions is 98.5% to 98.7% in all of them (vLLM against itself with another step size: 98.6% to 99.0%). The three long prompts part from vLLM at token 7 (the 1,030-token prompt, in every configuration), at token 35 or not at all (2,100 tokens) and at token 13 or not at all (3,500 tokens), each time where the engine's choice has a probability of 7 to 18% and the two runs' logprobs are within 0.01 to 0.10 of each other. The final build's tokens, logprobs and scores are the same bits as the build before it (the embedding gathered in batches, the fatbin compressed).
- **The 12-bit layout on the L4** (11.48 GiB against the tiered layout's 10.37): the same tokens, scores 98.70%, a 1,024-token prompt in 329 ms (323 in another hold) against the tiered layout's 398 to 412 (the library's fused product multiplies it, where the tiered layout decodes the matrices first) and a 3,500-token prompt in 1,179 to 1,197 ms against 1,150 to 1,168; decode the same (21.5 tokens a second). The tiered layout stays the default.
- **A server and a chat**: `glyd-engine serve` (the OpenAI API: chat and plain completions, streaming, usage, stop strings, seeds, an API key, the thinking returned apart as `reasoning_content`, requests it cannot honour refused in words) with the chat page; `run` (the same chat in the terminal and at one URL, or one answer with `--prompt`), `doctor`, `login`; the Rust `glyd` program starts the engine for `glyd run`, `serve`, `doctor` and `login` where it is installed and runs the model, else the Python tool. All 20 checks of the API pass (`api/serve-smoke-with-key.out`); a conversation's next turn reuses the cache of the turns before it: a 2,641-token prompt takes 0.96 s the first time and 0.06 s again, the next turn of a 1,723-token conversation 0.35 s against 1.13 s. Under 100 s of mixed traffic (94 rounds of requests of every length up to the context, streams abandoned half way, stop strings, several at once, prompts past the context) six fixed prompts gave the same replies at the end as on the cold server, with the GPU's memory unchanged (12,294 MiB before and after). The terminal chat, tried in a pseudo-terminal: Ctrl-C stops an answer and the prompt returns, `/bye` and Ctrl-D leave with status 0, Ctrl-C at the prompt with 130, `login` does not echo the token and Ctrl-C puts the terminal back.
- **An A100 and an H100** (one script, 3 to 6 minutes of its 38 minutes each; `a100/`, `h100/`): the same tokens as on the L4 (A100: 6 of 8 short prompts the same in all 64 tokens, 466 of 512 positions, teacher-forced top-1 98.70% with every score the L4's own; H100: 7 of 8, 479, 98.57%). One user, a 1,024-token prompt: **the A100 decodes at 92.8 tokens a second in the 12-bit layout (1.23x vLLM bf16's 75.3) and 65.0 in the tiered one (0.86x); the H100 at 162.6 (1.06x of bf16's 152.8) and 100.4 (0.66x)**: the 12-bit layout is the engine's default on both, as on every GPU but Ada. On an A100 cuBLAS and the engine's own products take the same time (12-bit, 3,500 tokens: 277.7 and 285.0 ms; the library's routes 346.3); on an H100 cuBLAS is 1.4 to 1.5 times faster than the engine's own products (36.7 and 115.2 ms against 36.2 and 170.9 for the 12-bit layout), which are 1.4 to 2.3 times faster than the library's routes (239.1 and 392.6 ms at 3,500 tokens). The tables for the 8B take 13 s on the A100 and 6 s on the H100 (the L4: 56 s), packing 23 s and 13 s.
- **Several users at once** (up to 8 on one GPU, their replies decoded together, the lossless cache checked as it is reused and as it runs short): [`users/`](users/README.md). **The library's new smallest layout** (glyd-v4: each matrix in the smaller of the two layouts, each pack loaded in its own; a copy of the earlier layout packed again on its first use; tokens the same as before on Qwen3-8B and Qwen3-4B): [`layouts/`](layouts/README.md). **Not here yet**: models other than Qwen3 dense, Blackwell.

## First run, from an empty machine

| | Qwen3-0.6B | Qwen3-8B |
| :--- | ---: | ---: |
| download (the Hub, this machine's network) | about 7 s (1.5 GB) | 1 min 07 s (16.4 GB, 246 MB/s) |
| packing for this GPU (layout `mma`) | 2 s | 22 s (12.4 GB on disk) |
| the KV tables, 4,096 / 8,192 positions | 5 s | 55 to 61 s |
| to the first answer | **15.8 s** | **2 min 28 s** through the `glyd` program (2 min 33 s the stages one after another: 67 + 22 + 64 s) |
| the second start, to the answer | 1.1 s | 3.2 s |

`l4/batch-b.out` (0.6B from empty caches, in one command), `l4/first-run-8b-glyd.out` (8B, through the `glyd` program), `l4/first-run-8b-download-and-pack.out` and `l4/batch-c.out` (the stages of 8B).

## The KV tables made by the engine against the plugin's

Qwen3-8B, 8,192 positions (`tables/`): windows equal 98.785% (keys, 448 of 36,864 differ) and 96.753% (values, 1,197 differ), by one step on average and two at most; the held-out rates 0.0131% and 0.0187% against the plugin's 0.0133% and 0.0185%; the calibration took 6 s, 6 s and 45 s in its three passes.

## Tokens and scores, against vLLM+Glyd (an L4's, 8,192 tokens a step)

`numbers.txt` (the data in `proofs/`; the reference is `glyd-engine/tests/fixtures/ref-8b-glyd1.json`):

| | short prompts, same in all 64 / positions equal | long prompts, same | teacher-forced top-1 | mean abs. logprob difference |
| :--- | ---: | ---: | ---: | ---: |
| plugin's tables, tiered, cuBLAS (earlier run) | 6 of 8 / 457 | 2 of 3 | 98.50% | 0.0329 |
| the engine's own tables, tiered, cuBLAS | 6 of 8 / 457 | 0 of 3 | 98.63% | 0.0307 |
| 12-bit layout, cuBLAS | 6 of 8 / 457 | 1 of 3 | 98.70% | 0.0335 |
| tiered, the engine's own products | 6 of 8 / 457 | 1 of 3 | 98.70% | 0.0335 |
| 12-bit, the engine's own products | 6 of 8 / 457 | 1 of 3 | | |

The first differences of the long prompts (tokens): 7, 35 and 13 (own tables); 7, none and 13 (12-bit, and the engine's own products); 7, none and none (the earlier run). The engine's own products and the library's fused 12-bit product give the same scores to the last digit.

## The dense products

One matrix of Qwen3-8B decoded to bf16, 0.8 s of sustained running a point (the L4 at its 72 W cap and 75 to 84 C): time against cuBLAS's with the engine's tile for the number of rows (`gemm/sweep-512-4096.out`):

| rows | gate and up (24,576 x 4,096) | down (4,096 x 12,288) | qkv (6,144 x 4,096) | o (4,096 x 4,096) | the four, summed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 0.96 | 1.12 | 0.89 | 1.05 | 0.995 |
| 768 | 1.02 | 0.96 | 0.87 | 1.00 | 0.978 |
| 1,024 | 1.07 | 1.08 | 0.89 | 1.01 | 1.039 |
| 1,536 | 1.06 | 1.04 | 1.08 | 0.96 | 1.050 |
| 2,048 | 1.04 | 1.09 | 1.07 | 0.88 | 1.038 |
| 2,560 | 1.07 | 1.10 | 1.11 | 0.84 | 1.056 |
| 3,072 | 1.03 | 1.10 | 1.06 | 1.08 | 1.058 |
| 3,500 | 1.01 | 1.05 | 1.05 | 1.05 | 1.030 |
| 4,096 | 1.00 | 1.05 | 1.04 | 1.02 | 1.024 |

The gate-and-up product with the SiLU in its epilogue against cuBLAS plus the SiLU kernel: 0.96 to 0.98 at 512 to 1,024 rows (its 128 x 256 tile with a step of 64), 0.86 to 0.95 from 1,536 rows (256 x 128), and no bit differs from the separate steps. On the L4 the engine's own products are used from the 896 rows where the library decodes the matrix first; below it the library's fused products are faster (512 rows: 225 ms the prompt against 254; 768: 320 against 332).

The kernel's earlier version ran at 1.16 to 1.27 times cuBLAS's time on the three large products at both sizes with its best tile (`gemm/gemm-v2.out`): its main loop spent 7.4 instructions on every tensor-core instruction, which an L4 pays for in clock under its power cap. With the address arithmetic done once before the loop it spends 2.05 (`gemm/gemm-v3.out` and the table above). Every tile gives the same bits: 0 of up to 86 million values of the seven tiles differ from the first one's on 12 sizes (`gemm/tile-bits.out`). The kernels built by nvcc 12.8 run as fast as those built by 13.0 (a 1,024-token prompt 388 to 390 ms against 385; 3,500 tokens 1,121 to 1,142 against 1,125 to 1,130), so one build serves drivers of both generations.

| the whole prompt, tiered, one user | 1,024 tokens | 3,500 tokens |
| :--- | ---: | ---: |
| cuBLAS for the decoded matrices | 411.9 ms | 1,167.8 ms |
| the engine's own products, SiLU fused | 398.0 ms | 1,150.4 ms |
| the engine's own, SiLU not fused | 410.6 ms | 1,219.1 ms (the GPU hotter) |
| the library's routes alone (no cuBLAS, no own products) | 410.9 ms | 1,468.2 ms (2,048 tokens a step) |
| 12-bit layout, cuBLAS / own | 329.0 / 335.1 ms | 1,196.5 / 1,179.1 ms |

By kind of work at 1,024 tokens (cuBLAS): the four products 352.7 ms (88%: the decoding of the matrices about 100 ms of it), the prompt attention 11.2 ms, the embedding 11.1 ms, the SiLU kernel 9.1 ms, the norms 7.5 ms. At 3,500 tokens: the products 975 ms (81%), the attention 105.5 ms (8.8%), the embedding 34.8 ms, the SiLU kernel 36.1 ms. The final build decodes each group of the embedding once a prompt and gathers the rows in batches: the embedding takes 6.7 to 6.9 ms (1,024 tokens) and 14.6 to 15.1 ms (3,500).

## Prompt time against vLLM+Glyd, alternated

vLLM+Glyd (7 prompts a run) and the engine (7 runs) in one hold, three rounds, prefill ms. The final build (`l4/batch-g.out`, `clocks/batch-g-smi.csv`):

| round | vLLM+Glyd 1,024 | engine, own | engine, cuBLAS | vLLM+Glyd 3,500 | engine, own | engine, cuBLAS |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 408.0 | 389.1 | 406.5 | 1,144.9 | 1,127.3 | 1,160.4 |
| 2 | 412.9 | 394.9 | 420.7 | 1,159.9 | 1,151.2 | 1,183.8 |
| 3 | 437.0 | 394.2 | 421.3 | 1,187.7 | 1,156.4 | 1,193.3 |
| **median** | **412.9** | **394.2 (0.955x)** | **420.7 (1.019x)** | **1,159.9** | **1,151.2 (0.992x)** | **1,183.8 (1.021x)** |

The build before the embedding change (`l4/batch-d.out`): 1,024 tokens vLLM+Glyd 412.3, own 391.5 (0.950x), cuBLAS 414.4 (1.005x); 3,500 tokens 1,176.5, 1,165.4 (0.991x) and 1,197.5 (1.018x).

## An A100 and an H100

`glyd-engine/jobs/engine_gpu_job.sh` on an A100 SXM4 40 GB (30 vCPUs) and an H100 80 GB HBM3 (26 vCPUs) of Lambda's `gpu-base-24-04` (driver 580.126.20): it built the engine for the one architecture (1 min 22 s and 1 min 05 s), downloaded Qwen3-8B meanwhile, packed it in both layouts, made the tables, and ran the proofs, the benches and a first run of Qwen3-0.6B. The A100's run took 6 min, the H100's 3. The library is the L4 record's wheel's. `a100/` and `h100/` hold each run's `summary.txt`, steps and logs.

Tokens and scores against vLLM+Glyd's (the L4's reference):

| | A100: short prompts same in all 64 / positions, long prompts same, teacher-forced top-1 | H100: the same |
| :--- | :--- | :--- |
| tiered, cuBLAS | 6 of 8 / 466, 2 of 3, 98.70% | 7 of 8 / 479, 2 of 3, 98.57% |
| 12-bit, cuBLAS | 6 of 8 / 466, 2 of 3, 98.70% | 7 of 8 / 479, 2 of 3, 98.57% |
| 12-bit, the engine's own products | 6 of 8 / 466, 2 of 3, 98.70% | 7 of 8 / 479, 2 of 3, 98.57% |

The A100's scores are the L4's exactly: all 1,535 largest-logit tokens and logprobs are the same, because the scoring pass does the same arithmetic on both GPUs. The H100's differ in their last bits (its largest-logit token is the L4's at 1,517 of 1,535 positions; the mean absolute logprob difference from vLLM+Glyd's 0.0296 against the L4's 0.0335). The greedy runs' tokens are the L4 engine's on 7 of 8 short prompts on the A100 (468 of 512 positions) and 6 of 8 on the H100 (455), differing at near ties.

One user, a 1,024-token prompt, 64 new tokens (the medians of the job's three ways of multiplying a prompt; vLLM bf16 with its own cache from [`a100-vllm-kv-step-2026-10-02`](../a100-vllm-kv-step-2026-10-02) and [`h100-vllm-kv-2026-10-02`](../h100-vllm-kv-2026-10-02)):

| | weights | decode ms a token | tokens a second | against vLLM bf16 |
| :--- | ---: | ---: | ---: | ---: |
| A100, 12-bit (the default) | 11.48 GiB | 10.78 | 92.8 | 1.23x (bf16 75.3, 13.27 ms) |
| A100, tiered | 10.37 GiB | 15.40 | 65.0 | 0.86x |
| H100, 12-bit (the default) | 11.48 GiB | 6.15 | 162.6 | 1.06x (bf16 152.8) |
| H100, tiered | 10.37 GiB | 9.96 | 100.4 | 0.66x |

Prompt time, ms (1,024 tokens / 3,500 tokens), by how a long prompt is multiplied (cuBLAS; the engine's own products; the library's routes):

| | cuBLAS | the engine's own | the library's routes |
| :--- | ---: | ---: | ---: |
| A100, 12-bit | 104.4 / 277.7 | 101.0 / 285.0 | 105.3 / 346.3 |
| A100, tiered | 180.6 / 606.5 | 180.3 / 606.4 | 180.8 / 623.2 |
| H100, 12-bit | 36.7 / 115.2 | 36.2 / 170.9 | 36.4 / 239.1 |
| H100, tiered | 41.5 / 117.3 | 57.3 / 173.4 | 111.2 / 392.6 |

The tiered layout's prompt on an A100 is dominated by decoding every matrix first (180 ms at 1,024 tokens against 104 for the 12-bit layout, which the library multiplies without that step). On an H100 the engine's own products (`mma.sync`, as on the L4) run at 0.67 to 0.72 of cuBLAS's speed where a matrix is decoded first, and cuBLAS uses Hopper's `wgmma`; they are 1.4 to 2.3 times faster than the library's routes where the library decodes a matrix first, so the engine uses cuBLAS where the machine has it and its own products where it does not. Setup: packing the 8B 23 s (A100) and 13 s (H100) in the 12-bit layout, the tables at 8,192 positions 13 s and 6 s, a first run of Qwen3-0.6B from nothing 29 s and 13 s (the second start 1 s and under 1 s).

## Files

`l4/`: `batch-b.out` (the doctor, the tables of the 8B, the proofs with them and with the 12-bit layout, the benches, the first run of 0.6B, the chat and the API), `batch-c.out` (the sweep, the engine with each way of multiplying a prompt, the proofs with its own products, the 8B's tables), `batch-d.out` and `batch-g.out` (the alternated timing), `batch-e.out` (the 8B through the `glyd` program, the final build's proofs, cuBLAS and the engine's own products alternated, nvcc 12.8 against 13.0, the API with a key, the terminal chat, the mixed traffic, the tiles' bits), `batch-f.out` (the compressed build, the terminal and login, the A100/H100 script's dry run), `first-run-8b-*.out`, `prefill-by-product-route.out`. `a100/`, `h100/`: the A100's and H100's runs of the job script. `gemm/`: the benchmarks of the kernels. `tables/`: the tables against the plugin's. `proofs/`: the engine's tokens and scores for each configuration. `api/`: the server's check of itself. `clocks/`: the GPU's clock, power and temperature once a second during each hold. `numbers.txt`: the figures above, computed from these files by [`glyd-engine/tools/record_mvp.py`](../../../glyd-engine/tools/record_mvp.py); the other tools (`gemm_bench.cu`, `serve_smoke.py`, `serve_soak.py`, `pty_chat_test.py`, `engine_job_summary.py`) and the scripts for the A100 and H100 (`glyd-engine/jobs/`) are beside it. The engine's lines about the cache's storage are shortened to "KV cache: nothing lost" in these logs.
