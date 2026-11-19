# glyd-engine runs Qwen3.5 (2026) on an RTX 4090 (2026-10-06)

Qwen/Qwen3.5-0.8B and Qwen/Qwen3.5-9B (released 2026-02) on `glyd-engine`'s own forward pass for Qwen3.5 (`src/model35.rs`, `kernels/qwen35.cu`: development build, one sequence,
not served, not in `machine.rs: GPUS`), every matrix packed by the library as it loads (tiered layout, the GPU's default on Ada), against vLLM 0.30.0 (torch 2.13.0+cu130,
transformers 5.18.0, `--language-model-only`) bf16 and vLLM with Glyd (the plugin of this tree, glyd-gpu 0.29.0's code). One job (`glyd-engine/jobs/engine35_job.sh`) on
two Runpod RTX 4090 pods, the plan's runs 2 and 3 in one (docs/superpowers/plans/2026-10-06-engine-qwen35.md, tasks 10 to 12).

**No result here measures the code as merged.** The tokens and the engine's speed come from run 1's tree (05379a9d), the kernel check from run 2's (cb2950d6). Since then: the
loader takes each layer's matrices by name (b and a apart, as the packer keeps them), the forward pass's decode step uses the attention with its keys in 16 parts, the gated
norm's weight is rounded to bf16 as it loads (vLLM's), `Buf::write` and `Buf::zero` wait for their copies (below), and the kernel check has new cases and stricter bounds. None
of that has run on a GPU: the next run is the job as it stands, which also adds the long prompts.

| | pod | tree | what it measured |
| :--- | :--- | :--- | :--- |
| `run1/` | RTX 4090, a 48 GB board, driver 595.91 | 05379a9d | the kernels (a check with a stream-ordering bug, below), the tokens of both models against vLLM bf16, one user's speed of the 9B on the engine |
| `run2/` | RTX 4090, 24 GB, driver 580.159 | cb2950d6 | the kernels (all within the check's bounds then), vLLM bf16's and vLLM with Glyd's one-user speed of the 9B; the engine itself did not load (below) |

## Results

**The kernels against the CPU reference** (`run2/kcheck35.out`, tree cb2950d6, the check as it was then). Equal bit for bit: the residual sum, the values copied into the cache,
the convolution's state; 0 bf16 places apart: the keys normed and turned into the cache, the small product, and the causal convolution with SiLU (227,328 of 227,328 values)
against the reference, which rounds once after SiLU as vLLM does (transformers' rounding was not checked). At most 1 place apart: the zero-centered norms and the queries' norm
and rotary (99.92 to 100% equal). The gated norm, given the checkpoint's fp32 weight: 36,863 of 36,864 equal, 1 place at most; against vLLM's arithmetic (its weight in bf16)
70.55% equal, 1 place at most, so the loader now rounds that weight to bf16 as vLLM 0.30 holds it (`A_log` stays fp32, as vLLM 0.30 holds it). The gated delta rule (2 value
heads a key head): over 150 tokens one at a time 99.97% equal, up to 5 places apart, the largest difference 6.5e-4 of the largest output, the state within 1.1e-7; in chunks of
64 (q and k rounded to bf16, as vLLM's prompt path keeps them) 99.94% equal, up to 13 places apart, 1.3e-3, the state within 1.3e-5, and within 5.2e-3 of the token-by-token
kernel; a decode step 512 of 512 equal, the state within 2.4e-7. Head-256 attention with the output gate, against an fp64 softmax (the kernel rounds the probabilities to bf16
for the values, as FlashAttention does): the prompt's rows and the decode steps with the keys in 16 parts each within 6.5e-3 of the largest output, at positions 40 to 46 only,
where at most 2 of the 16 parts hold keys. A decode step is not the prompt kernel's bits (84.74% of values equal, up to 27,959 places apart where the output is near zero);
the check that compared them to a 2-place bound passed by its fallback to a bound relative to the largest value, and is gone.

The check now (`src/kcheck35.rs`, not yet run on a GPU): a bound in places holds for every value, with no fallback; the delta rule's outputs are bounded within one bf16 place
of the largest output (3.9e-3) and the attention within 1e-2, their places shown; and it adds the attention at positions 1,500 to 1,502 (all 16 parts hold keys), the 27B's 6
query heads to a key-value head, and the delta rule at 1 and 3 value heads a key head (the 0.8B and 2B, the 27B).

`run1/kcheck35.out` failed the delta rule and the convolution: the check's inputs were copied on the legacy stream while the kernels ran on the engine's non-blocking stream.
`Buf::write` (whose copy from pageable memory returns once the data is staged) and `Buf::zero` now wait for the legacy stream themselves, for every caller, the Qwen3 path
included; run 2's check, which waited in each caller, passes on the same kernels.

**Greedy tokens against vLLM 0.30.0 bf16** (`run1/compare-*.out`, tree 05379a9d, 16 prompts, 64 tokens each; vLLM against itself, one batch against one prompt at a time, is the
floor):

| | prompts equal in all 64 tokens | first token equal | tokens equal from the start | mean / largest logprob difference over those |
| :--- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B, engine against vLLM (batch) | 11 of 16 | 16 of 16 | 782 of 980 | 0.0089 / 0.118 |
| Qwen3.5-0.8B, engine against vLLM (one at a time) | 11 of 16 | 15 of 16 | 816 of 980 | 0.0091 / 0.106 |
| Qwen3.5-0.8B, vLLM against itself | 11 of 16 | 15 of 16 | 768 of 980 | 0.0077 / 0.102 |
| Qwen3.5-9B, engine against vLLM (batch) | 10 of 16 | 16 of 16 | 762 of 1,024 | 0.0075 / 0.117 |
| Qwen3.5-9B, engine against vLLM (one at a time) | 11 of 16 | 16 of 16 | 769 of 1,024 | 0.0062 / 0.101 |
| Qwen3.5-9B, vLLM against itself | 11 of 16 | 16 of 16 | 792 of 1,024 | 0.0050 / 0.093 |

**Short contexts only.** The 16 prompts are 5 to 32 tokens, so every comparison stays within 96 positions: the delta rule in chunks never ran more than one 64-token chunk end to
end, and neither the prompt's attention over many tiles nor a decode step at a long context was compared with vLLM. The run with prompts in chunks of 100 tokens
(`compare-0.8b-chunk100.out`) split no prompt, so it checked nothing beyond `compare-0.8b.out`. The prompt's linear attention token by token instead of in chunks
(`compare-*-recurrent.out`) gives 12 and 11 of 16 (0.8B) and 10 and 11 of 16 (9B). The 9B completes "The capital of France is" with " Paris." (`generate-9b.out`).
Qwen3.5-9B's fourth prompt parts from vLLM at its second token in both of the engine's ways, where vLLM's two runs agree on all 64.

**Weights in GPU memory, Qwen3.5-9B** (the text model; its 348 vision-tower and MTP tensors are not loaded by any of the three):

| | weights | of vLLM bf16's |
| :--- | ---: | ---: |
| vLLM bf16 (`Model loading took`, run 2) | 16.8 GiB | 100% |
| vLLM with Glyd (run 2) | 12.0 GiB | 71.4% |
| glyd-engine (run 1; every matrix packed, the embedding and the LM head too; the norms, the convolution, `A_log`, `dt_bias` and the 32-row b and a products as the checkpoint has them) | 11.22 GiB | 66.8% |

The engine's 11.22 GiB is 67.3% of the checkpoint's 16.68 GiB for the same tensors. Its whole process held 12.46 GiB at 4,096 positions (states 0.048 GiB, cache 0.125 GiB,
workspaces 0.68 GiB); vLLM takes the share of the GPU it is given (`gpu_memory_utilization` 0.94) whatever the request needs.

**One user, Qwen3.5-9B, a prompt of 1,024 random tokens and 256 more** (the engine's `bench`, vLLM's `tools/ref_speed.py`: the same prompt; medians). **Two pods and two
boards: not one run's comparison.**

| | pod, board | tree | time to the first token | decode | tokens a second |
| :--- | :--- | :--- | ---: | ---: | ---: |
| glyd-engine (prompt's linear attention in chunks) | run 1: 48 GB board, driver 595.91 | 05379a9d | 141.8 ms | 14.36 ms | 69.6 |
| glyd-engine (token by token) | run 1: 48 GB board, driver 595.91 | 05379a9d | 145.9 ms | 14.37 ms | 69.6 |
| vLLM 0.30.0 bf16 | run 2: 24 GB board, driver 580.159 | cb2950d6 | 110.0 ms | 17.82 ms | 56.1 |
| vLLM 0.30.0 with Glyd | run 2: 24 GB board, driver 580.159 | cb2950d6 | 133.9 ms | 12.90 ms | 77.5 |

The job's vLLM speed script failed on run 1 (a vision model keeps its vocabulary in its text config) and the engine did not load on run 2. On these numbers the engine decodes
1.24 times as fast as vLLM bf16 and at 0.90 times vLLM with Glyd's rate, and its first token takes 1.29 and 1.06 times as long: **it does not meet the speed bar**, and `GPUS`
stays empty.

## Not measured here

- **The code as merged, end to end** (above). Run 2's engine refused to load ("fewer matrices than the layers take"): the merged host code (PR #18) keeps b and a as separate
  tensors instead of one 64-row pack, which the loader then counted on. The loader now takes each layer's matrices by name (`layer_mats`, with a host test over the 2026 dense
  configs: Qwen3.5-0.8B to 27B, Qwen3.6-27B, Qwen3.8-27B).
- **Long contexts against vLLM**: the job now gives vLLM and the engine three long prompts (the tree's README.md's first 1,030, 2,100 and 3,500 tokens), and the 0.8B's chunks of
  100 tokens split them.
- The hidden states against transformers layer by layer (`tools/ref_dump35.py`): it needed `accelerate` on run 1 (the job now installs it) and the engine's dump on run 2.
- The decode step's profile by kind of work (`bench --profile`), 8 to 256 users, other GPUs, Qwen3.5-27B.

## Files

`run1/`, `run2/`: `summary.txt` (the job's), `steps.txt`, `machine.txt`, `env.txt`, `COMMIT`, `kcheck35.out`, the vLLM references (`ref-*.json`: the prompts' ids, vLLM's tokens
and logprobs as one batch and one prompt at a time), the engine's tokens (`tokens-*.json`) and their comparisons (`compare-*.out`), the speed runs (`bench-9b*.out`,
`speed-9b-*.json`/`.out`), `smi.csv` (nvidia-smi each second). The pods' cost: about $0.50 (22 and 15 minutes of an RTX 4090 at about $0.78 an hour). The job pins uv 0.12.21
and Rust 1.98.1.
