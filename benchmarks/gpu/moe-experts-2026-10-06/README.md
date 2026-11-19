# Mixture-of-experts activations under the vLLM plugin (2026-10-06): the GPU checks, NOT RUN

Status: **no GPU result.** The batched check of PR #27 (clamped SwiGLU experts kept bf16 outside exact mode) and PR #29 (experts
with any activation vLLM's experts apply, packed as the products' sums then vLLM's own activation) did not run:

| pod | GPU | what happened | cost |
| :-- | :-- | :-- | --: |
| `moex1` | H100 PCIe | refused by `rp_run.sh` before creation: no data-center GPU pods (`moex1-refused.log`) | $0 |
| `moex2` | L40S | created 05:23 UTC; deleted at ~05:28 while the library built and the slices downloaded, when bf16 GPU work was paused for the quantized sessions (`rp_budget/ALLOW`); the ledger has it at 313 s | $0.10 |

Not retried (the launcher's rule). What did run on the pod before it went: the five slices were cut from the real checkpoints and
their bytes checked against the source (`moex2-stopped/log/dl-*.txt`): Nemotron-3.5-Lightning layers 0-6 8.82 GiB, gemma-4-26B-A4B-it
layers 0-5 10.51 GiB, Step-3.5-Flash layers 0 and 43 11.04 GiB; Ling-3.0-flash layer 35 (7.20 GiB) and MiniMax-M3 layers 0-3
(20.26 GiB) were still downloading.

## The plan (`job.sh`: one L40S, about 85 minutes, ~$1.6)

Slices (`gpu/fp8/make_slice.py`, real weights by HTTP range; `--pick` takes a layer past the first ones, renumbered, with the config's
per-layer entries for it), each through `gpu/vllm/check_vllm.py --brief` (bf16 graphs and eager, Glyd auto with every pack and product
checked, exact eager against bf16 eager bit for bit) and `gpu/vllm/moe_acts.py --no-tps` (one MoE layer by tokens a step, 1 to 2,048:
bf16's method, Glyd's grouped products, the decoded route, the route Glyd takes, and the rule "no slower than bf16 at every T"):

| slice | experts' activation | for |
| :-- | :-- | :-- |
| inclusionAI/Ling-3.0-flash layer 35 (MLA) | SiLU, swiglu_limit 4 | PR #27 (pr1 tree: kept bf16 in fused mode, exact bit for bit) and PR #29 |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 layers 0-6 | relu2 without a gate | PR #29 |
| google/gemma-4-26B-A4B-it layers 0-5 | GELU tanh | PR #29 |
| stepfun-ai/Step-3.5-Flash layers 0, 43 | swiglustep (limit 7) | PR #29 |
| MiniMaxAI/MiniMax-M3 layers 0-3 | swigluoai_uninterleave (limit 7, alpha 1.702, beta 1) | PR #29 |

GLM-5.3-Flash-BF16 and Hy4-preview (the clamp's first two models) need Hopper (their sparse attention), so the clamp is checked on
Ling 3.0's layer 35, whose routed experts carry the same `swiglu_limit` path. The whole gemma-4-26B-A4B-it (51.6 GB) and
Nemotron-3.5-Lightning (65.8 GB) runs with one-user and 8-user tok/s need an 80 GB GPU in bf16: not possible under the current rule.

```sh
# moex_src.tar = pr1/ (git archive of moe-swiglu-clamp: glyd-gpu-py) + pr2/ (moe-activations: glyd-gpu-py gpu) + COMMIT
GPU="NVIDIA L40S" MAX_MIN=110 NAME=moex2 RP_DISK=150 UPLOAD="moex_src.tar" bash rp_run.sh benchmarks/gpu/moe-experts-2026-10-06/job.sh
```

## Which 2026 bf16 models have clamped SwiGLU experts (config.json, 2026-10-06)

| model | layers with clamped routed experts | limit |
| :-- | :-- | --: |
| zai-org/GLM-5.3-Flash-BF16 | every MoE layer | 10 |
| tencent/Hy4-preview | every MoE layer | 10 |
| LGAI-EXAONE/K-EXAONE-2.0-750B-A37B | 62-77 (`swiglu_limits`) | 7 |
| inclusionAI/Ling-3.0-flash | 35-41 (`expert_swiglu_limit_list`) | 4 |
| stepfun-ai/Step-3.5-Flash | 43-44, as `swiglustep` (`swiglu_limits`) | 7 |
| MiniMaxAI/MiniMax-M3 | every MoE layer, as `swigluoai_uninterleave` | 7 |

vLLM 0.30 keeps the first four SiLU and applies the clamp in its activation (`silu_and_mul_with_clamp`: gate clamped above, up both
sides, on the first product's bf16 sums); v0.29.0's plugin packed them and its grouped products applied plain SiLU below the
decoded-route threshold.
