# Large 2026 mixtures of experts over 4 H100s: bf16 against Glyd 0.29.0, vLLM 0.30.0 (2026-10-06)

**Status: not measured.** `glyd-gpu 0.29.0` reached PyPI at 04:49 UTC and the 4 x H100 job (`job-4xh100.sh`: Qwen3.5-122B-A10B, then
NVIDIA-Nemotron-3-Super-120B-A12B-BF16) ran twice; neither run reached a measurement:

- **Try 1 (04:53 UTC, pod up about 2 min):** `glyd[vllm]==0.29.0` did not resolve on the pod ("there is no version of glyd-gpu[vllm]==0.29.0").
  Root cause: PyPI's simple index is served with `cache-control: max-age=600`; the wheels were uploaded at 04:49:25 and the pod's CDN edge still
  served the index from before them at 04:54. The metadata itself is right (`glyd-gpu 0.29.0` has the `vllm` extra, `vllm>=0.30,<0.31`). The second
  try waited past the 10 minutes and installed. The failed install left both models' downloads running past the job's end (35 of 49 and 38 of 57
  files when the pod ended); `big_moe.sh` now stops its download on any exit. `results-4xh100-try1/`.
- **Try 2 (05:01 UTC, pod up about 16 min):** vLLM 0.30.0, glyd 0.29.0 and glyd-gpu 0.29.0 installed; the checkpoint downloaded (250,196,367,179 bytes
  in 874 s, about 0.29 GB/s on this host); `check_vllm.py --brief --tp 4` started its first run (bf16, CUDA graphs) at 05:16:57
  (`results-4xh100-try2/qwen35-122b/check/Qwen3.5-122B-A10B-tp4-bf16.log`, its last line). The pod was deleted at
  about 05:17 when the owner stopped the big-MoE work to save the account's last Runpod credits (the launcher now refuses `bigmoe` pods). No
  "Model loading took", KV cache, startup, chat, bench or check result exists. `results-4xh100-try2/`.
- Nemotron-3-Super was not reached in either try.

Spend: the two pods' ledger 163 s + 1,071 s at USD 13.96 an hour = USD 4.79 (the ledger adds 2 minutes a pod; about USD 3.85 of uptime).

To run it later: **refused today.** The launcher refuses `bigmoe` pods (the owner's block list) and every H100, H200, B200 and A100 pod; a rerun
needs the owner to lift both. On this host the download alone took 15 minutes, so the 122B (check, bf16, Glyd) needs about 70 minutes and
Nemotron about 40 more: about 110 minutes, so `MAX_MIN=120` below, and about USD 26 to 28 for both at 4 x USD 3.49, over this fix's USD 25 cap
(the owner's approval for the overage, or one pod per model: `job-122b.sh` alone is about USD 17).

**The Qwen3.5-397B-A17B job on 8 x H100 was dropped by the owner**: no pods of more than 4 GPUs (the launcher refuses them). Its row below stays
as a projection only. A "fewer GPUs" case on a 4-GPU pod needs the owner's approval first; two that would show it: Qwen3.5-397B-A17B on 4 x B200
(bf16's 739 GiB is more than 4 x about 179 GiB; Glyd's 557 GiB 12-bit or 500 GiB tiered fits, if the plugin runs on sm_100), or
Qwen3.5-122B-A10B on 4 x L40S (bf16's 228 GiB is more than 4 x 44.4 GiB; tiered is 38.7 GiB a GPU, and auto picks tiered on Ada; at 0.9 utilization, 40 GiB a GPU, that leaves
about 1.3 GiB a GPU for the KV cache and activations, so it may not start at 32,768 tokens).

Nothing in this folder is a measurement until the tables below are filled from the runs' logs.

## The plan, from the checkpoints' bytes

Bytes from the safetensors headers on the Hub (the 2026-10-06 tensor inventory: what Glyd's vLLM plugin packs under vLLM 0.30.0 and what it leaves),
per GPU at tensor parallel N, against an H100 80GB HBM3's 79.65 GiB (81,559 MiB). "Loaded" leaves out what vLLM does not load (the MTP head).
These are projections; the measured numbers replace them.

| Model | bf16 loaded (GiB) | Glyd packs (GiB of it) | GPUs | bf16 a GPU | Glyd 12-bit a GPU | Glyd tiered a GPU | bf16 fits? |
| :-- | --: | --: | --: | --: | --: | --: | :-- |
| Qwen/Qwen3.5-122B-A10B | 228.3 | 226.5 | 4 x H100 | 57.1 | 43.1 | 38.7 | yes (14.6 GiB a GPU left at 0.9) |
| Qwen/Qwen3.5-397B-A17B (dropped, not run) | 739.1 | 736.7 | 8 x H100 | 92.4 | 69.6 | 62.5 | **no**: 739 GiB is more than the 8 GPUs' 637 GiB |

- **Qwen3.5-122B-A10B, 4 x H100.** Both fit; Glyd leaves about twice bf16's room for the KV cache. Neither fits 2 GPUs (Glyd's smallest is 77 GiB a GPU at TP 2).
  The saturated bench (128 prompts of 1,256 tokens) fits in either cache: Qwen3.5 keeps a KV cache for its 12 full-attention layers only
  (2 KV heads, head 256), on the order of a million tokens a GPU for both. So the bench measures Glyd's speed against bf16's, not the extra KV room.
- **Qwen3.5-397B-A17B, 8 x H100 (dropped by the owner).** Only Glyd would fit. bf16 needs a second 8-GPU machine (16 H100s at tensor parallel 8 x
  pipeline parallel 2; by bytes at least 10 to 12 GPUs, e.g. tensor parallel 4 x pipeline parallel 3 at about 61.6 GiB a GPU), or 8 H200s.
  On an H100 Glyd's auto layout is the 12-bit one: `best_layout` sizes the room as the GPU's whole memory less 2 GiB and does not read
  `--gpu-memory-utilization`, which would leave about 2 GiB a GPU at 0.9 (an owner decision, listed below).

Not run, with the reason (vLLM 0.30.0's model files and the plugin's rules, read from source). These are coverage gaps against the goal of
33% on every model, about 1.4 TB of 2026 weights Glyd does not compress today (Nemotron-3-Super's 210 GiB of experts, about 225 GB, + MiniMax-M3's
854 GB + Qwen3.8-Flash-Next's 360 GB = about 1,440 GB): ReLU-squared experts, `swigluoai` experts and the Qwen4Exp per-layer embedding table. Closing them belongs to the packing work (coverage-2026), not to this benchmark:

- **nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16** (230 GiB): its experts use a ReLU-squared activation without a gate; the plugin packs
  SiLU-gated experts only (`GlydMoEMethod.unsupported`: "activation ..., not SiLU"), so 210 GiB of experts stay bf16 and Glyd saves about 2%.
  `job-4xh100.sh` runs it second (owner-approved, 2026-10-06) to measure that under 0.29.0; not reached (see Status).
- **MiniMaxAI/MiniMax-M3** (854 GB bf16, MiniMaxM3SparseForConditionalGeneration): its experts' activation is `swigluoai_uninterleave`
  (vLLM `models/minimax_m3/nvidia/model.py`), not SiLU, so the experts stay bf16 under the plugin and the model does not fit 8 x H100 with or
  without Glyd.
- **Qwen/Qwen3.8-Flash-Next** (360 GB, Qwen4ExpForConditionalGeneration): vLLM 0.30.0 does not load it with `--quantization glyd` at all:
  `Qwen4ExpPLEEmbeddingMethod.from_quant_config` (`models/qwen4_exp/nvidia/ngram_embedding.py`) raises
  `NotImplementedError("Qwen4Exp PLE embedding does not support quantization config GlydConfig")` for any quantization config but FP8 and ModelOpt.

## What each job measures

`big_moe.sh` (one model, all the machine's GPUs): uv; vLLM 0.30.0 and `glyd[vllm]==0.29.0` from PyPI; the model downloaded meanwhile
(`huggingface_hub` with hf_xet / hf_transfer). Per server: `vllm serve MODEL --tensor-parallel-size N --gpu-memory-utilization U --max-model-len 32768 --language-model-only`
(Glyd: `--quantization glyd`, layout and kv auto), the seconds until it answers, every GPU's "Model loading took", the KV cache's tokens and
maximum concurrency at 32,768 tokens as vLLM logs them, nvidia-smi's memory, one greedy chat answer (thinking off), then
`vllm bench serve --dataset-name random --random-input-len 1000 --random-output-len 256 --ignore-eos --request-rate inf --num-prompts 128 --seed 0`
after 8 warm-up prompts (seed 1000). The Glyd servers run with `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`; the bf16 server keeps vLLM's shared-experts stream (its default), so on models
with shared experts (Qwen3.5-122B-A10B) bf16 runs with that overlap and Glyd without it. Bit for bit: on the 122B, `gpu/vllm/check_vllm.py --brief --tp 4` (run first, before the servers; bf16 with CUDA graphs and eager,
Glyd auto with every pack decoded to its weights bit for bit and every layer's product checked, exact mode's tokens and logprobs bf16 eager's
bit for bit). If it times out the summary says TIMEOUT, not a count of passes.
`VERIFY=1` (not used by `job-122b.sh`) adds `GLYD_VERIFY=1`: its startup times include the check and are never compared with default startups.

## Commands

From this folder, with the Runpod launcher (one pod, deleted when the job ends or at `MAX_MIN`). The launcher is `tools/runpod/rp_run.sh` once
private PR #20 merges; until then the copy in the old session scratchpad (`$S/rp_run.sh`). It pulls the pod's `~/results` into
`$S/ceiling/results-$NAME`; copy that here as `results-4xh100/` (with `qwen35-122b/` and `nemotron3-super/` in it).

```
GPU="NVIDIA H100 80GB HBM3" RP_GPUS=4 RP_DISK=400 MAX_MIN=120 NAME=bigmoe-4xh100 UPLOAD="big_moe.sh ../../../gpu/vllm/check_vllm.py" bash "$S/rp_run.sh" job-4xh100.sh
```

## Results

None yet (see Status).

## Files

- `big_moe.sh`: the job; `job-4xh100.sh`: both models on one 4 x H100 pod (one model on the disk at a time); `job-122b.sh`: the 122B alone.
- `results-4xh100-try1/`, `results-4xh100-try2/`: the two stopped runs, per model machine, environment, download and summary; try 1 also has the
  launcher's log (`launcher.log`, `job.log`); try 2 has the check's log so far (`qwen35-122b/check/`) and no launcher log (the pod was deleted from outside).
- After the run: `results-4xh100/qwen35-122b/` and `results-4xh100/nemotron3-super/` (machine.txt, env.txt, summary.txt, check.txt, per server `serve.log`,
  `lines.txt`, `startup.txt`, `memory.csv`, `smi.csv`, `chat.json`, `bench.json`).

## Owner decisions

- `best_layout` sizes the room as the GPU's whole memory less 2 GiB and ignores `--gpu-memory-utilization`; on Qwen3.5-397B-A17B over 8 x H100 it
  would pick 12-bit with about 2 GiB a GPU left at 0.9. Keep it, or size against the utilization?
- A 4-GPU "fewer GPUs" run (4 x B200 or 4 x L40S, above) within the USD 100 cap: approve or not.
