# v0.29.0 on an RTX 4090: the release acceptance and the 2026 bf16 sweep (2026-10-06, stopped early)

One Runpod RTX 4090 pod (driver 595.91.07, Ubuntu 24.04, 32 CPUs), started 05:02 UTC with `jobA.sh`, Glyd as released:
`glyd 0.29.0`, `glyd-gpu 0.29.0` (PyPI), vLLM 0.30.0, torch 2.13.0, transformers 5.18.0, installed by tag v0.29.0's `scripts/install.sh`.

**The pod was deleted from outside this run at about 05:27 UTC**, 25 minutes in, while the sweep was on its sixth model. It was not
`rp_run.sh`'s doing: the session-wide Runpod budget files were changed at 05:18-05:27 (the account balance was about $49) and now
refuse new pods for this work ("bf16 GPU work is paused by the owner while the quantized sessions run"). So: both acceptance runs
are partial (see below), 5 of the 68 rows of `listA.txt` were measured, and the RTX 5090 run (`../rtx5090-v029-2026-10-06/`)
did not start. Spend on this pod: about $0.33 (0.74 $/h).

## 1. The release acceptance (gpu/vllm/acceptance.sh of private tag v0.29.0, d7f53f4a)

As run (as a user that is not root, `/usr/local/cuda*` moved to `/opt/hidden`; `logs/acc-*.out`, `logs/acc-*/summary.txt`, `logs/acc-*/logs/`):

    bash acceptance.sh --work ~/w-24gb  --flow glyd --host --install-url https://raw.githubusercontent.com/surya-koritala/Glyd/v0.29.0/scripts/install.sh --model Qwen/Qwen3.5-9B --webui uvx
    bash acceptance.sh --work ~/w-4080s --flow glyd --host --install-url (the same) --model Qwen/Qwen3.5-9B --webui uvx --card 4080s

| Run | ok | FAIL | Ended |
| :-- | --: | --: | :-- |
| 24 GB (the whole RTX 4090) | 40 | 7 | killed (exit 137) at the Ctrl-C check, by the check itself (below) |
| `--card 4080s` (14,828 MiB free, the owner's 16 GB card) | 10 | 8 | exit 1: Glyd refused Qwen3.5-9B on this card |

What failed, and why:

- **Real, in the released v0.29.0:**
  - `glyd serve Qwen/Qwen3.5-9B` starts vLLM with **no tool-call parser and no reasoning parser** (the server log's non-default args have
    neither). `preflight.parsers()` reads `m.text["model_type"]` first, which for Qwen3.5 is `qwen3_5_text`, not `qwen3_5`, so the
    Qwen3.5 rule never matches. Effects measured: the answer's thinking comes back inside `content` ("Thinking Process: ..."),
    a request with `tools` gets HTTP 400 (`logs/acc-24gb/logs/chat.log`), and Open WebUI's chat (it offers its built-in tools)
    gets HTTP 400 (`webui-uvx.check.log`). Open WebUI's login, admin account, model list and CORS checks passed.
    Private PR #21 (`run-2026-families`) changes `parsers()` to look at both model types; it was open, not merged, when this ran.
  - `--card 4080s`: `glyd run Qwen/Qwen3.5-9B` is refused: "needs about 16.3 GB of GPU memory with Glyd (14.3 GB of weights and room
    for a 4,096-token chat); your GPU has 15.7 GB free", with Qwen3.5-4B suggested. So the default model of v0.29.0 does not run on
    the owner's 16 GB RTX 4080 SUPER (as emulated: 14,828 MiB free).
  - `install.sh` and `glyd doctor` end with "Next: glyd run Qwen/Qwen3-8B" (a 2025 model).
- **The check, not Glyd:**
  - "no plain message from glyd run for a prompt longer than the window": the check's prompt is 60,000 words; Qwen3.5-9B's window
    under Glyd here is 189,440 tokens, so the prompt fits and glyd run (rightly) used the running server. The check needs a prompt
    longer than the window it reads from `/v1/models`.
- **This harness, not Glyd** (fixed in `jobA2.sh`, which has not run):
  - Ctrl-C: the check looks for leftover processes with `pgrep -f "EngineCore|vllm"` and kills them; this run's own wrapper had
    `gpu/vllm` in its command line, so it was reported as "left running" and killed, which killed the acceptance run (exit 137).
    `jobA2.sh` unpacks the tree as `gpu/gate`.
  - "install.sh did not install uv": the pod image has `/usr/local/bin/uv`. "no gcc here, and install.sh added no ziglang": the pod has
    gcc. `jobA2.sh` moves uv, uvx and gcc/cc out of the way during the acceptance, as the CUDA toolkit is, and puts them back after.
  - "packages installed at versions that install.sh's list does not give": `scripts/install_constraints.py` was not uploaded (only
    `gpu/vllm` was), so the list was empty. `jobA2.sh` uploads `scripts/` beside it.

Passed in the 24 GB run, among others: no nvcc; a user's own `glyd` program left alone; the update path and PATH lines; `glyd doctor`
("Ready: glyd run Qwen/Qwen3.5-9B"); `glyd run` (189,440-token context, compiled, 92% of GPU memory); `glyd serve` up in 66 s; the chat
page and its CSP; the Origin/Host guards (11 checks); nohup/SIGHUP; streamed turns; escape sequences; the terminal chat's message past
the window; the API key by flag and by `VLLM_API_KEY` (not on any command line or log; 401 without, 200 with); SIGTERM.

## 2. The 2026 bf16 sweep (`sweep.sh` on `listA.txt`)

For each model: `GLYD_KV=off vllm serve MODEL --quantization glyd` and plain `vllm serve MODEL`, both from Glyd's tool environment and
both with `--enforce-eager --max-model-len 8192 --max-num-seqs 32 --dtype bfloat16` (the same room asked of both; eager so that each
start is about a minute; the compiled numbers are the acceptance's and the speed runs'), one greedy chat answer each (`chat1.py`).
Weights are vLLM's "Model loading took"; KV tokens are vLLM's "GPU KV cache size" at its default 90% of GPU memory.
Rows: `logs/sweep.tsv`; logs: `logs/sweep/<model>/` (`.log` files saved as `.txt`). Table: `python3 sweep_table.py listA.txt logs/sweep.tsv`.

| Model | Glyd weights GiB | bf16 weights GiB | Saved | Glyd KV tokens | bf16 KV tokens | Glyd answer | bf16 answer | Note |
| :-- | --: | --: | --: | --: | --: | :-- | :-- | :-- |
| Qwen/Qwen3.5-9B | 12.68 | 17.66 | 28.2% | 157,882 | 43,938 | Paris | Paris | tiered layout |
| google/gemma-4-12B-it | | | | | | | | Glyd: the plugin stops on Gemma 4's per-layer config (transformers' `AmbiguousGlobalPerLayerAttributeError`: `num_key_value_heads`, in `vllm_plugin._linear_bytes`); bf16: out of GPU memory (22.3 GiB of weights on 24 GB) |
| google/gemma-4-E4B-it | | 15.08 | | | 167,508 | | Paris | Glyd: the same Gemma 4 error (`head_dim`) |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | | 7.47 | | | 356,937 | | Paris | Glyd: the plugin's own fit check refused it: "the model's weights take about 33.6 GiB a GPU packed in the mma layout, at least 22.8" (the checkpoint is 7.9 GB) |
| ibm-granite/granite-4.2-8b | 11.18 | 16.42 | 31.9% | 65,104 | 31,936 | Paris | Paris | tiered layout |
| Qwen/Qwen3.5-4B | | | | | | | | interrupted (the pod was deleted while Glyd's server was loading) |

Not run: the other 62 rows of `listA.txt`: 41 that vLLM 0.30.0 has a class for (one of them `llm-jp-4-8b-instruct`, the same shape as
`llm-jp-4.1-8b-thinking`) and 21 configs-only rows for architectures it has none for. Five gated rows are not in the list (no Hugging Face token on pods): facebook/MobileMoE-S/M/L-SFT,
CohereLabs/tiny-aya-global, swiss-ai/Apertus-v1.5-8B.

## Files

- `jobA.sh` the pod's job as run; `jobA2.sh` the same with the harness fixes above and the sweep on the rows left (`LISTFILE`), not run
- `sweep.sh`, `chat1.py`, `sweep_table.py`, `listA.txt` (REPO MODE BYTES: `full`, or `config` where vLLM 0.30.0 has no class)
- `logs/` what the pod sent back before it was deleted (its `job.log` was lost when the pod went: the launcher's copy was overwritten)

Launched with the session's `rp_run.sh`: `GPU="NVIDIA GeForce RTX 4090" MAX_MIN=150 RP_DISK=200 NAME=m29a UPLOAD="accsrc.tar sweep.sh chat1.py listA.txt" bash rp_run.sh jobA.sh`
(`accsrc.tar` = `git archive v0.29.0 gpu/vllm`).
