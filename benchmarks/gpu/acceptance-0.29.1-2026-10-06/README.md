# v0.29.1 release acceptance on an RTX 4090 (2026-10-06)

## Served installer, complete flow

`pod3-served/`: the live `curl -LsSf https://getglyd.com/install.sh | sh` flow on 2026-10-06, after the v0.29.1 site deployment.
**55 checks passed, no failures.** The served installer equals the public v0.29.1 tag byte for byte (sha256
`79ea9e3dbd25a71aee4dda61e265b9d63dc5c628e88ce0f3909f2fc06c96a152`). Published `glyd==0.29.1` and `glyd-gpu==0.29.1` were installed
from PyPI; the 199 pinned packages matched the installer. The run used Qwen/Qwen3.5-9B, a non-root user, no CUDA toolkit, no gcc and no preinstalled uv.

The gate source is private commit `dc56ad719140f1dac1ea1b971638b4d49e2ebe5f`; its prompt-refusal checks read the served context before making
an oversized prompt. The binaries under test are the published release. Commands and setup: `job_pod3.sh`; exact results: `pod3-served/acc-24gb/summary.txt`
and `pod3-served/acc-24gb.out`. All installation, doctor, run, serve, streaming, reasoning, tool-call, Open WebUI, API-key, signal and uninstall checks passed.

The pod ran about 24 minutes and was deleted when the job finished. Recorded cost including the ledger's polling allowance: about $0.31,
inside the approved $0.93 maximum. The earlier two pods below cost about $0.44; the release acceptance total is about $0.75.
Logs whose original extension was `.log` are retained as `.txt`. Earlier attempts and their failures below are preserved.

## Earlier runs

The published release as a user installs it: `install.sh` of public tag v0.29.1 (raw GitHub, sha256 `79ea9e3d...6a152`, `GLYD_VERSION` 0.29.1),
which installs `glyd[vllm]==0.29.1` and `glyd-gpu==0.29.1` from PyPI ([published-artifacts.txt](published-artifacts.txt): the wheels and their sha256
as PyPI lists them), vLLM 0.30.0, torch 2.13.0, transformers 5.18.0. Two Runpod RTX 4090 pods (secure cloud, Ubuntu 24.04), each with the CUDA toolkit
(`/usr/local/cuda*`), uv/uvx and gcc moved out of the way first, the commands run as a user that is not root (`acc`), in a stripped environment.
The script is `gpu/vllm/acceptance.sh` of private tag v0.29.1 (c6ad9ac7), unpacked as `gpu/gate` beside its README and `scripts/install*`
(`accsrc.tar` = `git archive c6ad9ac7 gpu/vllm/acceptance.sh gpu/vllm/README.md scripts/install_constraints.py scripts/install.sh`, sha256 `06f11935...6e2`).

| Run | Command | Result |
| :-- | :-- | :-- |
| Pod 2 (`pod2-acceptance/`) | `acceptance.sh --flow glyd --host --install-url https://raw.githubusercontent.com/surya-koritala/Glyd/v0.29.1/scripts/install.sh --model Qwen/Qwen3.5-9B --webui uvx` | **53 checks passed, 1 failed** (the check's own prompt, below) |
| Pod 1 (`pod1/gemma/`) | the same install line in a fresh home, then `glyd run google/gemma-4-12B-it --prompt 'What is the capital of France? Answer in one word.'` | **passed, 3 of 3**: glyd 0.29.1 / glyd-gpu 0.29.1 installed; exit 0, answer `Paris`, no traceback; one start |
| Pod 1 (`pod1/acceptance-attempt/`) | the acceptance as on pod 2 | stopped at the install: PyPI's index did not list glyd 0.29.1 yet (below); run again on pod 2 |

## Not `--served`

`getglyd.com/install.sh` still served v0.29.0's script (`GLYD_VERSION="${GLYD_VERSION:-0.29.0}"`, sha256 `f32be245...a156`) at both runs (15:56 and 16:07 UTC;
`*/served.txt`). With `--served` the acceptance would have installed 0.29.0 and failed its byte-for-byte check against the tag's file, so the job ran without
`--served` (it adds `--served` only where the two files are the same bytes). The site's install.sh is updated with the release's site entry; the
`--served` run belongs after that.

## The one failure (pod 2), the check's prompt and not Glyd

    FAIL no plain message from glyd run for a prompt longer than the window: ...done thinking.  rc=0

The check pipes 60,000 words (`yes word | head -n 60000`) into `glyd run`, which attached to the running `glyd serve`. That prompt is about 60,000 tokens
(the server's log has `Avg prompt throughput: 6000.8 tokens/s` over its 10-second interval: `logs/serve-20261006-161844.txt`), and Qwen3.5-9B's window
under Glyd 0.29.1 on this card is 183,296 tokens, so the prompt fits and `glyd run` rightly answered it (exit 0). The same check failed the same way in
the v0.29.0 acceptance (`../rtx4090-v029-2026-10-06/`: window 189,440). It needs a prompt longer than the window it reads from `/v1/models`, as the
next check (the terminal chat's conversation past the window, which passed) already does.

## What passed on pod 2

- **The install**: no nvcc; a `glyd` of the user's own in `~/.local/bin` left alone; `install.sh` 63 s, exit 0, ending with `glyd doctor`'s
  "Ready: glyd run Qwen/Qwen3.5-9B"; uv 0.12.21 by its own installer; no shell startup file touched; resolved `glyd==0.29.1`, vLLM 0.30.0, torch 2.13.0,
  transformers 5.18.0, flashinfer-python 0.6.18.post1, ziglang 0.16.0 (no gcc); no pre-release; **all 199 packages at the versions install.sh lists**;
  the update (install.sh again, `~/.local/bin` off PATH), a `glyd` ahead on PATH said by install.sh and `glyd doctor`.
- **`glyd run Qwen/Qwen3.5-9B --prompt`**: `Paris`; settings a 183,296-token context, CUDA graphs, 92% of GPU memory, tool calls `qwen3_coder`,
  thinking apart `qwen3`; Ready in 4m22s with the 19.3 GB download; vLLM: `Model loading took 12.01 GiB`, `GPU KV cache size: 217,469 tokens`, one start.
- **`glyd serve`** up in 103 s; the chat page (CSP `default-src 'none'`); `/nope` 404; the Origin/Host guard (11 checks); SIGHUP under nohup;
  the OpenAI API: two streamed turns (`Paris`, `Berlin`), a tool call (`get_weather`, `{"city": "Paris"}`), the thinking apart (1,772 characters,
  the answer without `<think>`), a conversation past the window refused with HTTP 400 naming 183,296; escape sequences: none in glyd run's output;
  the terminal chat's message for a conversation past the window.
- **Open WebUI by uvx** (11 GB into uv's cache): login on, the first account the administrator, the model listed, CORS limited to its own
  addresses, a chat (`Paris`) and a tool call in the stream (`get_current_timestamp`).
- **A key**: as `-- --api-key` and as `VLLM_API_KEY` (the README's line): ready, the key on no other command line and in no log, `/v1` 401 without
  and 200 with, `/health` open as the note says.
- **Stopping**: SIGTERM; 4 server logs with no traceback and no allocator out-of-memory warning; Ctrl-C while loading, once (exit 130 in 3 s) and
  twice (exit 130 in 0 s), no vLLM process left; the engine killed under a terminal chat's answer (a plain message, glyd serve exited);
  `uv tool uninstall glyd`. The GPU at 1 MiB after the run.

## Gemma 4 (pod 1)

`glyd run google/gemma-4-12B-it --prompt` on v0.29.1 as installed (v0.29.0 stopped with `AmbiguousGlobalPerLayerAttributeError`): "16.6 GB on the GPU
with Glyd, instead of 23.9 GB"; settings a 124,928-token context, CUDA graphs, 92% of GPU memory, tool calls `gemma4`, thinking apart `gemma4`; Ready in
3m08s with the 24.0 GB download; answer `Paris`, exit 0 after 213 s; vLLM: `Model loading took 15.67 GiB`, `GPU KV cache size: 147,784 tokens`; one
start; no traceback before the server's shutdown lines. On this pod `/usr/bin/uv` (the image's own) was not moved, so install.sh used it instead of
installing uv 0.12.21; the packages are the same pinned list either way (pod 2 moved it).

## Pod 1's acceptance attempt

Started at 15:56:36 UTC, two minutes after glyd 0.29.1's upload (15:54:20): uv answered "Because there is no version of glyd[vllm]==0.29.1 ...
your requirements are unsatisfiable" (`pod1/acceptance-attempt/logs/install.txt`): PyPI's index had not yet listed it. The Gemma install on the same
pod, 7 s later, resolved it. The acceptance was run again on pod 2 (`job_pod2.sh`: the acceptance alone, `/usr/bin/uv` and uvx moved out too).

## Spend

`rp_run.sh`, `GPU="NVIDIA GeForce RTX 4090"`, $0.74/h: pod 1 `r291-acc` 7 min, pod 2 `r291-acc2` 29 min; about **$0.44** in all.

## Files

- `job_pod1.sh`, `job_pod2.sh`: the pods' jobs as run (`UPLOAD=accsrc.tar`)
- `pod2-acceptance/`: `summary.txt` and `acceptance.out` (the acceptance's own output), `logs/` (its logs; glyd's server logs `run-*`, `serve-*`),
  `job.txt`, `machine.txt`, `hidden.txt` (what was moved out), `served.txt`, `accsrc.sha256`, `ports-before.txt`
- `pod1/gemma/`: `install.txt`, `versions.txt`, `run.out`, `run.err`, the server's log `run-20261006-155736.txt`, `numbers.txt`, `summary.txt`
- `pod1/acceptance-attempt/`: the stopped attempt's summary and install log
- `.log` files are saved as `.txt` (the repo ignores `*.log`)
