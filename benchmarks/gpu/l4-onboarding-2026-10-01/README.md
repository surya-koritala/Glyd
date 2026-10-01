# `glyd run`, `glyd serve` and the installer, from nothing, on an L4 (2026-10-01)

What was run: [scripts/install.sh](../../../scripts/install.sh) and `glyd run`, `glyd serve`, `glyd doctor` of the `onboarding` branch,
by [gpu/vllm/acceptance.sh](../../../gpu/vllm/acceptance.sh), in a clean container (Ubuntu 26.04, a user that is not root, no CUDA
toolkit, no gcc) on the AWS dev machine: an NVIDIA L4 (24 GB, Ada, sm_89), driver 595.91.07 (CUDA 13.2). The install resolved vLLM
0.30.0, PyTorch 2.13.0 (CUDA 13.0), Triton 3.7.1, transformers 5.18.0, safetensors 0.8.0, tokenizers 0.23.2, pydantic 2.13.5 and,
where there was no gcc, ziglang 0.16.0 (`runs/*/logs/freeze.txt`); uv 0.12.21. The wheel is the branch's, built as release.yml builds it,
with the plugin of `vllm-plugin` 8d4e835 (the allocator and nvcc fixes). The L4 stands in for other cards where a process holds the rest of
its memory (the hog, [scripts/transcript.sh](scripts/transcript.sh), the acceptance script's own): `--card 4080s` leaves 14,828 MiB
free to the server (vLLM logs 14.48 GiB free at start, as the owner's RTX 4080 SUPER with a desktop does) and has the plugin read the
GPU as a GeForce Ada card; `--card 8gb` leaves 7.5 GiB.

## The runs (each PASSED; `runs/NAME/acceptance.out` is the whole output, `summary.txt` its summary)

| Run | Command | What it checked |
| :--- | :--- | :--- |
| `16gb-card` | `acceptance.sh --wheel W --card 4080s --webui all --pip-refusal` | install.sh, the resolved versions (no pre-release among the dependencies but opentelemetry's beta-only 0.66b0 packages), `glyd doctor`, `glyd run --prompt`, `glyd serve`, the page, the API, the thinking apart, a conversation past the window (API 400, `glyd run` by stdin, the terminal chat typed through a pty), Open WebUI by uvx, by Docker with host networking and by Docker on its own network with a key, the stop, 3 server logs, and the pip install with no compiler refused with the install command |
| `16gb-card-final-code` | `acceptance.sh --wheel W --card 4080s --webui none` | the same without the Open WebUI routes and the pip install, on the last commit's wheel |
| `24gb` | `acceptance.sh --wheel W --webui none` | the whole L4: a 40,960-token context |
| `8gb-card` | `acceptance.sh --wheel W --card 8gb` | Qwen3-4B refused with a model to try, and Qwen3-1.7B (the one named) run |
| `by-hand-vllm-serve-16gb-card` | `acceptance.sh --flow vllm --wheel W --card 4080s --webui uvx` | the README's by-hand `vllm serve` command as written |

In the 9 servers' logs of these runs (`runs/*/glyd-logs`, 8, and the by-hand run's `logs/server.log`): no allocator warning ("memory allocation
failed with OOM", "memory mapping failed with OOM") and no traceback. The first attempt of the 16 GB run stopped when the box's disk filled
(uv was copying its cache, and Open WebUI's uvx route is 7.1 GB of PyTorch): its Open WebUI and pip steps did not run, and it was run again
with uv's hardlinks (`16gb-card`).

## What `glyd run` chose, and what vLLM logged

| Free at start (glyd) | Chose | vLLM logged | `glyd run` ready in | `glyd serve` ready in |
| :--- | :--- | :--- | ---: | ---: |
| 23.7 GB | 92% (21.8 GB), 40,960 tokens (the model's own limit) | weights 11.38 GiB, KV cache 61,104 tokens | 61 s | 41 s |
| 15.7 GB | 62% (14.7 GB), 11,264 tokens | weights 11.39 GiB, KV cache 12,960 tokens, "Free memory on device (14.48/22.04 GiB)" | 78 s (71 s on the last commit's wheel) | 40 s |
| 8.1 GB, Qwen3-4B | refused: "needs about 8.4 GB of GPU memory with Glyd (5.9 GB of weights and room for a 4,096-token chat); your GPU has 8.1 GB free. Or try Qwen/Qwen3-1.7B, which needs about 5.1 GB" | | | |
| 8.1 GB, Qwen3-1.7B | 29% (6.9 GB), 24,576 tokens | weights 2.47 GiB, KV cache 33,184 tokens | 58 s | |

`calibration/` is where the constants of preflight.py come from: the weights `glyd run` counts from a config, and the memory vLLM keeps
besides them, against vLLM's own log lines ([scripts/calib.py](scripts/calib.py)). In the 13 logs of the settings as shipped (`a25-*` and the
runs above) the weights are within +0.11 GiB of vLLM's (0.00 for Qwen3-8B, over for the smaller models) and the KV cache tokens `glyd run`
predicts 3.6-17.9% under vLLM's: it never chose a context vLLM then refused. (The `a24-*` logs, with expandable segments, are against
the constants of today too.)

`calibration/a24-session.log` has the first session's eager and compiled servers
with Qwen3-8B on the whole L4, one to eight chats at once ([scripts/bench_conc.py](scripts/bench_conc.py)):

| | 1 at once | 4 | 8 | start |
| :--- | ---: | ---: | ---: | ---: |
| eager | 21.2 tokens/s in all | 82.3 | 161.2 | 47 s |
| compiled | 21.7 | 84.8 | 165.8 | 2 min 45 s |

compiled needed 2.15 GiB beyond the weights where eager needed 0.49-0.50 (`calibration/calib-a24.txt`). The same session ran the
allocator setting `expandable_segments:True` at a 16 GB budget: 24 allocator warnings in the load, where the default allocator
had none (`calibration/a24-server-logs`).

## A first run, as a terminal shows it

On a 16 GB card's memory, with no model cache and no uv cache; [scripts/transcript.sh](scripts/transcript.sh) drives the terminal through a
pty and [scripts/render.py](scripts/render.py) turns its bytes into the screen's last state. The install took 14 s here (uv fetched 8 GB at
600 MB/s: a home connection is slower), and `glyd run`, download and a chat, 331 s. In a terminal the download bar and the loading line
redraw in place, so they are not in the text below: `Downloading Qwen3-8B  [#############           ]  57%  9.3 GB of 16.4 GB  250 MB/s,
28s left`, and `Loading Qwen3-8B with Glyd: 12.2 GB instead of 16.4 GB... 36s (loading the weights, 3 of 5 parts)`, whose parenthesis
goes through reading the model's details, starting the engine, loading the weights and warming up. The first start took 1 min 44 s:
Triton builds its launchers once, here with ziglang, and the checkpoint was just downloaded.

`curl -LsSf https://getglyd.com/install.sh | sh` (this tree's install.sh and this wheel in its place, the package lines of uv's output left out):

```
==> Installing uv, which manages Python for Glyd (its own installer, astral.sh)
downloading uv 0.12.21 x86_64-unknown-linux-gnu
installing to /work/home/.local/bin
  uv
  uvx
everything's installed!
==> Installing /work/wheel/glyd-0.26.0rc2-py3-none-manylinux_2_28_x86_64.whl[vllm] and Python 3.12 (PyTorch and vLLM: several GB, a few minutes)
==> No C compiler found, and vLLM needs one: adding ziglang, a compiler from PyPI (no sudo)
Resolved 200 packages in 619ms
Prepared 200 packages in 9.44s
Installed 200 packages in 1.30s
Installed 1 executable: glyd
==> Checking this machine (glyd doctor)
 ok  Glyd             0.26.0rc2, Python 3.12 (/work/home/.local/share/uv/tools/glyd/bin/python)
 ok  GPU              NVIDIA L4: 23.7 GB, 23.7 GB free, compute capability 8.9
 ok  Driver           595.91.07, runs CUDA 13.2
 ok  PyTorch          2.13.0, built for CUDA 13.0
 ok  vLLM             0.30.0
 ok  Glyd GPU library libglyd_gpu_cuda13.so
 ok  C compiler       ziglang (a C compiler from PyPI), and Python.h: vLLM builds its Triton launchers with them
 ok  CUDA compiler    not found: not needed (glyd run turns FlashInfer's sampler off, which would compile)
 ok  Disk             60.2 GB free for models, in /work/hf/hub

Models that fit this GPU with Glyd (a chat as long as the GPU allows):
  Qwen/Qwen3-8B     12.2 GB on the GPU (bf16 16.4 GB): fits, a 40,960-token context
  Qwen/Qwen3-14B    21.5 GB on the GPU (bf16 29.5 GB): does not fit; it needs 24.0 GB free
  Qwen/Qwen3-32B    45.9 GB on the GPU (bf16 65.5 GB): does not fit; it needs 48.9 GB free

Ready: glyd run Qwen/Qwen3-8B
```

`glyd run Qwen/Qwen3-8B`, a question typed at the prompt, then `/bye` (the model's thinking cut short here):

```
Qwen3-8B: 12.2 GB on the GPU with Glyd, instead of 16.4 GB, which does not fit the 15.7 GB your GPU has free.
Downloaded Qwen3-8B (16.4 GB).
Settings: 11,264-token context (the most that fits), eager mode, 62% of GPU memory (14.7 GB), tool calls (hermes), thinking shown apart (qwen3); PyTorch sampler (no CUDA toolkit); ziglang as the C compiler (no gcc).
Ready in 1m44s. (The server's log: /work/home/.local/state/glyd/logs/run-20261001-034013.log)
Chat here, or open http://localhost:8000 in a browser.
Chatting with Qwen/Qwen3-8B. /bye to leave, /clear for a new chat, /? for more.
>>> Say hello in five words.
Thinking...
Okay, the user wants me to say hello in five words. Let me think. First, I need to make sure the response is exactly f ...

...done thinking.

Hello how are you?

>>> /bye
Stopping the server...
```

(`transcript/install.txt` and `transcript/run.txt` are the whole text. The container's home is /work/home, where a person's is /home/NAME.)

## Files

Files the box named `.log` are `.txt` here (the repository ignores `*.log`); the text inside still says `.log`.

- `runs/NAME/`: `acceptance.out` and `summary.txt`; `glyd-logs/`, the servers' logs; `logs/`, what each step printed (`freeze.txt`, `doctor*.txt`,
  `chat.log` (the API checks), `run.err` (`glyd run`'s own output), `webui-*.check.log`, `pip.log`, `page.head`).
- `calibration/`: the servers' logs of the two sessions on the box before the runs (`a24-*`: the first, with expandable segments and
  the compiled server; `a25-*`: the constants as shipped), their session logs, and `calib-*.txt`, the check against the constants.
- `scripts/`: what ran on the box: `acc.sh` (acceptance.sh under the box's lock), `transcript.sh`, `ptydrive.py`, `render.py`,
  the sessions of the first two days' `session.sh`, `inside.sh`, `plan_a24.sh`, `plan_a25.sh` and their checks, `calib.py`, and
  `parser_probe.py` (vLLM's qwen3 reasoning parser, token groups against its output, which showed the template's newlines to be the
  answer's and not a bug of the parser).
