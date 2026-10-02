# Local chat on a 16 GB GeForce card, tested on an L4 (2026-09-30)

`vllm serve Qwen/Qwen3-8B --quantization glyd` for one user, and Open WebUI in front of it: the test behind the "Local
chat, like Ollama" section of `gpu/vllm/README.md`. The card it is for is an RTX 4080 SUPER (16,376 MiB, GeForce Ada).
This was run on the dev L4 instead, made to look like that card in two ways (below). Nothing here ran on an RTX 4080
SUPER, so no speed here is that card's.

## The two stand-ins

- **The hog (`hog.py 15876`)** holds all of the L4's memory but 15,876 MiB: an RTX 4080 SUPER's 16,376 MiB less 500 MiB
  for a desktop. A server started after it sees what such a card leaves free. `hog.txt`: "6495 MiB held, 15875 of 22565
  MiB free". vLLM logs "Free memory on device (15.32/22.04 GiB) on startup"; its own CUDA context is the rest.
  - The hog only holds memory. It draws no screen and asks for no memory later, so it is not a desktop.
  - `--gpu-memory-utilization` is a share of the card's total memory, not of what is free: the budget is the fraction
    times the total, and vLLM logs it ("Desired GPU memory utilization is (0.638, 14.06 GiB)"). The L4's total is 22.04
    GiB, an RTX 4080 SUPER's 16,376 MiB (15.99 GiB). The same budget as 0.88 gives there (14.07 GiB) is **0.638** on
    the L4, so the L4 runs below use 0.638 where the README says 0.88. The first runs used 0.641 (14.13 GiB).
- **The GeForce emulation (`geforce/sitecustomize.py`)**, on `PYTHONPATH` (`serve.sh` sets it), makes Glyd's vLLM plugin
  read this GPU as GeForce Ada (code 1089, `glyd_gpu.h`'s GLYD_GPU_GEFORCE + 89) instead of as an L4 (3089, which the
  library reads from the device's name). Every server log prints "glyd test: the plugin reads this GPU as GeForce Ada
  (1089)".
  - The plugin sizes its scratch buffer by that code, at load. It is in the weights' memory in the logs below: 11.83
    GiB with vLLM's own chunk of prompt tokens, 11.64 GiB with `--max-num-batched-tokens 512`.
  - The library's own kernel choices stay the L4's, and `layout auto` takes the smallest layout (`mma`) on Ada (`glyd:
    mma layout` in the log of each server that loaded Glyd's weights). So the memory numbers are what the plugin asks of
    a GeForce Ada card; the speeds are the L4's.

## Setup

- **Machine:** the dev box (AWS g6.4xlarge, NVIDIA L4, 72 W), driver 595.91.07 (the RTX 4080 SUPER logs' driver too),
  Ubuntu 24.04.5, vLLM 0.30.0, torch 2.13.0, Python 3.12 (`env.txt`). The box runs one job at a time under a lock;
  `hold.sh` held it while `~/quick/HOLD` existed, so the steps ran back to back.
- **Glyd:** a wheel built from release-0.26.0 at c956e54 (`build_wheel.sh`: the GPU library by `gpu/build_lib.sh`, the
  codec by cargo, then `python -m build`; `build.txt`, `build_wheel.txt`, `cargo.txt`), installed beside vLLM in a
  venv. Its version string is still 0.25.1. The plugin is the vllm-plugin branch's, as release-0.26.0 merges it.
- **Model:** Qwen/Qwen3-8B, revision b968826d9c46dd6066d109eabc6255188de91218, 15.26 GiB of bf16 safetensors, already in
  the Hugging Face cache (`HF_HUB_OFFLINE=1`).
- **A server:** `bash serve.sh NAME vllm-serve-args...` starts `vllm serve` in the background, logs to
  `serve-NAME.log`, and waits until it is up or has exited. `bash stop.sh` stops it. `serve.sh` keeps its caller's
  output open while the server runs, so run it with its output in a file. The repo's `.gitignore` skips `*.log`, so the
  logs here are named `serve-NAME.txt`, `build.txt` and `open-webui.txt`.
- **One user's chat (`chat_test.py`):** a multi-turn streaming chat through the OpenAI API, thinking off: three turns
  (each with its first token's time, and its tokens a second after it) and one 512-token answer (`min_tokens`); then one
  answer of 1,024 tokens with thinking on. A rate is the usage's completion tokens less one, over the time from the
  first streamed text to the last. It writes `chat-*.json`, and the console's lines are in `chat-*.txt`.
- **Open WebUI:** `ghcr.io/open-webui/open-webui:main`, version 0.11.4, digest
  sha256:8b432fe0a65b91116afc7961365c6cca5379cc923171386a96691f3471f3cae9 (`docker-pull.txt`, `env.txt`), run as the
  README says (`--network=host`, `WEBUI_AUTH=False`); its log is `open-webui.txt`. Docker ran without sudo on the box.
  `webui_test.py` calls its API: health, sign-in, the model list, and a chat whole and streamed.
- **`memwatch.sh`** samples `nvidia-smi`'s memory.used every 0.1 s while a command runs (`memwatch-*.csv`, `.txt`).

## Runs

Every run is Qwen3-8B with the emulation and, but for the last, the hog; `HF_HUB_OFFLINE=1`. "Flags" are those after
`vllm serve Qwen/Qwen3-8B`. "KV" is vLLM's "Available KV cache memory" and "GPU KV cache size".

| Log | Flags | Weights | KV | Result |
| :--- | :--- | ---: | ---: | :--- |
| `serve-glyd-default` | `--quantization glyd --gpu-memory-utilization 0.641` | 11.83 GiB | -0.83 GiB | no server: "No available memory for the cache blocks" |
| `serve-glyd-seqs8` | as above, `--max-model-len 8192 --max-num-seqs 8` | 11.83 GiB | -0.27 GiB | no server |
| `serve-glyd-b512` | as above, `--max-model-len 4096 --max-num-seqs 8 --max-num-batched-tokens 512` (first start, compile cache empty) | 11.64 GiB | -0.05 GiB | no server |
| `serve-glyd-b512-warm` | the same, its second start, on the compile cache the first filled | 11.64 GiB | 2.18 GiB, 15,856 tokens | up; `chat-glyd-b512.json` |
| `serve-glyd-eager` | `--quantization glyd --gpu-memory-utilization 0.641 --max-model-len 4096 --max-num-seqs 8 --max-num-batched-tokens 512 --enforce-eager` | 11.64 GiB | 2.26 GiB, 16,432 tokens | up; `chat-glyd-eager.json` |
| `serve-glyd-eager-8k` | `--quantization glyd --gpu-memory-utilization 0.641 --max-model-len 8192 --enforce-eager` | 11.83 GiB | 1.77 GiB, 12,912 tokens | up; `chat-glyd-eager8k.json` |
| `serve-glyd-eager-8k-b` | the same again: the server Open WebUI was first tested against | 11.83 GiB | 1.77 GiB, 12,912 tokens | up; `webui-test-0.641.txt`, `context-limit.txt` |
| `serve-bf16-8k` | `--gpu-memory-utilization 0.641 --max-model-len 8192 --enforce-eager` (bf16: no `--quantization`) | | | no server: CUDA out of memory loading the weights |
| `serve-glyd-too-high` | `--quantization glyd --enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.88` | | | no server: refused at start (below) |
| `serve-glyd-eager-8k-0.638` | `--quantization glyd --enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.638 --host 127.0.0.1` | 11.83 GiB | 1.71 GiB, 12,432 tokens | up; `chat-glyd-eager-8k-0.638.json`, `webui-test-0.638.txt`, `memwatch-glyd-eager-8k-0.638.*`, `footprint-0.638.txt` |
| `serve-glyd-eager-8k-0.638-expandable` | the same with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 11.31 GiB | 2.13 GiB, 15,504 tokens | up; `chat-glyd-eager-8k-0.638-expandable.json`, `memwatch-...-expandable.*`, `footprint-0.638-expandable.txt` |
| `serve-bf16-eager-8k-0.638` | `--enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.638 --host 127.0.0.1` (bf16) | | | no server: CUDA out of memory loading the weights |
| `serve-glyd-eager-8k-0.638-curl` | the 0.638 run, without the hog (21.85 GiB free at start) | 11.83 GiB | 1.71 GiB, 12,432 tokens | up; `curl-test.txt` |

## Results

- **vLLM's defaults did not start** at the test's budget. Compiled, with CUDA graphs and the model's own 40,960-token
  context, the KV cache had -0.83 GiB. With `--max-model-len 8192 --max-num-seqs 8` it had -0.27 GiB.
- **Compiled started with a 4,096-token context, 8 sequences and 512 tokens a step, on its second start.** Its first
  start compiled the model inside the memory profile (85 s, a torch peak increase of 1.17 GiB) and found -0.05 GiB. The
  second start loaded the compiled graph from the cache (5 s, a peak increase of 0.07 GiB) and found 2.18 GiB for the KV
  cache.
- **Eager started on its first try** in each configuration, with only `--max-model-len` set for 8,192 tokens (1.77 GiB
  at 0.641), and it ran as fast for one user as compiled:

  | One user, tokens/s (first token, ms) | Turn 1 | Turn 2 | Turn 3 | 512 tokens | Thinking, 1,024 tokens |
  | :--- | ---: | ---: | ---: | ---: | ---: |
  | Compiled, context 4,096 | 22.4 (111) | 22.0 (118) | 22.5 (125) | 21.5 (97) | 21.4 (56) |
  | Eager, context 4,096 | 22.3 (115) | 21.6 (120) | 22.2 (125) | 21.1 (99) | 21.0 (58) |
  | Eager, context 8,192, 0.641 | 22.3 (112) | 21.6 (119) | 22.2 (124) | 21.2 (99) | 21.0 (56) |
  | Eager, context 8,192, 0.638 | 22.2 (94) | 21.6 (120) | 22.2 (126) | 21.1 (104) | 21.0 (57) |
  | Eager, context 8,192, 0.638, expandable segments | 22.3 (115) | 21.7 (119) | 22.3 (123) | 21.2 (99) | 21.0 (56) |

  Turn 1 has 35 prompt tokens, turn 3 about 170. Compiled is 0.2 to 0.4 tokens/s ahead, 1-2%.
- **Start time.** The eager 0.638 server was up 39 s after its first log line, 21.8 s of it loading the weights. The
  first start of the session took 124 s to load them (`serve-glyd-default.txt`); the later ones took 21 to 22 s, and 32 s
  with expandable segments.
- **bf16 did not start** at either budget: "CUDA out of memory. Tried to allocate 1.16 GiB. GPU 0 has a total
  capacity of 22.04 GiB of which 1.16 GiB is free", with 14.11 GiB allocated by PyTorch, while loading. The checkpoint is
  15.26 GiB and the card had about 15.3 GiB free. Glyd's weights took 11.83 GiB.
- **What the server holds.** `nvidia-smi` showed VLLM::EngineCore at 14,958 MiB at 0.638 (`footprint-0.638.txt`), 0.55
  GiB more than the 14.06 GiB budget. Of a 16,376 MiB card that leaves 1,418 MiB for a desktop and a browser. Only the
  hog and the server were on the GPU: Open WebUI, running at the time, used none.
- **A budget too high is refused at start.** The L4's 0.88 is 19.39 GiB: "Free memory on device cuda:0 (15.32/22.04
  GiB) on startup is less than desired GPU memory utilization (0.88, 19.39 GiB). Decrease GPU memory utilization or
  reduce GPU memory used by other processes." (`serve-glyd-too-high.txt`). The second number is the fraction times the
  card's total: 0.88 of 16,376 MiB is 14.07 GiB.
- **While the weights load, the server takes nearly all the free memory.** `memwatch-glyd-eager-8k-0.638.txt`: the
  most used was 22,564 MiB of the 22,565 MiB CUDA sees, 1 MiB free, and 81 samples (8.1 s) had under 512 MiB free. The
  allocator logged 209 "memory allocation failed with OOM" warnings and recovered each time, and the load went on. The
  hog is not a desktop, so what a desktop's own allocations do in those seconds was not measured.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** at the same budget: no allocator warnings, 11.31 GiB of
  weights against 11.83, so 2.13 GiB of KV cache (15,504 tokens) against 1.71 GiB (12,432), and the same tokens a
  second. The lowest free memory was 9 MiB, with 20 samples (2 s) under 512 MiB, and the weights loaded in 32.2 s
  against 21.8 s. One run each. The server held the same memory (14,918 MiB against 14,958): the budget is the same,
  and less of it was lost in the load.
- **A chat past the context is refused,** by the server and through Open WebUI: HTTP 400 "This model's maximum context
  length is 8192 tokens" (`context-limit.txt`, 9,000 words asked for).
- **Open WebUI 0.11.4** was answering `/health` 28 s after `docker run`. With `WEBUI_AUTH=False` its API still wants a
  token: `/api/models` without one gave 401 (`open-webui.txt`). A sign-in call with empty fields returns the session of
  its one built-in user, with a token, and creates that user on the first call; `webui_test.py` uses it, so the test
  sends no credentials. Through the API:
  - the model list was `['Qwen/Qwen3-8B', 'arena-model']` (`arena-model` is Open WebUI's own);
  - a chat of 35 tokens took 1.67 s, whole;
  - a streamed answer of 300 tokens (a chunk a token) began at 99 ms and ran at 21.2 tokens/s, as from the server
    directly (`webui-test-0.638.txt`, `webui-test-0.641.txt`: 97 ms, 21.2);
  - the log shows one error per model list: Open WebUI tries its default Ollama address, which does not exist here.
- **The README's curl commands** ran as written against the server (`curl-test.txt`: the run's console output; the
  file on the box was emptied by a stray command). `/no_think` at the end of a message gave an empty `<think>` block,
  then the answer.

## Not measured

- Anything on an RTX 4080 SUPER: its speed, and what a desktop does while the server loads.
- Open WebUI's page in a browser: only its API was called. How the page shows Qwen3's thinking is not seen either.
- A first start that downloads the model: it was in the cache.

## Run it again

On a machine with the venv, the model in its cache, and this directory as `~/quick`:

```bash
python hog.py                                 # holds all but 15,876 MiB; leave it running
bash serve.sh NAME --quantization glyd ... > /dev/null   # serve.sh: vllm serve "$@", with PYTHONPATH=geforce
python chat_test.py http://localhost:8000/v1 Qwen/Qwen3-8B chat-NAME.json
bash stop.sh
```
