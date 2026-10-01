# Second round: after the review of v0.26.0rc3 (2026-10-01)

The review of the first round's tree (1 blocker, 15 should-fix, 19 nits) was answered on the `onboarding` branch (merged with `release-0.26.0`
at the rc3 bump) and the acceptance run again, with cases for what the review found. What changed, as the review numbered it, is in the
CHANGELOG's v0.26.0 entry and in the commits; what was run is here. The machine is the first round's: the L4, driver 595.91.07 (CUDA 13.2),
Ubuntu 26.04 containers with no CUDA toolkit and no gcc, vLLM 0.30.0 and PyTorch 2.13.0. The wheel of the three runs is the branch's at 109923e, built as
release.yml builds it. The Rust `glyd` for the cases that need it was built from the same tree on the box (`scripts/rustbuild.sh`, `cargo build
--release --bin glyd`). What the tests and the runs say about the code before the fixes is in `review-1/tests/`.

| Run (`review-1/runs/NAME/`) | Command | Result |
| :--- | :--- | :--- |
| `24gb` | `acceptance.sh --wheel W --rust-glyd R --webui none` (a fresh work directory: a cold uv cache) | PASSED, 49 checks |
| `16gb-card` | `acceptance.sh --wheel W --rust-glyd R --card 4080s --webui all --pip-refusal` | PASSED, 58 checks, with Open WebUI as the README had it then (`WEBUI_AUTH=False`: no login) |
| `16gb-card-login` | the same, after the README's Open WebUI commands lost `WEBUI_AUTH=False` (264af19; the run is of be01a5d, which also makes the pip refusal's environment new) | PASSED, 64 checks: in each of the three routes Open WebUI asks for a login and its first account signs up as the administrator |
| `16gb-card-rereview` | the same, on the code after the re-review's fixes (df2be01: `glyd run` looks at standard input after the checks of the machine and the model, SIGTERM and SIGHUP during a download end at once, one API key and `--hf-token` moved to the environment, a loopback address that was chosen says its checks are off, a MIG GPU not picked ahead of a usable one); the wheel and the Rust `glyd` are built from df2be01 | PASSED, 64 checks |
| `8gb-card` | `acceptance.sh --wheel W --rust-glyd R --card 8gb` | PASSED, 17 checks: Qwen3-4B refused with a model to try, Qwen3-1.7B (24,576 tokens, 29%) run |
| `head-smoke-1.7b` | `acceptance.sh --wheel W --rust-glyd R --model Qwen/Qwen3-1.7B --webui none` on the branch's head (cf616b7), after the three runs: the one commit of Python code since their wheel (1260fa1: a server's message that ends in a full stop is not given a second), this install.sh, and the Ctrl-C cases' stricter check that glyd says it is stopping | PASSED, 49 checks |
| `cli-x86_64-no-gpu` | `acceptance.sh --flow cli --rust-glyd R` (a container with no GPU on the box) | PASSED, 11 checks |
| `cli-aarch64-docker-on-a-mac` | `acceptance.sh --flow cli` (Docker on an Apple-silicon Mac: the linux-aarch64 tarball) | PASSED, 9 checks |
| `rc3-wheel-red` | the same acceptance.sh against the published rc3 wheel (the code before), `--up-wait 240` | FAILED, 9 of 32 checks, as the review said it would (below) |

What the new cases checked, in the order the glyd flow runs them (`runs/24gb/acceptance.out` has each line):

- **install.sh, a program of the user's own at `~/.local/bin/glyd`** (a script that prints "mine"): it stops before installing Glyd, says what to
  do (the tool in another folder, or remove it) and leaves the program as it was. (S4)
- **install.sh, the install:** uv 0.12.21 by its own installer (checked against its sha256, told to edit no startup file), 12 s to install 199
  packages from a cold cache, and *every one of the 199 packages at the version install.sh's list gives*: the list in the script is this run's own
  `freeze.txt` (`scripts/install_constraints.py --check runs/24gb/logs/freeze.txt` says so), taken from the first round's last run before and held
  by uv's `--constraints` now. No pre-release among them. (S3)
- **install.sh again** (the update the README gives): with `~/.local/bin` off PATH it says it will edit the shell's startup file, uv names the
  file it made, and `.bashrc` has the line; with another `glyd` first on PATH it names it and gives the line to add and the path to run the new
  one by; `glyd doctor` says the same. The Rust `glyd` built from this tree, first on PATH, hands `glyd doctor` to the Python tool. (S4, S13)
- **`glyd serve` under `nohup`**, started through the Rust glyd: SIGHUP, which `nohup` ignores, does not stop it (the ignore survives the
  Rust glyd's exec and glyd's own handlers). (S7d, S13)
- **The server on 127.0.0.1** (`logs/guard.txt`, 11 checks): a program with no Origin, the page's own origin (localhost and 127.0.0.1) are answered; another
  site's Origin is 403 (a GET, its preflight, its POST); a rebinding Host is 421 (the API and the page); CORS names only the page's own origin and
  grants nothing to another. (S1)
- **A prompt full of escape sequences** (`ESC[31m`, an OSC title with BEL): the model's own answer to it carried 4 control characters, `glyd run`'s
  output none. (S5)
- **A server with an API key**, as `-- --api-key KEY` and as the README's `VLLM_API_KEY=KEY glyd serve ... --host 0.0.0.0`: ready and serving
  (glyd's readiness check no longer needs the key), its banner says it asks for the key, the note names plain HTTP and what the key does not guard,
  the key is on no command line (`ps`) and in no log or output, `/v1` is 401 without it and 200 with it, `/health` stays open. Open WebUI's bridge
  route runs against the second. (B1, S2)
- **Open WebUI** (each of the three routes of `16gb-card-login`, `check.py`'s `webui`): its login is on, as the README's commands leave it: `/api/models`
  with no token is 401 and the sign-in that no-login mode took (empty credentials) is 400; the first account, signed up through the API as a person
  does on first visit, is the administrator; the model is listed and chats (a tool call in the stream too) with that account's token; and its CORS is
  limited to its own addresses (another site's request gets no `access-control-allow-origin`, its preflight is 400, the page's own origin is
  granted). (S14; `openwebui-login/` is the probe of that first visit, `openwebui-cors/` the one of no-login mode that decided the default.)
- **Ctrl-C while the model loads**, once and twice: glyd exits 130, prints that it is stopping, no vLLM process is left. The first run of this case
  took 50 s from the Ctrl-C to the exit (the engine does not answer SIGTERM until the load is over, and glyd waited 30 s for it): a server that is
  still loading is now swept after 5 s. (S7)
- **The engine killed under a terminal chat's answer:** "The server could not finish the answer: EngineCore encountered an issue... Ask again.", and
  `glyd serve` exits saying the server stopped. (S6)
- **`uv tool uninstall glyd`:** the tool and its link are gone, and `~/.local/state/glyd` (logs, `serving`, `zigcc`), the Hugging Face cache and uv stay,
  as the README says. (nit 18)

The three runs share one work directory, in this order (`scripts/chain.sh`): the 24 GB run started it (its install, 12 s, and its `glyd run`, 1 min 19 s,
are a cold uv cache and a cold Triton cache; vLLM logged the same weights and KV cache tokens as in the first round, 11.38 GiB and 61,104), and
the 16 GB and 8 GB runs after it kept uv's cache and Triton's launchers (their `glyd run` took 58 s and 48 s, which are not first starts: the
README's table has the first round's, from a fresh home each time). Everything else in a run's home is cleaned before its install.

The server logs the checks scan for tracebacks and allocator warnings are those before the Ctrl-C and engine-kill cases, which make their own:
4 in `24gb`, 4 in `16gb-card`, 4 in `16gb-card-login`. None has a traceback or an allocator warning.

`openwebui-cors/` (S14, run on the box with Open WebUI 0.11.4's image, CPU only): with `WEBUI_AUTH=False` and the default `CORS_ALLOW_ORIGIN` (`*`)
a sign-in with `Origin: http://evil.example` is answered with the administrator's token and `access-control-allow-origin: http://evil.example`,
and the administrator's `/api/v1/functions/`, `/tools/` and `/users/` answer 200; with `CORS_ALLOW_ORIGIN=http://localhost:3000;http://127.0.0.1:3000`
the same request gets no grant, its preflight is 400, and the page's own origin is granted. A request with `Host: evil.example:3000` is answered
either way: Open WebUI does not check the Host (DNS rebinding is not closed by this setting). That is why the README's commands no longer set
`WEBUI_AUTH=False`; its "if you want no login" note names the risk.

`openwebui-login/` (the same image and box, the README's Docker command without `WEBUI_AUTH=False`): `/api/models` with no token is 401; a sign-in with
empty credentials is 400; `/api/config` says `onboarding: true`; the first sign-up (`/api/v1/auths/signup`) returns `role: admin` and a token; a second
sign-up is refused (403: "contact your administrator"); with the token `/api/models` is 200; a sign-in from `Origin: http://evil.example` and its
preflight get no `access-control-allow-origin` (400), the page's own origin does. Until the first account exists, whoever reaches the port first makes it:
the README says to make it at once.

`rc3-wheel-red` is what the acceptance run says about the code before the fixes (the rc3 wheel from the release, this tree's `install.sh` and
`acceptance.sh`): the server on 127.0.0.1 answers another site's Origin with 200 (its preflight 200, its POST 400, vLLM's CORS `*`) and a rebinding
Host with 200, for the API and the page (S1); `glyd serve` under `nohup` stops on SIGHUP (S7d); `glyd serve --host 0.0.0.0` with a key, as a flag and
in `VLLM_API_KEY`, never prints that it is serving, so a key's server was not usable from `glyd` (B1; the old log's first line shows the key on the
command line, S2: masked here as `acceptance-KEY`); `glyd run` prints the escape sequences a model repeats (S5); with the engine killed under a
chat's answer the terminal shows nothing (S6); `glyd doctor` has no row for a glyd ahead of it (S13). Two of the nine, the API checks and the
long conversation in the terminal chat, fail because the SIGHUP stopped the server they use. The Ctrl-C cases and `uv tool uninstall` pass on the old code: a
Linux server is also stopped when glyd dies (`PR_SET_PDEATHSIG`), so a second Ctrl-C leaves no engine behind (the stand-in test of
`test_a_second_ctrl_c_does_not_skip_the_sweep` is what shows the skipped sweep); what the new code adds there is the message, and 5 s instead of 30 s.

`review-1/tests/python-rereview-nits-on-the-code-before.txt` is the re-review's new tests against the code before its fixes (f5c2791): 5 of the 7 fail (N1 says "no prompt: standard input is empty" on a machine with no GPU, SIGTERM during a download takes 12 s for a 12 s shard, several keys are not refused, a chosen loopback address says nothing, a MIG card is picked ahead of an RTX 4090); the other two are the updated empty-prompt test, which holds on both orders, and the SIGINT-ignored test, whose fix is in the test file itself. 64 of 64 pass on the branch.

`review-1/tests/`: the review's tests, run on the first round's code (4ed64ef) and on the branch. `python-on-the-code-before.txt`: 27 of the 59 tests of
`bindings/python/test_onboard.py` pass there (the old ones) and 32 do not (the new ones and the old ones changed for the new behavior: some fail
for what the code does, some for a name it does not have yet); `python-on-the-code-after.txt`: 59 of 59. `rust-on-the-code-before.txt`:
`tests/cli_python_tool.rs`, 4 of 5 fail on the old `glyd.rs` (the fifth is a file named run, still compressed as `./run`);
`rust-on-the-code-after.txt`: 5 of 5. `install-sh-on-the-code-before.txt`: the nine install tests, all failing on the old script.

# `glyd run`, `glyd serve` and the installer, from nothing, on an L4: the first round

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
