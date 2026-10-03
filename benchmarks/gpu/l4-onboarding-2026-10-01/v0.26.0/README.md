# v0.26.0, after the release: the acceptance on the published artifacts (2026-10-01)

The same gate as the release candidates', run against what was published: `glyd[vllm]==0.26.0` from PyPI (the wheel pip and uv download, not a
build of this tree), `install.sh` as the v0.26.0 tag has it (fetched from its raw GitHub URL, and then, once the site was live, from
getglyd.com), and the Rust `glyd` of the release's linux-x86_64 tarball. The machine is the first round's: the AWS L4 (24 GB, Ada, sm_89), driver
595.91.07 (CUDA 13.2), clean Ubuntu 26.04 containers with no CUDA toolkit and no gcc, as a user that is not root. What the artifacts are, with
hashes: [published-artifacts.txt](published-artifacts.txt).

[gpu/vllm/acceptance.sh](https://github.com/surya-koritala/Glyd/blob/v0.26.0/gpu/vllm/acceptance.sh) of this branch is the script (v0.26.0's, with two options added for this: `--install-url`,
a release's raw `install.sh` in place of the checkout's, and `--served`, the README's install line run as it is, from the network, with a copy of
what the site served kept for the cases that run the script again and compared with the tag's). Every run takes the model's files from the
box's Hugging Face cache; the packages come from PyPI.

| Run (`runs/NAME/`) | Command | Result |
| :--- | :--- | :--- |
| `16gb-card` | `acceptance.sh --rust-glyd R --install-url TAG_RAW --card 4080s --webui all --pip-refusal` | **PASSED, 64 checks, 0 failed**. A fresh work directory: a cold uv cache, 200 packages fetched in 14 s. |
| `24gb` | `acceptance.sh --rust-glyd R --install-url TAG_RAW --webui none` | **PASSED, 49 checks, 0 failed** (the whole L4: a 40,960-token context; Ctrl-C while loading, a killed engine and `uv tool uninstall` included). |
| `16gb-card-served-script` | `acceptance.sh --rust-glyd R --served --install-url TAG_RAW --card 4080s --webui all --pip-refusal` | **PASSED, 65 checks, 0 failed**: the install was `curl -LsSf https://getglyd.com/install.sh \| sh` as the README gives it. |

`R` is the release's Rust `glyd` (`~/onb/post-rust/glyd`, [scripts/post-setup.sh](scripts/post-setup.sh)); `TAG_RAW` is
`https://raw.githubusercontent.com/surya-koritala/Glyd/v0.26.0/scripts/install.sh`. The runs went under the box's GPU lock, one after the other, in one work
directory ([scripts/acc-post-and-chains.sh](scripts/acc-post-and-chains.sh)); only the first had a cold uv cache.

## What was proved on the published artifacts

- **The install from PyPI.** `install.sh` (sha256 `f6277177...01e2`, 19,794 bytes) installed uv 0.12.21 by its own installer, then `glyd[vllm]==0.26.0` and Python
  3.12 (uv's output in the first run: `Downloading glyd (15.8MiB)`, the x86_64 wheel's size, and `glyd==0.26.0` in the resolved list, not a file URL), ziglang
  0.16.0 where there was no compiler, and **every one of the 199 packages at the version the script's own list gives**. No pre-release among the dependencies
  but opentelemetry's beta-only packages. No shell startup file touched where `~/.local/bin` was on PATH; the update (`install.sh` again, `glyd[vllm]==0.26.0` is already installed) said
  its profile edit first where it was not.
- **The script the site serves is the tag's.** getglyd.com/install.sh answered 200 as `text/plain` and is `install.sh` of v0.26.0 byte for byte (the check is in
  `16gb-card-served-script/summary.txt`); the third run's install is the README's line itself, from the network.
- **At a 16 GB card's memory (14.8 GB free, a desktop's), Qwen3-8B**: `glyd run` chose a 11,264-token context at 62% (14.7 GB) and vLLM logged 12,960 KV tokens, as in the
  release candidates; first `glyd run` 1 min 17 s, then `glyd serve` 41 s. **At 24 GB**: 40,960 tokens at 92% (21.8 GB), 61,104 KV tokens; 1 min 4 s, then 41 s. (The third run's first
  start took 3 min 4 s: the box had been restarted and the model's 16 GB came off a cold disk.)
- **Open WebUI 0.11.4 in all three ways** (uvx, Docker with host networking, Docker's own network with a key; the first and third run): its login is on
  (`/api/models` with no token 401, the no-login sign-in 400), the first account signed up is the administrator, the model is listed, it chats and calls a tool with that account's
  token, and its CORS is limited to its own addresses.
- **What a user meets:** a program of the user's own in `~/.local/bin` is not replaced; a `glyd` ahead on PATH is said, with the line to add, by the script and by `glyd doctor`; the release's
  Rust `glyd` first on PATH hands `run`, `serve`, `doctor` and `login` to the Python tool, and a `glyd serve` started through it under `nohup` survives SIGHUP; the server on 127.0.0.1 refuses another site's
  Origin (403) and a rebinding Host (421) and answers its own page; a server with an API key, as `-- --api-key` and as `VLLM_API_KEY`, is ready, keeps the key off every command line and log, and asks for it on `/v1` only;
  a prompt full of escape sequences comes out with none; a conversation past the window is said so; and, at 24 GB, Ctrl-C while the model loads (once and twice: exit 130 in 2 s and 0 s, no vLLM process
  left), the engine killed under a terminal chat's answer ("The server could not finish the answer ... Ask again."), and `uv tool uninstall glyd` leaving the logs, the models and uv, as the README says.
- **`pip install` with no compiler** (the pip refusal, first and third run): `glyd run` stops with "vLLM needs a C compiler to start ... Install one: sudo apt install build-essential python3-dev".

## Warnings and tracebacks

`python3 post_warnings.py` over each run's server logs (`runs/*/glyd-logs`); the acceptance itself fails a run on a traceback before a shutdown line or an allocator
out-of-memory warning in the first four logs (`glyd run` once, `glyd serve` once and with a key twice):

| Run | Server logs checked | Tracebacks | Allocator OOM warnings | `warning:` in the install output |
| :--- | ---: | ---: | ---: | ---: |
| `16gb-card` | 4 | 0 | 0 | 0 |
| `24gb` | 4, and 3 of the cases that stop a server on purpose | 0 in the 4; 4 in the 3 (see below) | 0 | 0 |
| `16gb-card-served-script` | 4 | 0 | 0 | 0 |

The 4 tracebacks of the three logs that stop a server on purpose: one in each of the two Ctrl-C-while-loading logs (vLLM's API server, interrupted inside its own engine start, which is what the case
does), and the two of the killed engine's log (`EngineDeadError: EngineCore encountered an issue`, after the log's shutdown line), which is what `glyd` then shows the chat as "The server could not finish the answer".

The WARNING lines that are in the 4 checked logs of every run are vLLM's own notices and nothing else: "Enforce eager set, disabling torch.compile and CUDAGraphs" and "Inductor compilation was
disabled by user settings" (glyd starts the server eager, on purpose), "Default vLLM sampling parameters have been overridden by the model's `generation_config.json`", and, at the server's
stop, "[shutdown] Process manager: force killing remaining process EngineCore" (vLLM's own teardown). No other WARNING line is in the server logs, and uv printed no `warning:` line.

## One thing that happened

The first attempt of the third run was cut off by the box: the dev machine stops itself on a timer (set by a `devbox start`; this one fired in the middle of the run), after 15
checks, none failed (`runs/16gb-card-served-script/first-attempt-cut-off-by-the-box-shutdown.out`). The box was started again and the run made again from nothing; that is the run in the table.

## Files

`runs/NAME/`: `acceptance.out` and `summary.txt`; `glyd-logs/`, the servers' logs; `logs/`, what each step printed (`install.txt`, `freeze.txt`, `doctor*.txt`, `guard.txt`, `chat.txt`, `webui-*.check.txt`, `pip.txt`, ...). Files
the box named `.log` are `.txt` here (the repository ignores `*.log`); the text inside still says `.log`. A random API key of a run's (`acceptance-NNNN`) is masked as `acceptance-KEY`. `published-artifacts.txt`: the PyPI files, the scripts'
hashes, the release assets. `scripts/`: what ran on the box.
