# v0.28.0rc1, after the publish: the acceptance on the published artifacts (2026-10-04)

The same gate as v0.27.0's, run against what was published: `glyd[vllm]==0.28.0rc1` from PyPI (the wheels uv downloads: glyd 2.2 MiB and glyd-gpu 28.9 MiB; both are pre-releases, `pip install glyd` and `pip install glyd-gpu`
still give 0.27.0), `install.sh` as the release branch has it (`release-0.28.0`, raw GitHub; the file of tag v0.28.0rc1 too, byte for byte), and the Rust `glyd` of the pre-release's linux-x86_64 tarball (sha256 checked). The machine is
the AWS L4 (24 GB, Ada, sm_89), driver 595.91.07 (CUDA 13.2), clean Ubuntu 26.04 containers with no CUDA toolkit and no gcc, as a user that is not root. What the artifacts are, with hashes: [published-artifacts.txt](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/acceptance-v0.28.0rc1-l4-2026-10-04/published-artifacts.txt).
The script is `gpu/vllm/acceptance.sh` of the private branch `acceptance-028` (e8506743: private main 11c0661f and [scripts/acceptance-028.patch](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/acceptance-v0.28.0rc1-l4-2026-10-04/scripts/acceptance-028.patch)). Every run took the model's files from the box's
Hugging Face cache; the packages came from PyPI.

| Run (`runs/NAME/`) | Command | Result |
| :--- | :--- | :--- |
| `16gb-card` | `acceptance.sh --rust-glyd R --install-url REL --card 4080s --webui all --pip-refusal` | **PASSED, 64 checks, 0 failed** (v0.27.0: 64). A fresh work directory: a cold uv cache, the install 12 s. |
| `24gb` | `acceptance.sh --rust-glyd R --install-url REL --webui none` | **PASSED, 49 checks, 0 failed** (v0.27.0: 49). The whole L4: a 40,960-token context, the lossless KV cache on; Ctrl-C while loading, a killed engine and `uv tool uninstall` included. |
| `16gb-card-served-script` | `acceptance.sh --rust-glyd R --served --prerelease --install-url REL --card 4080s --webui all --pip-refusal` | **64 checks passed, 1 failed, the one expected** (v0.27.0: 65 passed): the install is the pre-release's line, `curl -LsSf REL \| sh`, from the network, and every check on what it installed passes; the one that fails is the check that `getglyd.com/install.sh` is the release's script (see "The site"). |
| `cli` | `acceptance.sh --flow cli --rust-glyd R` | **PASSED, 10 checks** (v0.27.0: 10): the pre-release's tarball checked against its sha256, `glyd 0.28.0-rc.1`, a file through glyd and back, the update, a program of the user's own left alone, `glyd run` saying what it needs. |

`R` is the pre-release's Rust `glyd` (`~/rel028/rust/glyd`, [scripts/prep.sh](scripts/prep.sh)); `REL` is `https://raw.githubusercontent.com/surya-koritala/Glyd/release-0.28.0/scripts/install.sh`. No `--version` in any run:
the install is what the script pins, `0.28.0rc1` (and `--pip-refusal` now installs that version too: see "What the script needed"). The runs went under the box's GPU lock, one at a time with a minute between them
([scripts/chain.sh](scripts/chain.sh), [acc.sh](scripts/acc.sh)); the cli flow uses no GPU and its checkout's `scripts/install.sh` is the release branch's. 2026-10-04, 23:14 to 23:45 UTC.

## What was proved on the published artifacts

- **The install from PyPI.** `install.sh` (sha256 `97d26dde...5e`, 19,718 bytes; v0.27.0's with the default version `0.28.0rc1` and nothing else changed) installed uv 0.12.21 by its own installer, then `glyd[vllm]==0.28.0rc1` and Python 3.12:
  uv's output has `Downloading glyd-gpu (28.9MiB)` and `Downloading glyd (2.2MiB)`, 201 packages, `glyd==0.28.0rc1` and `glyd-gpu==0.28.0rc1` in the resolved list, ziglang 0.16.0 where there is no compiler, and **every one of the 199 other
  packages at the version the script's own list gives**; no pre-release among the dependencies but opentelemetry's betas. No shell startup file touched where `~/.local/bin` was on PATH; the update (`install.sh` again) said its profile edit first
  where it was not.
- **A clean `pip install "glyd[vllm]==0.28.0rc1"` resolves to the packages of 0.27.0's** but glyd itself and glyd-gpu ([resolve/diff.txt](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/acceptance-v0.28.0rc1-l4-2026-10-04/resolve/diff.txt): pip 26.2.1, CPython 3.12.3, x86_64, 200 packages each; `glyd` 0.27.0 -> 0.28.0rc1, `glyd-gpu`
  0.27.0 -> 0.28.0rc1, nothing else differs). The pip install without a compiler (`--pip-refusal`, in both GPU-card runs) stops with "vLLM needs a C compiler to start ... Install one: sudo apt install build-essential python3-dev", on 0.28.0rc1's own `glyd run`.
- **The published wheels pass the wheel gate** ([wheel-gate.txt](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/acceptance-v0.28.0rc1-l4-2026-10-04/wheel-gate.txt), [gate.out](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/acceptance-v0.28.0rc1-l4-2026-10-04/gate.out)): `check_wheel.py` of the release commit, on the x86_64 and the aarch64 glyd-gpu wheel as PyPI serves them (sha256 as PyPI's), with cuobjdump (CUDA 13.4.92): nothing wrong.
- **Old saves** ([oldsave/](https://github.com/surya-koritala/Glyd/tree/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/acceptance-v0.28.0rc1-l4-2026-10-04/oldsave), below).
- **At a 16 GB card's memory (14.8 GB free), Qwen3-8B**: `glyd doctor` says it fits with a 17,408-token context; `glyd run` chose 17,408 tokens at 62% (14.7 GB), vLLM logged 19,440 KV tokens and 10.38 GiB for the model (v0.27.0: 11,264 tokens, 12,960 KV tokens,
  11.39 GiB); `glyd serve` up in 56 s (Ready in 53 s). The first `glyd run` of the first run was Ready in 3 min 11 s: loading the weights took 126.7 s because the box had just booted and the model's 16 GB came off a cold disk (every later start loaded them in 25.6 to 25.9 s,
  and the served run's `glyd run` was Ready in 1 min 11 s).
- **At 24 GB**: 40,960 tokens at 92% (21.8 GB); `glyd run` 1 min 13 s on vLLM's own cache (67,600 KV tokens; v0.27.0: 61,104); the first `glyd serve` 4 min 35 s (278 s up), because the lossless KV cache takes the model's tables on a first start (222 s on
  this L4, then kept: the later starts were Ready in 53 s), and then 87,072 KV tokens (+28.8%) in the same memory (v0.27.0: 78,480).
- **Open WebUI 0.11.4 in all three ways** (uvx, Docker with host networking, Docker's own network with a key), in both 16 GB runs: its login is on, the first account is the administrator, the model is listed, it chats and calls a tool, its CORS is limited
  to its own addresses. **What a user meets**, as before: a program of the user's own in `~/.local/bin` is not replaced; a `glyd` ahead on PATH is said, with the line to add; the release's Rust `glyd` first on PATH hands `run`, `serve` and `doctor` to the
  Python tool, and a `glyd serve` under `nohup` survives SIGHUP; the server refuses another site's Origin and a rebinding Host; a server with an API key (as `-- --api-key` and as `VLLM_API_KEY`) keeps the key off every command line and log; a prompt full of
  escape sequences comes out with none; a conversation past the window is said so; and, at 24 GB, Ctrl-C while the model loads (once and twice), the engine killed under a terminal chat's answer, and `uv tool uninstall glyd`.

## Old saves ([oldsave/](https://github.com/surya-koritala/Glyd/tree/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/acceptance-v0.28.0rc1-l4-2026-10-04/oldsave), [scripts/oldsave_run.sh](scripts/oldsave_run.sh))

Two virtual environments on the box, `glyd[gpu]==0.27.0` and `glyd[gpu]==0.28.0rc1` from PyPI, each with install.sh's list of package versions as constraints (torch 2.13.0, transformers 5.18.0 ...): they differ in glyd and glyd-gpu alone
([oldsave/freeze-v027.txt](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/acceptance-v0.28.0rc1-l4-2026-10-04/oldsave/freeze-v027.txt), [freeze-v028rc1.txt](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/acceptance-v0.28.0rc1-l4-2026-10-04/oldsave/freeze-v028rc1.txt)). Qwen3-0.6B, four short prompts, greedy, 32 new tokens at most.

1. **A smallest-layout save by glyd-gpu 0.27.0** (`from_pretrained(.., layout="mma")`, `save_pretrained(.., layout="mma")`): format `glyd-v1`, 112 packs; 0.27.0 loads it again with `verify=True` (198 packed tensors checked, 114 saved as they are) and answers as it
   did in memory ([logs/1-make-027.txt](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/acceptance-v0.28.0rc1-l4-2026-10-04/oldsave/logs/1-make-027.txt)).
2. **0.28.0rc1 loads that save with `verify=True`** (198 checked, 114 hashed, 0 unhashed) and its greedy answers are **token for token 0.27.0's, 4 of 4** (against the answers of 0.27.0's model in memory and of its save loaded):
   `'Paris'` to "What is the capital of France? Answer in one word.", `'17 + 25 = 42.'`, `'The quick brown fox jumps over the fence.'`, and a 23-token sentence about the sea ([logs/2-load-028rc1.txt](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/acceptance-v0.28.0rc1-l4-2026-10-04/oldsave/logs/2-load-028rc1.txt)).
3. **A glyd-v4 save made by 0.28.0rc1** (`save_pretrained`: `glyd-v4`, 112 packs; and `glyd pack`: its 198 tensors checked, and `glyd verify` of 0.28.0rc1 passes it): 0.28.0rc1 loads it with `verify=True` and answers the same 4 of 4. The v4 save of the Hub's model and the
   v4 save of the old save as loaded (packed again as it loads) are the same bytes (one `model.safetensors`, sha256 `c30b9b8c2e41...`).
4. **0.27.0 refuses the glyd-v4 save, in words**: `glyd.from_pretrained(path)` of 0.27.0 raises
   `ValueError: .../glyd.json: format 'glyd-v4'; this glyd reads glyd-v1 and glyd-v2 and glyd-v3` ([logs/3-refuse-027.txt](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/acceptance-v0.28.0rc1-l4-2026-10-04/oldsave/logs/3-refuse-027.txt)). `glyd verify PATH` of 0.27.0 (exit 1) ends with the same line, for the `save_pretrained` save and
   for the `glyd pack` one alike, after the Python traceback 0.27.0 does not catch ([logs/4-verify-027.txt](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/acceptance-v0.28.0rc1-l4-2026-10-04/oldsave/logs/4-verify-027.txt), [7-verify-027-cli-save.txt](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/acceptance-v0.28.0rc1-l4-2026-10-04/oldsave/logs/7-verify-027-cli-save.txt)): those two tracebacks are
   0.27.0's own handling of the refusal, the case under test, and not a failure of 0.28.0rc1 (which printed no traceback in any run, outside the deliberate Ctrl-C cases).

## Warnings and tracebacks

`python3 post_warnings.py .` over each run's server logs (`runs/*/glyd-logs`, [warnings.txt](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/acceptance-v0.28.0rc1-l4-2026-10-04/warnings.txt)); the acceptance itself fails a run on a traceback before a shutdown line or an allocator out-of-memory warning in the first four logs
(`glyd run` once, `glyd serve` once and with a key twice):

| Run | Server logs checked | Tracebacks | Allocator OOM warnings | `warning:` in the install output | WARNING lines besides vLLM's own notices |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `16gb-card` | 4 | 0 | 0 | 0 | none |
| `24gb` | 4, and 3 of the cases that stop a server on purpose | 0 in the 4 | 0 | 0 | 3: vLLM's notice on glyd's scheduler (below) |
| `16gb-card-served-script` | 4 | 0 | 0 | 0 | none |
| `cli` | no server | | | 0 | |

No failed check anywhere but the one expected. `grep -c Traceback` over every file of every run finds 4, all in the 24 GB run's three on-purpose logs: the Ctrl-C-twice-while-loading one has 1 (it has no shutdown line: vLLM's API server interrupted inside its own
engine start, which is what the case does), the Ctrl-C-once one has 1 after its shutdown line, and the killed engine's server log has 2 after its shutdown lines (vLLM's output handler finding the engine stopped, which the case does on purpose).
`with OOM` is in no file; the virtual environments' pip logs have no warning.

vLLM's own notices, in every run as in v0.27.0's: "Enforce eager set, disabling torch.compile and CUDAGraphs", "Inductor compilation was disabled by user settings" (glyd starts the server eager, on purpose), "Default vLLM sampling parameters have been
overridden by the model's `generation_config.json`", and at the server's stop "[shutdown] Process manager: force killing remaining process(es)". One more WARNING line comes with the lossless KV cache, in every server log where it is on (the 24 GB run's three
`glyd serve` starts, as in v0.27.0's): `[scheduler.py:224] Using custom scheduler class glyd_gpu.vllm_sched.make. This scheduler interface is not public and compatibility may not be maintained. ...` It is vLLM's own text, once a start; nothing failed.
The 16 GB runs have none: with the GeForce emulation (`--card 4080s`) the kv auto line is an INFO ("the KV cache stays vLLM's here").
One more line is no WARNING of vLLM's logger and is in every server log as in v0.27.0's, 4, 5 and 4 of them in the three runs here and there: Python's own `UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown`,
printed once as a server's process exits, after `Application shutdown complete`. And Open WebUI's uvx route prints two warnings of its own packages (pydub's ffmpeg, a SQLAlchemy table note) in `logs/webui-uvx.txt`.

## The site

`getglyd.com/install.sh` answers 200 and is **v0.27.0's script** (sha256 `51ee191e...e0`, 19,715 bytes, last modified 2026-10-04 12:15 UTC): the site does not change for a pre-release. So the served flow is run as the pre-release is tested, with the README's line and
the release branch's raw file in place of `getglyd.com` (`--served --prerelease --install-url REL`, the patch below): `curl -LsSf REL | sh`, from the network, installed `glyd==0.28.0rc1` and every package of the same list, 64 checks.
The 65th, **the script getglyd.com serves is the release's, byte for byte**, fails, as expected until the final release: `FAIL the script getglyd.com serves is not REL (sha256 51ee191e..., that file's 97d26dde...)`. The site's `install.sh` is to be
replaced at the final release, as every release's; the final release's run is then v0.27.0's command, `--served --install-url TAG_RAW` without `--prerelease`, whose last check compares the site's script with the tag's.

## What the script needed (scripts/acceptance-028.patch, committed as e8506743 on `acceptance-028`, pushed to `private` only)

1. `--prerelease`, with `--served` and `--install-url`: the served line is run with that URL's script in place of getglyd.com's, that copy is the one the again-cases (a program of the user's own, an update, a `glyd` ahead on PATH) run, and the site's script is still compared with it.
2. `--pip-refusal` pip-installs the version the script under test pins (`GLYD_VERSION="${GLYD_VERSION:-...}"`), where it took an unpinned `glyd[vllm]`, which is the last stable release (0.27.0), not the release candidate (v0.27.0rc2's record said so: "pass `--version`").
   Its check names the spec it installed (`pip install of glyd[vllm]==0.28.0rc1, no compiler`).

The runs needed nothing else.

## Seen, not a check

- On the 24 GB L4, `glyd doctor` (the last lines of the install too) now says `Ready: glyd run Qwen/Qwen3-14B` (Qwen3-14B packs to 20.5 GB: it fits with a 4,096-token context), and the line after it, `install.sh`'s fixed one, is `Next: glyd run Qwen/Qwen3-8B`
  (v0.27.0: the same 8B in both). The gate checks only that both lines are there. `glyd run Qwen/Qwen3-14B` was not run here.
- A start takes longer than v0.27.0's: vLLM's "Loading weights took" is 25.6 to 25.9 s in every warm start of the three runs (13.2 s in v0.27.0's 16 GB run), the model's load 27 to 29 s (16.5 s), and `glyd serve` is Ready in 53 s on a warm cache
  (v0.27.0: 40 s in the 16 GB run; the 24 GB run's `glyd run` 1 min 13 s against 1 min 0 s). v0.28.0 packs the embedding and the output layer as well as the Linears as the model loads (the servers' logs say "glyd: the embedding and the LM head packed too");
  how much of the added time that is was not measured.

## Files

`runs/NAME/`: `acceptance.out` and `summary.txt`; `glyd-logs/`, the servers' logs; `logs/`, what each step printed (`install.txt`, `freeze.txt`, `doctor*.txt`, `guard.txt`, `chat.txt`, `webui-*.check.txt`, `pip.txt`, ...). Files the box named `.log` are `.txt` here (the repository
ignores `*.log`); the text inside still says `.log`. `oldsave/`: the old-save check (`oldsave-run.out`, `logs/` of its seven steps, `answers-027.json`, the two environments' freezes and pip logs). `resolve/`: the pip resolves. `published-artifacts.txt`, `wheel-gate.txt`,
`gate.out`, `post_warnings.py`, `warnings.txt`, `scripts/`: what ran on the box and here (`fetch.sh` counts the served run's `FAILED (1)` line as a second FAIL).
