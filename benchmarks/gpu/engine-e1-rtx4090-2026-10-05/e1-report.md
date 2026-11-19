# E1: glyd-engine as what `glyd run` and `glyd serve` start (branch engine-v029, private repo glyd-gpu-private)

Worktree /Users/suryakoritala/glyd-engine029, branch `engine-v029` (pushed to `private` only; `origin` was never pushed to). Report written by the E1 worker.
S = /private/tmp/claude-501/-Users-suryakoritala-alatirok/a9feedc0-e067-4d76-bca6-e4b6b9538345/scratchpad


## Summary

Status: DONE (see "what is left" and the concerns in the reply). engine-v029 = 6c54b3c1 (private only). The v0.28.0 code is merged; the engine ships as `glyd_gpu/glyd-engine` in both Linux wheels (built by release.yml's new engine job for sm_80/86/89/90a/100/120, no PTX, stripped, gated like the library, with the engine's Rust sources gated too);
`glyd run`/`glyd serve` (Python tool and Rust glyd) hand Qwen3 models to it and everything else to vLLM, one line saying which, `GLYD_ENGINE=0` for vLLM; CI dispatch green on both workflows with nothing published; the exact install.sh flow on one RTX 4090 pod: 35 of 36 checks, the 36th a test-tool assumption since fixed.

## Commits (all signed off as Surya Koritala, no AI attribution)

| hash | what |
| :--- | :--- |
| 5b9b1104 | Merge private/main (v0.28.0, 4642e053) into engine-v029 |
| 18a4809c | glyd-engine: a release build (no dev commands or tuning variables, the corpus compressed, tables of the plugin's rule 8, machine code for Blackwell, no PTX) |
| 5d67413b | check_wheel: the gate reads the engine's program and its Rust sources; bump_version keeps the engine's version |
| 05f91279 | glyd-gpu wheel: the engine's program beside the libraries, executable (build.sh GLYD_GPU_ENGINE, setup.py, pyproject) |
| a0846547 | glyd run / glyd serve: the engine runs the models it supports (Python tool and Rust glyd), one line says which, GLYD_ENGINE=0 keeps vLLM |
| dd0ccee8 | release.yml: the engine's job per architecture and its program in the wheels; ci.yml: the engine's tests; check_install runs the engine; docs |
| d5a96fd5 | a test that the tables `calibrate` writes are current and read back; check_install: the hand-off finds the program |
| 62e5ea16 | `glyd doctor` says whether the engine is there; the docs say what the engine's API refuses and the exact auto settings |
| c9d7e16b | gpu/vllm/README: the running server `glyd run` attaches to is vLLM's |
| 6c54b3c1 | serve_smoke: the over-long prompt is longer than the server's own context (max_model_len) (found by the pod run; head of the branch, pushed) |

## 1. The merge (5b9b1104)

`git merge private/main` (4642e053, the released v0.28.0): no textual conflict (git auto-merged `Cargo.toml` and `Cargo.lock`). Both sides kept:
main's `version = "0.28.0"` (glyd, glyd-gpu, glyd-store, and the three entries of Cargo.lock) with engine-v029's workspace member `glyd-engine`, its `exclude` entry and its lock packages.
Decisions the merge left to make (not conflicts):
- the glyd-engine crate stayed at 0.27.0-rc.1: it is now 0.28.0 and **part of `scripts/bump_version.py`** (its CODE table has glyd-engine/Cargo.toml and a fourth Cargo.lock entry; the selftest's counts follow), so `bump_version.py 0.29.0` moves it with the rest and `--check` fails if it drifts;
- main's new library code (gpu/glyd_gpu.cu: the tiered layout's product for 1 to 16 tokens, `GLYD_GPU_STEP`) is behind the unchanged C API 15: the engine's binding is untouched, and `check_wheel.py`'s INTERNAL_ENV (which main extended with STEP) is kept, the engine gate uses it;
- the engine's own measurements (benchmarks/gpu/engine-*) were made on the library before main's step kernel; none of them were redone.
Tests after the merge: `cargo test --release --workspace` 281 passed, 0 failed, 34 ignored ($S/engine/e1-cargo-test-release-merge.log; plain debug `cargo test` fails 6 codec tests with "attempt to multiply with overflow": the project runs `cargo test --release`, CONTRIBUTING.md,
and the failures are in src/cm.rs, ldm.rs, reflate, safetensors, deflate, none of which this branch touches); Python (Docker python:3.12 with torch CPU, ci.yml's `sources` job as it runs them): test_gpu.py exit 0 (the GPU and torch tests skip), test_onboard.py 64 of 64
(65 of 65 with the hand-off's test).

## 2. Packaging

What ships: `glyd_gpu/glyd-engine` in each Linux wheel (x86_64 and aarch64), beside `libglyd_gpu_cuda12.so` and `_cuda13.so`: an executable (mode 755 in the wheel: setup.py's BuildPy sets it, check_wheel.py checks it), 8.4 MB (x86_64) and 7.6 MB (aarch64) stripped, a wheel of 33.8 MB where it was 30.0.

**The release build of the engine** (glyd-engine/Cargo.toml: a `dev` feature, on by default; `cargo build --release --no-default-features` with `GLYD_ENGINE_RELEASE=1` is the shipped one):
- the development build keeps everything it had (the checks, the proofs, the `GLYD_ENGINE_*` tuning variables); the release has none of it: `src/main.rs` is the release commands (run, serve, doctor, login, supports, fetch, prepare, `--version`, `--help`), `src/devcmd.rs`, `kcheck`, `kvtest`, `report`, `stress` are `#[cfg(feature = "dev")]`,
  every tuning variable is read through `tune!("NAME")` (a macro: `None` in the release build, which therefore does not name the variable), the dev flags of `args.rs` and the scheduler's trace are gated; what the release reads: `GLYD_CACHE`, `GLYD_GPU_LIB`, `GLYD_API_KEY`;
- build.rs: `GLYD_ENGINE_RELEASE=1` takes sm_80, 86, 89, 90a, 100 and 120 (nvcc without Blackwell, or none, is an error), no PTX; the one kernel named with a method word (`kv_page_hp`) is `kv_leaf_hp` (the library's own word for a page);
- the calibration corpus (182 KB of plain text) is `data/kvcorpus.bin`, the zlib file the Python package ships, inflated once at run time (`calib::corpus()`, flate2 was already in the lock through ureq): plain, it carried the repository's own docs, and with them every method word, in the binary;
- the KV tables' file now has the plugin's neutral key names, as the plugin writes them (rule VERSION 8: `store`, `bay`, `hard`, `leaves`, `..._leaf`, `second_layers`; engine/src/kvcal.rs, calib.rs, kv.rs): the algorithm is unchanged, and the Rust port was checked again against the plugin's Python
  (tests/fixtures/kvcal.json regenerated by tools/kvcal_fixtures.py with the Python 3.12 + torch image: only `sweep` (renamed keys) and `constants.version` differ from the old fixture). A tables file of another version (an earlier engine's, a dev box's cache) is deleted and made again (`store::current_tables_for`, tested);
- the strings the Rust sources had that say a method word (about 50: "pages", "tiered layout", "bits a weight", "staging", the tables' keys, the library calls' labels) are reworded ("blocks", "layout", "work cache" ...);
- `-p` is accepted for `--prompt` (glyd run's short form), `--version`/`--help` exist.
`cargo test -p glyd-engine`: 66 passed (dev) and 64 (release features); the same two runs are ci.yml's new `engine` job.

**The gate** (glyd-gpu-py/check_wheel.py; its tests in glyd-gpu-py/tests/test_gpu.py: `test_check_engine`, `test_check_engine_sources`, and the engine cases of `test_check_wheel`): the library's gate extended, in three places.
- `--engine PROGRAM [--engine-source glyd-engine]` (the engine job): ELF for the architecture; no symbol table or debug section; **no export**; NEEDED only libc, libm, libdl, libpthread, librt, libgcc_s, ld-linux; a CUDA fat binary in it (found by its magic and size) whose machine code, read with cuobjdump, is exactly sm_80, 86, 89, 90a, 100, 120 with **no PTX**,
  and whose kernels' names carry no method word; no method phrase (PHRASES) and no word of STABLE (tiered, mantissa, exponent, huffman, entropy, arena, staging, ...) anywhere in the program (the fat binary's own bytes blanked: they are not text), but minijinja's `exponent must be an integer`;
  no `GLYD_` name the release build does not read (every `GLYD_` occurrence is a name's start: Rust stores adjacent strings run together, and an x86-64 compiler puts the names it compares in the instructions as 8-byte immediates: a name's first 8 bytes may be an allowed name's).
- `--engine-source DIR` (alone in ci.yml, with `--engine` in the engine job, with the wheel in the wheels job): the Rust sources **as the release build compiles them** (a module declared `#[cfg(feature = "dev")]`, an item, statement or match arm under it, a `#[cfg(test)]` tail, `tune!("NAME")` and comments are left out; a Rust lexer in Python: string, raw and byte literals, chars, lifetimes):
  every string that is text must say no METHOD word (the names of inline format arguments are not in a binary and are not read), every name or key (a string of word characters) none of IDENT_DENY (but glyd.json's own `tiers` key), no `GLYD_` name the release does not read; and the page the engine serves (data/chat.html).
- in a wheel: `glyd_gpu/glyd-engine` required where the libraries are (`--no-libs` or `--no-engine` for the others), executable, for the wheel's architecture, its sha256 the one the engine job wrote (`--sums .../glyd-engine.sha256`, as the libraries'), and no engine where there is no library.
- `check_install.py` (the wheels job, in every CPython): the engine runs from the installed wheel (`--version` is the wheel's version, `doctor` finds the library beside it, `supports` of a Qwen3 config is 0 and of a Llama config 3, with no GPU and no network), and `glyd_gpu.engine.find()` (the compiled module) finds it.
Residual (said, not hidden): the engine's own kernel names are readable (`rmsnorm`, `gemm3_...`, and the four KV calibration kernels `kv_hist`, `kv_outside_total`, `kv_leaf_hp`, `kv_sweep`): they carry no word of IDENT_DENY, but unlike the library's (`gk_N`) they are not neutral; a `gen_names`-style renaming of engine.cu is the way if you want it.

**The workflows.** release.yml: a new job `engine` (matrix x86_64 on ubuntu-latest, aarch64 on ubuntu-24.04-arm; container nvidia/cuda:13.0.3-devel-rockylinux8 as the libraries; gcc-toolset-13, rustup with Rust 1.98.1, `GLYD_ENGINE_RELEASE=1`, `CARGO_PROFILE_RELEASE_STRIP=symbols`,
`cargo build --release --locked -p glyd-engine --no-default-features`; checks: glibc 2.28, stripped, `check_wheel.py --engine ... --engine-source glyd-engine` (with cuobjdump), the sha256 file; artifact `glyd-engine-linux-<arch>`); the `wheels` job needs it, passes `GLYD_GPU_ENGINE` to build.sh, and checks the wheel with `--sums libs/*.sha256 engine/*.sha256 --engine-source glyd-engine`.
ci.yml: a job `engine` (cargo test of glyd-engine in both builds, the sources gate, the Rust glyd's hand-off test). actionlint passes on both. **A dispatch does not publish** (read before running): `plan` sets `publish=false` for `workflow_dispatch`, and `publish` runs only for `github.event_name == 'push'` on a `v*` tag, in the `pypi` environment that takes `v*` tags alone.
Replicated locally before any dispatch, step for step from the workflow's own `run:` blocks (Docker on this Mac, aarch64 native, the x86_64 engine and Rust glyd emulated): the engine job (build 1m43s, check) in nvidia/cuda:13.0.3-devel-rockylinux8, the wheels job (build, auditwheel `manylinux_2_28_aarch64`, check_wheel, install and test in CPython 3.12) in quay.io/pypa/manylinux_2_28_aarch64 with the libraries of the published 0.28.0 aarch64 wheel
($S/engine/wheel-aarch64/run.log, $S/engine/job-engine-out/run.log); the x86_64 engine built and gated in an emulated Rocky 8 ($S/engine/out-rel-x86/): 3 `[env]` findings that were the 8-byte immediates above, which the gate now knows.

## 3. The hand-off

`glyd run MODEL` / `glyd serve MODEL`:
- **Python tool** (glyd-gpu-py/glyd_gpu/engine.py, called first by `run.main`): `GLYD_ENGINE=0` (also off, no, false, vllm) keeps vLLM; `GLYD_ENGINE=PATH` names an engine program; else the engine beside the library in the package, else `glyd-engine` on PATH. It asks `glyd-engine supports MODEL`: 0 runs the engine in this process's place (`os.execv`), **3 (not a Qwen3 dense model, rope scaling) is vLLM**, 1 (could not tell: a name the Hub does not show, offline) is the engine's own refusal, in words, a crash or a missing program is vLLM.
  A missing model, `--help` and vLLM's flags after a lone `--` are the vLLM path's. One line on stderr says which: `glyd run: Qwen/Qwen3-8B on glyd-engine (GLYD_ENGINE=0: vLLM instead).` or `glyd run: Qwen2.5-7B-Instruct is a Qwen2ForCausalLM model, and this engine runs Qwen3 models so far; using vLLM.` (the engine's own reason), `...: GLYD_ENGINE=0; using vLLM.`, `...: glyd-engine is not installed here; using vLLM.`
- **Rust glyd** (src/bin/glyd.rs `local_chat`): the same decisions where it finds an engine itself (GLYD_ENGINE=PATH, beside it, on PATH); where it finds none it passes the command to the Python tool as before (which has the engine of its package, and decides). Where it has said which path a command takes it starts the tool with `GLYD_ENGINE=0`, and the tool says nothing more (one line, never two; `GLYD_FORWARDED` was the existing loop guard). The old dev knob `GLYD_RUNNER` is gone (never released).
  `doctor` and `login` stay the tool's (the engine's where Rust finds a standalone engine).
- `glyd doctor` has a row "Glyd engine" (found where, or GLYD_ENGINE=0, or not in this install).
- **Defaults**: nothing is passed but the flags the user gave (`-p`, `--prompt`, `--context`, `--users`, `--port`, ...): the engine's own settings from the GPU's memory are used (`fit::plan`: 8,192 tokens of context where there is room, the model's own limit at most, not below 4,096; 4 users by default, the cache's pool for them limited to what the memory has).
- Tests: tests/cli_python_tool.rs (Rust: 8, two new: the engine runs the model, and every way to vLLM with exactly one line and the tool told once), test_onboard.py `test_the_engine_takes_glyd_run_where_it_runs_the_model` (the same cases and the real process: line on stderr, answer on stdout), `test_doctor`.
- Docs (public, what and measured only): CHANGELOG `v0.29.0 (Unreleased)` (numbers quoted from the L4 records), gpu/vllm/README.md (which model runs where, what the engine's API refuses), glyd-gpu-py/README.md.
Known differences from the vLLM path (also in "what is left"): `glyd run` does not attach to a `glyd serve` that is up (the engine has no marker); the engine's API refuses `tools`, `n`>1, `response_format`, `logprobs`, `logit_bias` (400, in words); `--hf-token` is not read by the engine (HF_TOKEN / `glyd login` are); the engine uses the first GPU.

## 4. The CI dispatch (build only, nothing published)

`gh run`s on branch engine-v029 of surya-koritala/glyd-gpu-private (the repository's only workflows; `gh workflow run release.yml --ref engine-v029 -f pythons=3.12`):
- release.yml run **37381063168** at **c9d7e16b** (the final head): **success**, every job: plan, gpu-lib x4, **engine x2 (x86_64, aarch64)**, **wheels x2**, collect (twine check --strict); **publish: skipped** (as read: a dispatch has `publish=false`). The wheels are the run's artifact `glyd-gpu-wheels`, kept at $S/engine/ci-wheels-2/:
  `glyd_gpu-0.28.0-cp310-abi3-manylinux_2_28_x86_64.whl` 34,282,675 bytes (sha256 91476c71...3381) and `..._aarch64.whl` 33,827,498 bytes, each with `glyd_gpu/glyd-engine` (8,443,344 / 7,631,056 bytes, mode 755, sha256 3cdd0a50... / 85796565...); the wheel check `ok`, auditwheel `manylinux_2_28_{x86_64,aarch64}`, install and check_install in CPython 3.12 (the engine ran from the installed wheel).
- release.yml run 37380019654 at dd0ccee8 (the same pipeline before the last small commits): success, too (its engine binaries are byte for byte the second run's: the build is reproducible).
- ci.yml run **37380629264** at d5a96fd5: success: versions, sources (3.10 and 3.14), the compiled wheel, vLLM plugin (vLLM installed), the Rust crate, and the new **engine** job (66 + 64 tests, the sources' gate, the Rust glyd's hand-off tests).
Total CI time of one release dispatch: about 17 minutes.

## 5. The user flow on a GPU pod (ONE pod, 9 minutes)

Pod `glyd-bench-e1-4090` (id o5nsasg1ls4v8p, Runpod secure cloud, $0.74/h, image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404, deleted by rp_run.sh: "pods left: []"): **NVIDIA GeForce RTX 4090** (Ada, 24,564 MiB), driver 595.91.07 (CUDA 13.2), Ubuntu 24.04.3, Python 3.12.3, gcc 13.3, 48 cores, 503 GB RAM.
Created 22:34:48 UTC, job 22:35 to 22:44, 9 minutes in all (rp_run's own count). Pods used for this task: **1 of the 2 allowed**.
Run as: `GPU="NVIDIA GeForce RTX 4090" MAX_MIN=52 NAME=e1-4090 UPLOAD="install.sh glyd_gpu-0.28.0-...-manylinux_2_28_x86_64.whl glyd serve_smoke.py" bash $S/rp_run.sh $S/engine/e1_pod_job.sh`.
Uploaded: the branch's `scripts/install.sh` (unchanged from v0.28.0's), **the x86_64 wheel of release run 37381063168** (the final head's), the Rust `glyd` (x86_64, glibc 2.28, built in the emulated Rocky 8 from the branch's src/bin/glyd.rs), serve_smoke.py.
The job ($S/engine/e1_pod_job.sh; dry-run first against stubs in a container, shellchecked) did what a user does: `curl -LsSf http://127.0.0.1:8765/install.sh | sh` (the script served from the pod's own http.server: the exact command, from the branch's file instead of getglyd.com; `UV_FIND_LINKS=~/wheels` so that uv takes the CI wheel where it asks for glyd-gpu 0.28.0,
the version PyPI has too: tested beforehand locally that uv prefers the flat index for an equal version), then the commands below. Results: **$S/ceiling/results-e1-4090/** (summary.txt, job.log, one .out/.err a step, glyd-logs/, cache.txt, nvidia-smi-end.txt).

**35 checks passed, 1 failed** (the failure is the test tool's, below). In order:
| step | result |
| :--- | :--- |
| `curl ... install.sh | sh` | exit 0 in 61 s (uv, glyd 0.28.0, glyd-gpu 0.28.0 from the CI wheel, vLLM 0.30.0, torch 2.13.0): no traceback; `uv tool list`: glyd v0.28.0 |
| the engine in the tool | `.../site-packages/glyd_gpu/glyd-engine`, executable, **sha256 equal to the program in the CI wheel** (3cdd0a50...); `glyd-engine --version` = glyd-engine 0.28.0; `glyd doctor` (install.sh's last step) has the row `ok Glyd engine <that path>: glyd run and glyd serve start it for Qwen3 models (GLYD_ENGINE=0: vLLM instead)`, GPU ok, driver ok, "Ready: glyd run Qwen/Qwen3-14B" |
| `glyd run Qwen/Qwen3-8B --prompt "Say hello in five words." --no-think` | exit 0, **143 s from an empty machine to the answer** (the engine downloaded 16.4 GB, packed 12.4 GB in 26 s in layout mma "Ada: mma, the smallest layout", made the KV tables in 21 s, "Ready in 2m22s: Qwen3-8B on the NVIDIA GeForce RTX 4090, 10.3 GiB of weights, 8192 tokens of context"); stderr's first line `glyd run: Qwen/Qwen3-8B on glyd-engine (GLYD_ENGINE=0: vLLM instead).`, exactly one; stdout `Hello, how can I help?`; the GPU idle afterwards |
| `glyd serve Qwen/Qwen3-8B --port 8000` + requests | ready in 4 s ("4 users at once", 8192 tokens), the same one line; `/v1/models` lists the model (max_model_len 8192); a chat completion over the API ("What is 2 plus 3?" -> "Five."); the chat page at `/`; **serve_smoke.py (the engine's API check): 19 of 20 ok**: chat, streaming = non-streaming, seeds, stop strings, completions, prefix reuse (2,024 tokens: 0.22 s then 0.023 s), a next turn, bad JSON, `n`, `tools`, `logprobs`, `response_format` refused in words, two requests at once;
the one FAIL, "a prompt longer than the context is refused", is the tool's assumption: it sent 6,000 tokens (`"x " * 6000`) which the engine correctly **accepted** into its 8,192-token window (fixed in 6c54b3c1: the prompt is `max_model_len + 500` tokens; not rerun on a GPU: it is a one-line change of a test tool, and a second pod for it is not worth a GPU). SIGTERM stopped the server and the GPU was idle |
| `glyd run Qwen/Qwen2.5-1.5B-Instruct --prompt ...` (a model the engine does not run) | exit 0 in 89 s (3.1 GB downloaded, vLLM started and answered "Hello there! How can I help you today?"); stderr: `glyd run: Qwen2.5-1.5B-Instruct is a Qwen2ForCausalLM model, and this engine runs Qwen3 models so far; using vLLM.` once; the GPU idle afterwards |
| `glyd run Qwen/Qwen3-8B ...` again | **4 s** to the answer ("Ready in 2s") |
| the Rust `glyd` first on PATH (2.7 MB, x86_64): `glyd --version` | glyd 0.28.0 |
| Rust glyd -> Python tool -> engine (no engine of its own to find) | exit 0, the answer, one line (the tool's), 3 s |
| Rust glyd with `GLYD_ENGINE=<the engine>` (it finds and starts the engine itself, `supports` first) | exit 0, the answer, one line (the Rust one: the same text), 4 s |
| the same with Qwen2.5-1.5B-Instruct (the engine says 3) | exactly one line (`glyd run: Qwen2.5-1.5B-Instruct is a Qwen2ForCausalLM model, ...; using vLLM.`, the Rust glyd's), the Python tool told `GLYD_ENGINE=0` and silent, vLLM ran and answered (35 s, exit 0) |
| `GLYD_ENGINE=0` with the Rust glyd and with the Python tool (Qwen3-0.6B) | `glyd run: GLYD_ENGINE=0; using vLLM.` once in each, then vLLM (Python tool: answered "Hello there!" in 44 s; the Rust one was stopped at 45 s by the job's own cap, after its line) |
| the whole | no traceback and no out-of-memory in any step's output; the GPU idle at the end (nvidia-smi-end.txt) |
Not done on the pod (said): a second GPU type, the 16 GB desktop card, long runs under load (the engine's records have the speed numbers; this was a flow test, nothing measured here is a speed claim), Open WebUI.
Notes from the run: `glyd doctor`'s "Models that fit" table is vLLM's estimate (it says a 40,960-token context for Qwen3-8B, where the engine runs 8,192 by default): wording to reconcile; the engine's first run needs no CUDA toolkit on the machine (the pod had one, unused: the engine opens libcuda and its library only).

## 6. What is left for a v0.29 release

Release mechanics (nothing of it done here: the owner's call):
1. Versions: `python3 scripts/bump_version.py 0.29.0rc1` (it now moves glyd-engine too), an rc through the release workflow on a tag (that is the only thing that publishes), install.sh with `GLYD_VERSION=0.29.0rc1`, then the final: CHANGELOG heading is already `## v0.29.0 (Unreleased)` (it becomes dated), ROADMAP/docs by the script, `scripts/bump_formula.py`.
2. The public repository: it must get the Rust glyd's hand-off (src/bin/glyd.rs and tests/cli_python_tool.rs) and the public `glyd` Python package pinning `glyd-gpu[...]==0.29.0` (the glyd-shim branch's bindings/python: its pyproject, version, `glyd.cli`); the engine's sources stay private (they are in the private repo only, `glyd-engine/` is excluded from the codec crate and the wheel carries the program, not the sources). The public CHANGELOG/README texts are in this branch (CHANGELOG quotes the L4 numbers of the engine records, which are in the private repo: publish the records or reword).
3. License: `gpu/LICENSE` (what the wheel ships) names `gpu/`, `glyd-gpu/` and `bindings/python/glyd/gpu/` as the Licensed Work, not `glyd-engine/`; `glyd-engine/LICENSE` is the same text but with the old gmail contact (the others have suryakoritala@getglyd.com). Your wording to decide (I did not touch license text).
4. acceptance.sh (branch acceptance-028) is vLLM's flow: its README blocks, `--model`, tool-call and Open WebUI checks need an engine-aware run (the engine's API refuses `tools`: the tool-call check needs `GLYD_ENGINE=0` or a vLLM model).
5. The L4 and RTX 4080 SUPER (16 GB, a desktop) flows with the engine; A100/H100 with the final wheel (the engine's records are from the engine-mvp builds, before main's step kernel and before this release build); Blackwell (sm_100/sm_120) machine code is built and gated, never run on hardware.

Engine behaviour that differs from the vLLM path (decide if it blocks the release):
- `glyd run MODEL` does not attach to a `glyd serve` of the same model that is up (vLLM's path has a marker file); the engine's `run` would load a second copy (its fit check refuses in words where the GPU has no room).
- The engine's API refuses `tools`, `n`>1, `response_format`, `logprobs`, `logit_bias` (400, in words); tool calling is vLLM's.
- Flags: `--hf-token` is ignored by the engine (HF_TOKEN and `glyd login` work); the engine takes the first GPU (the Python tool picks by free memory and sets CUDA_VISIBLE_DEVICES for vLLM); users default to 4 (a fixed number, the cache's pool for them is what fits), not memory-sized.
- The engine's chat has no `glyd run` marker/logs directory (the vLLM path keeps `~/.local/state/glyd/logs`); progress goes to the terminal only.
- The engine's own kernel names are readable (see 2): neutral names are a generator away if you want the library's rule applied to engine.cu.
- Dev machines with caches from earlier engine builds: tables of rule 7 are deleted and made again automatically (a first run of a model takes its minute again).
- `GLYD_ENGINE_*` variables (and all the dev commands) exist only in a development build now: the GPU job scripts under glyd-engine/jobs use the default build, so they are unchanged.

## Housekeeping

- Everything is on `private/engine-v029` (head 6c54b3c1); `origin` (public) was never pushed to and has no such branch. No token was read, printed or stored by me (`gh` and `runpodctl` used their own credentials).
- Scratch of this task under $S/engine/: the Docker scripts (dk/), the local replicas' logs (wheel-aarch64/, job-engine-out/, out-rel/, out-rel-x86/, out-glyd-x86/), CI's artifacts (ci-wheels-1/, ci-wheels-2/, ci-engine/), the pod job and its dry-run stubs (e1_pod_job.sh, dry/), the pod run's log (e1-pod-run.log) and results ($S/ceiling/results-e1-4090/). The Docker volumes (cargo, rustup, target caches) and the python test image this task made were removed; the CUDA images stay.
- Compute: 1 pod (9 minutes, $0.74/h), CI: two release dispatches (about 17 minutes each; the first was queued behind by the workflow's concurrency group, so they ran one after the other) and one ci.yml dispatch.
