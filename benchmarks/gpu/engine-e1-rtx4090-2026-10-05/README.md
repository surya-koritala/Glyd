# The engine hand-off's user flow on an RTX 4090 (Runpod, 2026-10-05)

Functional record of the engine hand-off on pre-2026 models: pass/fail only, not for publication or headline numbers.

The pod check of the E1 work (`glyd-engine` as what `glyd run` and `glyd serve` start, branch `engine-v029`), cited by
`e1-report.md` (the worker's report, copied here): **35 of 36 checks passed**; the 36th was the API check's own assumption
(`serve_smoke.py`'s over-long prompt was shorter than the server's context), fixed in `6c54b3c1`. Model: the engine's family at the
time (pre-2026; named in the job script), pass/fail only, nothing published. One pod, 9 minutes, 22:35-22:44 UTC.

```sh
GPU="NVIDIA GeForce RTX 4090" MAX_MIN=52 NAME=e1-4090 \
  UPLOAD="install.sh glyd_gpu-0.28.0-cp310-abi3-manylinux_2_28_x86_64.whl glyd serve_smoke.py" bash rp_run.sh e1_pod_job.sh
```

(`install.sh` the branch's, the wheel from CI's release build of the branch, `glyd` the branch's Rust glyd for x86_64.)

What it checked (`results/summary.txt`): install.sh as `curl | sh`, the engine in the tool is CI's (sha256), `glyd-engine
--version`, `glyd doctor` lists it, `glyd run` from an empty machine and warm, one line saying which runs the model, `glyd serve`
and a chat, `GLYD_ENGINE=0`, a model the engine does not run going to vLLM, the Rust `glyd` hand-off, the GPU let go at the end.
Times in `results/times.txt`.

Files: `results/` (everything the job wrote), `job/e1_pod_job.sh`, `job/serve_smoke.py`, `job/rp_run.log`, `e1-report.md`.
