#!/usr/bin/env python3
"""The speed bar of glyd-engine/src/machine.rs (GPUS), now the harness's: `glyd-engine bench --bar`.

    python3 scripts/gpu_bar.py ENGINE.json VLLM.json "NVIDIA GeForce RTX 4080 SUPER" 8.9 [--family Qwen3] [--released YYYY-MM-DD] [--stale-ok]
    python3 scripts/gpu_bar.py e1.json,e2.json,... v1.json,v2.json,... "NVIDIA GeForce RTX 4080 SUPER" 8.9   # the runs of several files together

ENGINE.json and VLLM.json are results of `glyd-engine bench --against URL --suite bar` (glyd-bench/1, docs/bench.md), one for each server, measured by the one program: the same prompts, loads and
statistics, so a table of the two compares them and nothing else. The summary files of the earlier benchmarks/gpu/rtx4080s-bar-2026-11-20 are read as well. A GPU passes where, at every load, the
engine's decode tokens/s a user and in all and its time to first token are at least vLLM's (within 3%) and its peak memory is lower: it prints its line for GPUS; a failing one prints each load it
failed and no line. Exit 0 passes, 1 fails. docs/adding-a-gpu.md.

This file only finds the program (GLYD_ENGINE_BIN, then target/release/glyd-engine of this checkout, then glyd-engine or glyd on PATH) and hands it the arguments.
"""
import os
import shutil
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) < 3:
    sys.exit(__doc__)
candidates = [os.environ.get("GLYD_ENGINE_BIN"), os.path.join(here, "..", "target", "release", "glyd-engine"), shutil.which("glyd-engine")]
engine = next((c for c in candidates if c and os.path.isfile(c)), None)
if engine is None and shutil.which("glyd"):
    engine = shutil.which("glyd")
if engine is None:
    sys.exit("gpu_bar.py: no glyd-engine (set GLYD_ENGINE_BIN, build it with cargo build --release -p glyd-engine, or install it)")
cmd = [engine, "bench", "--bar", sys.argv[1], sys.argv[2]]
rest = sys.argv[3:]
if len(rest) >= 2 and not rest[0].startswith("--"):
    cmd += ["--gpu-name", rest[0], "--cc", rest[1]]
    rest = rest[2:]
cmd += rest  # (--family NAME, --released DATE, --stale-ok, --noise X: glyd-engine bench --bar)
sys.exit(subprocess.call(cmd))
