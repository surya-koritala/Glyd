#!/usr/bin/env python3
"""The gate on the engine's program (check_wheel.py --engine) caught, each kind of leak: one injected into a scratch copy of the release program, and the gate must fail on it.

    python check_engine_leaks.py PROGRAM [--engine-source ../glyd-engine] [--table-header ../gpu/glyd_gpu.h]

release.yml's engine job runs it after the gate, on the program it built, where nvcc, cuobjdump and objcopy are. The program itself must pass first. Then, one scratch copy for each leak:

  text         a method word, a method phrase, a development command, a development flag, a variable that tunes the code, a count of a development build, a wording of how, the build
               machine's path, a kernel of the sources by name, a function of the library that its table hides: appended to the program as strings
  sections     a symbol table and debug sections (objcopy)
  machine code a second fat binary appended, made here by nvcc: a kernel under a readable name, a device function under one, line information, debug information, PTX, an architecture
               that is not the six, one short of them
  the tool     the gate run where cuobjdump is not on PATH: it must fail, not pass

Exit status 1 where the program does not pass, or a leak is not caught (a line for each)."""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_wheel as cw  # noqa: E402

ARCHS = ("80", "86", "89", "90a", "100", "120")


def fatbin(directory, name, source, archs=ARCHS, extra=()):
    """A fat binary nvcc makes of `source` for `archs` (a string of CUDA), as the engine's own build does: the bytes."""
    src, out = os.path.join(directory, name + ".cu"), os.path.join(directory, name + ".fatbin")
    with open(src, "w") as f:
        f.write(source)
    cmd = [os.environ.get("NVCC") or shutil.which("nvcc") or "nvcc", "-fatbin", "-O3", "-o", out]
    for a in archs:
        cmd += ["-gencode", f"arch=compute_{a},code=sm_{a}"]
    cmd += list(extra) + [src]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"nvcc could not make {name}: {done.stderr[-400:]}")
    with open(out, "rb") as f:
        return f.read()


def probe(kernel, helper=None):
    """A small CUDA source with a kernel of this name (and, if given, a device function that is not inlined, which the kernel calls)."""
    call = f"p[0] = {helper}(p[1]);" if helper else "p[0] = 1;"
    fn = f"__device__ __noinline__ int {helper}(int x) {{ return x * 3 + 1; }}\n" if helper else ""
    return fn + f'extern "C" __global__ void {kernel}(int* p) {{ {call} }}\n'


def objcopy(directory, program, *args):
    """The program put through objcopy with args (the bytes of the result)."""
    out = os.path.join(directory, "objcopy.out")
    done = subprocess.run([shutil.which("objcopy") or "objcopy", *args, program, out], capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"objcopy {' '.join(args)}: {done.stderr[-300:]}")
    with open(out, "rb") as f:
        return f.read()


def rodata(directory, program, text):
    """The program with a section of data (loaded, read-only, not code) that holds text added: where a program's strings are."""
    part = os.path.join(directory, "leak.txt")
    with open(part, "wb") as f:
        f.write(b"\0" + text.encode() + b"\0")
    return objcopy(directory, program, "--add-section", f".leak={part}", "--set-section-flags", ".leak=alloc,load,readonly,data")


def leaks(directory, program, data, kernels, internal):
    """[(what, the bytes of the program with it, the text the gate's line says)]: every kind of leak."""
    text = lambda t: data + b"\0" + t.encode() + b"\0"
    a_kernel = sorted(kernels)[0] if kernels else "kv_hist"
    a_function = sorted(internal)[0] if internal else "glyd_gpu_kv_second_bytes"
    out = [
        ("a method word, in the data a program's strings are in", rodata(directory, program, "the warps of a kernel"), "[words]"),
        ("a method phrase", text("the tiered layout"), "[words]"),
        ("a development command", text("make tables for more: glyd-engine kvcal MODEL OUT.safetensors --positions N"), "[dev]"),
        ("a development command's name", text("rowcheck"), "[dev]"),
        ("a development flag", text("no-graph"), "[dev]"),
        ("a variable that tunes the code", text("GLYD_ENGINE_TILE"), "[env]"),
        ("a count of a development build's stats", text('{"steps":1,"lost_blocks":2}'), "[dev]"),
        ("a wording of how", text("its keys and values need more of the cache's extra memory than there is"), "[dev]"),
        ("a pass count", text("pass 1 of 3"), "[dev]"),
        ("a block's shape", text("rows a multiple of 64, columns of 16"), "[dev]"),
        ("the machine that built it", text("/github/home/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/ureq-3.4.2/src/lib.rs"), "[path]"),
        ("a kernel of the sources by name", text(a_kernel), "[kernel]"),
        ("a function of the library that its table hides", text(a_function), "[name]"),
        ("a symbol table", objcopy(directory, program, "--add-symbol", "leak=.text:0,global,function"), "symbol table"),
        ("debug sections", objcopy(directory, program, "--add-section", ".debug_info=" + program), "debug"),
    ]
    probes = {
        "a kernel under a readable name": (probe("kv_hist"), {}),
        "a kernel under a name of the sources": (probe(a_kernel), {}),
        "a device function under a readable name": (probe("gk_9999", "block_sum"), {}),
        "line information": (probe("gk_9999"), {"extra": ("-lineinfo",)}),
        "debug information": (probe("gk_9999"), {"extra": ("-G",)}),
        "PTX": (probe("gk_9999"), {"extra": ("-gencode", "arch=compute_80,code=compute_80")}),
        "a seventh architecture": (probe("gk_9999"), {"archs": ARCHS + ("75",)}),
        "an architecture short": (probe("gk_9999"), {"archs": ARCHS[:-1]}),
    }
    expect = {"a kernel under a readable name": "[kernel]", "a kernel under a name of the sources": "[kernel]", "a device function under a readable name": "[kernel]", "line information": "line or debug",
              "debug information": "debug", "PTX": "holds PTX", "a seventh architecture": "machine code for", "an architecture short": "machine code for"}
    for what, (source, kw) in probes.items():
        out.append((what, data + fatbin(directory, what.replace(" ", "_"), source, **kw), expect[what]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("program")
    ap.add_argument("--engine-source", help="the glyd-engine directory (the names of its kernels)")
    ap.add_argument("--table-header", help="gpu/glyd_gpu.h (the functions of the library that its table hides)")
    a = ap.parse_args(argv)
    cuobjdump = shutil.which("cuobjdump")
    missing = [t for t in ("cuobjdump", "nvcc", "objcopy") if not (shutil.which(t) or (t == "nvcc" and os.environ.get("NVCC")))]
    if missing:
        print(f"FAIL the leaks cannot be injected here: {', '.join(missing)} not on PATH (the engine job has them)")
        return 1
    kernels = cw.engine_kernel_names(a.engine_source) if a.engine_source else set()
    internal = cw.internal_names(a.table_header) if a.table_header else set()
    gate = lambda path: cw.check_engine(path, a.engine_source, cuobjdump, a.table_header)
    bad = []
    with open(a.program, "rb") as f:
        data = f.read()
    clean = gate(a.program)
    if clean:
        bad.append("the program itself does not pass the gate: " + "; ".join(clean[:5]))
    with tempfile.TemporaryDirectory() as d:
        for what, leaked, expect in leaks(d, a.program, data, kernels, internal):
            path = os.path.join(d, "glyd-engine")
            with open(path, "wb") as f:
                f.write(leaked)
            got = gate(path)
            if any(expect in line for line in got):
                print(f"caught  {what}: {next(line for line in got if expect in line)[:150]}")
            else:
                print(f"MISSED  {what}: the gate said {got[:3]}")
                bad.append(f"{what}: not caught")
        # the gate needs cuobjdump: where it is not on PATH the gate fails, whatever the program
        path = os.path.join(d, "glyd-engine")
        with open(path, "wb") as f:
            f.write(data)
        done = subprocess.run([sys.executable, os.path.join(HERE, "check_wheel.py"), "--engine", path], env={**os.environ, "PATH": d}, capture_output=True, text=True)
        if done.returncode == 1 and "cuobjdump is not on PATH" in done.stdout:
            print("caught  no cuobjdump: the gate fails where it cannot read the machine code")
        else:
            print(f"MISSED  no cuobjdump: exit {done.returncode}, {done.stdout[-200:]!r}")
            bad.append("no cuobjdump: not caught")
    for line in bad:
        print("  " + line)
    print(f"{'FAIL' if bad else 'ok  '} {len(bad)} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
