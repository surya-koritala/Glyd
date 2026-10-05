"""Compress and decompress every corpus file with a set of codecs, one thread each; ratio and MB/s each way.

    python bench.py OUT.tsv JOBS TOOLS FILE...

TOOLS: comma list of names from TOOLSET. A TSV line per (file, tool): file, tool, size, comp_bytes, comp_wall_s,
comp_cpu_s, decomp_wall_s, decomp_cpu_s, roundtrip ('ok' for glyd/zstd/xz/brotli: cmp against the input).
"""
import os, subprocess, sys, time, threading, concurrent.futures as cf, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GLYD = os.environ.get("GLYD", "glyd")  # the glyd under test

# name -> (compress argv with {i} {o}, decompress argv with {o}) ; None as the output path means stdout is the output
TOOLSET = {
    "zstd19": (["zstd", "-19", "-T1", "-q", "-f", "{i}", "-o", "{o}"], ["zstd", "-d", "-q", "-c", "{o}"]),
    "zstd19long": (["zstd", "-19", "--long=27", "-T1", "-q", "-f", "{i}", "-o", "{o}"], ["zstd", "-d", "--long=27", "-q", "-c", "{o}"]),
    "xz9": (["xz", "-9", "-T1", "-f", "-k", "-c", "{i}"], ["xz", "-d", "-c", "{o}"]),
    "brotli11": (["brotli", "-q", "11", "-w", "24", "-f", "-c", "{i}"], ["brotli", "-d", "-c", "{o}"]),
    "glyd": ([GLYD, "-s", "{i}", "-o", "{o}"], [GLYD, "-d", "-s", "{o}", "-o", "/dev/null"]),
    "glyd-fast": ([GLYD, "--fast", "-s", "{i}", "-o", "{o}"], [GLYD, "-d", "-s", "{o}", "-o", "/dev/null"]),
    "glyd-turbo": ([GLYD, "--turbo", "-s", "{i}", "-o", "{o}"], [GLYD, "-d", "-s", "{o}", "-o", "/dev/null"]),
    "glyd-max": ([GLYD, "--max", "-s", "{i}", "-o", "{o}"], [GLYD, "-d", "-s", "{o}", "-o", "/dev/null"]),
    "glyd-ultra": ([GLYD, "--ultra", "-s", "{i}", "-o", "{o}"], [GLYD, "-d", "-s", "{o}", "-o", "/dev/null"]),
    "glyd-cold": ([GLYD, "--cold", "-s", "{i}", "-o", "{o}"], [GLYD, "-d", "-s", "{o}", "-o", "/dev/null"]),
}
STDOUT_OUT = {"xz9", "brotli11"}  # the compressor writes to stdout


def run(argv, stdout=None):
    t0 = time.perf_counter()
    p = subprocess.Popen(argv, stdout=stdout, stderr=subprocess.PIPE)
    _, status, ru = os.wait4(p.pid, 0)
    wall = time.perf_counter() - t0
    err = p.stderr.read()
    p.stderr.close()
    p.returncode = os.waitstatus_to_exitcode(status)
    return p.returncode, wall, ru.ru_utime + ru.ru_stime, err


def one(path, tool, tmp, lock, out_tsv):
    comp, decomp = TOOLSET[tool]
    out = os.path.join(tmp, os.path.basename(path) + "." + tool)
    size = os.path.getsize(path)
    ci = [a.replace("{i}", path).replace("{o}", out) for a in comp]
    if tool in STDOUT_OUT:
        with open(out, "wb") as f:
            rc, cw, cc, err = run(ci, stdout=f)
    else:
        rc, cw, cc, err = run(ci)
    if rc != 0:
        line = f"{os.path.basename(path)}\t{tool}\t{size}\tFAIL compress {rc} {err[:80]!r}"
    else:
        csize = os.path.getsize(out)
        di = [a.replace("{o}", out) for a in decomp]
        rc2, dw, dc, err2 = run(di, stdout=subprocess.DEVNULL)
        ok = "ok" if rc2 == 0 else f"FAIL decompress {rc2}"
        if rc2 == 0 and tool.startswith("glyd") or tool in ("zstd19", "zstd19long", "xz9", "brotli11"):
            # byte-for-byte: decode to a temp file and cmp
            back = out + ".back"
            if tool.startswith("glyd"):
                rc3, _, _, _ = run([GLYD, "-d", "-s", out, "-o", back])
            else:
                with open(back, "wb") as f:
                    rc3, _, _, _ = run(di, stdout=f)
            same = rc3 == 0 and subprocess.run(["cmp", "-s", back, path]).returncode == 0
            ok = "ok" if same and rc2 == 0 else "MISMATCH"
            try:
                os.unlink(back)
            except OSError:
                pass
        line = f"{os.path.basename(path)}\t{tool}\t{size}\t{csize}\t{cw:.3f}\t{cc:.3f}\t{dw:.3f}\t{dc:.3f}\t{ok}"
    try:
        os.unlink(out)
    except OSError:
        pass
    with lock:
        with open(out_tsv, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)


def main():
    out_tsv, jobs, tools = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
    files = sys.argv[4:]
    tmp = tempfile.mkdtemp(prefix="codecw_bench_", dir=os.environ.get("TMPDIR", "/private/tmp"))
    lock = threading.Lock()
    work = [(f, t) for f in files for t in tools]
    with cf.ThreadPoolExecutor(jobs) as ex:
        list(ex.map(lambda ft: one(ft[0], ft[1], tmp, lock, out_tsv), work))


main()
