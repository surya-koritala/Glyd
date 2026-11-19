"""tools/autotune's tables: gpu/route_tables.h is the record's (tables.py --render), the record derives again from its inputs, the
rules (GAIN, WORST, the routes kept) hold on a made-up sweep, and the header's tuned_route, compiled by the C++ compiler where there is
one, answers as tables.lookup does (the shipped header and a made-up one).   python test_tables.py"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import tables as T  # noqa: E402

REC = os.path.join(ROOT, "benchmarks", "gpu", "autotune-2026-10-06")


def test_header():
    rec = json.load(open(os.path.join(REC, "tables.json")))
    assert open(os.path.join(ROOT, "gpu", "route_tables.h")).read() == T.render(rec), "gpu/route_tables.h is not tables.py --render of the record"


def test_record():
    rec = json.load(open(os.path.join(REC, "tables.json")))
    for g in rec["gpus"]:
        points, table = T.decide(json.load(open(os.path.join(REC, f"sweep-{g['code']}.json"))))
        assert table == g["table"] and json.loads(json.dumps(points)) == g["points"], g["name"]


def row(code, layout, M, O, K, route, us, default=False, way=1, variant=0, env=None):
    return {"model": "m", "name": f"{O}x{K}", "O": O, "K": K, "M": M, "layout": layout, "head": False, "way": way, "route": route, "variant": variant,
            "env": env or {}, "default": default, "us": us, "err": 0.0}


def made_up():
    """GeForce Ada (1089), 12-bit, two matrices of one size class: big/1 10% faster on both at 32 and 48 (a row 32-48), at 64 one
    matrix 2% slower (none), at 128 today's route decoding ahead (kept), a development variable's 20% (not shipped); compute
    capability 12 (1120), tiered: big/1 faster at 32 (another change's: kept) and at 128 (a row 128-128)."""
    rows = []
    for O, K in ((4096, 4096), (4096, 5120)):
        for M in (32, 48, 64):
            rows += [row(1089, "mma12", M, O, K, "default", 100.0, True, way=2), row(1089, "mma12", M, O, K, "big", 102.0 if (M, K) == (64, 5120) else 90.0, variant=1),
                     row(1089, "mma12", M, O, K, "gemm", 80.0, env={"GLYD_GPU_PFD": "2"}), row(1089, "mma12", M, O, K, "decode", 300.0)]
        rows += [row(1089, "mma12", 128, O, K, "decode", 100.0, True, way=5), row(1089, "mma12", 128, O, K, "big", 50.0, variant=1)]
    a = {"gpu": "Ada", "code": 1089, "cc": "8.9", "rows": rows, "failed": []}
    rows = []
    for M, way in ((32, 1), (128, 4)):
        rows += [row(1120, "mma", M, 4096, 4096, "default", 100.0, True, way=way), row(1120, "mma", M, 4096, 4096, "big", 50.0, variant=1)]
    return a, {"gpu": "cc12", "code": 1120, "cc": "12.0", "rows": rows, "failed": []}


def test_rules():
    a, b = made_up()
    points, table = T.decide(a)
    assert table == [{"gpu": 1089, "twelve": 1, "wlo": 12_000_000, "whi": 24_000_000, "mlo": 32, "mhi": 48, "route": 4, "variant": 1, "config": "big/1"}], table
    why = {p["M"]: p["why"] for p in points}
    assert why[64].startswith("a matrix over") and "ahead" in why[128], why
    assert next(p for p in points if p["M"] == 32)["best"] == "gemm GLYD_GPU_PFD=2"  # (the fastest, reported; a variable is never shipped)
    points, table = T.decide(b)
    assert [(t["mlo"], t["mhi"], t["twelve"]) for t in table] == [(128, 128, 0)], table
    assert "another change" in next(p for p in points if p["M"] == 32)["why"]


QUERIES = [(g, tw, O, K, M) for g in (1089, 1120, 4089, 7090, 80) for tw in (0, 1) for O, K in ((4096, 4096), (4096, 4100), (4100, 4096), (2048, 512), (248320, 4096), (34816, 5120))
           for M in (0, 1, 16, 17, 31, 32, 48, 49, 64, 65, 127, 128, 129, 512, 2048, 4096)]


def compiled(cxx, header):
    """tuned_route's answers to QUERIES from header, compiled."""
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "route_tables.h"), "w").write(header)
        open(os.path.join(d, "t.cpp"), "w").write('#include "route_tables.h"\n#include <cstdio>\nint main() {\n    long long g, tw, O, K, M;\n'
                                                   '    while (scanf("%lld %lld %lld %lld %lld", &g, &tw, &O, &K, &M) == 5) {\n        int v = 0, r = tuned_route(tw != 0, g, O, K, M, &v);\n'
                                                   '        printf("%d %d\\n", r, r >= 0 ? v : 0);\n    }\n}\n')
        subprocess.run([cxx, "-std=c++17", "-Wall", "-Werror", "-o", os.path.join(d, "t"), os.path.join(d, "t.cpp")], check=True)
        out = subprocess.run([os.path.join(d, "t")], input="".join(" ".join(map(str, q)) + "\n" for q in QUERIES), capture_output=True, text=True, check=True).stdout
    return [tuple(map(int, line.split())) for line in out.splitlines()]


def test_cpp():
    cxx = shutil.which("g++") or shutil.which("c++")
    if not cxx:
        print("  (no C++ compiler here: tuned_route not compiled)")
        return
    rec = json.load(open(os.path.join(REC, "tables.json")))
    made = {"record": "made-up", "gpus": [{"table": [
        {"gpu": 1089, "twelve": 1, "wlo": 12_000_000, "whi": 24_000_000, "mlo": 17, "mhi": 64, "route": 4, "variant": 2, "config": "big/2"},
        {"gpu": 1089, "twelve": 0, "wlo": 0, "whi": 24_000_000, "mlo": 1, "mhi": 128, "route": 1, "variant": 0, "config": "gemm"},
        {"gpu": 1120, "twelve": 1, "wlo": 12_000_000, "whi": 1 << 62, "mlo": 32, "mhi": 512, "route": 2, "variant": 0, "config": "mid"},
        {"gpu": 7090, "twelve": 0, "wlo": 0, "whi": 1 << 62, "mlo": 129, "mhi": 4096, "route": 0, "variant": 0, "config": "decode"},
        {"gpu": 4089, "twelve": 0, "wlo": 0, "whi": 1 << 62, "mlo": 1, "mhi": 64, "route": 2, "variant": 0, "config": "mid (refused: tiered)"}]}]}
    for r in (rec, made):
        table = [t for g in r["gpus"] for t in g["table"]]
        want = [T.lookup(table, *q) for q in QUERIES]
        got = compiled(cxx, T.render(r))
        assert got == [(rt, v if rt >= 0 else 0) for rt, v in want], r["record"]
    assert any(rt >= 0 for rt, _ in (T.lookup(made["gpus"][0]["table"], *q) for q in QUERIES))
    assert all(T.lookup(made["gpus"][0]["table"], *q)[0] < 0 for q in QUERIES if q[0] == 4089)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(f"{name}: ok")
