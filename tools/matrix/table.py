#!/usr/bin/env python3
"""The release matrix's README from the pods' logs (matrix_job.sh's files, one directory a GPU class under DIR):  table.py DIR VERSION > DIR/README.md
Rows: GPU class; columns: model:level; a cell: tokens a second at a 4k and a 32k context, or `does not fit: <the engine's message>` (verbatim, the last line it printed).
table.py --selftest checks it."""
import re, sys, tempfile
from pathlib import Path


def cell(d, row):
    m, l, ctx, rc, secs, n = row
    out = (d / f"c{n}.out").read_text(errors="replace") if (d / f"c{n}.out").exists() else ""
    err = (d / f"c{n}.err").read_text(errors="replace") if (d / f"c{n}.err").exists() else ""
    text = [x.strip() for x in (err + "\n" + out).splitlines() if x.strip()]
    if rc == "0":
        tps = re.findall(r"([\d.]+) tokens/s", err + out)
        load = re.findall(r"Ready in ([^:]+):", err + out)
        first = next((x.strip() for x in out.splitlines() if x.strip()), "")
        return {"ok": True, "tps": tps[-1] if tps else None, "load": load[0] if load else None, "first": first, "secs": secs}
    return {"ok": False, "why": text[-1] if text else f"exit {rc}, no output", "rc": rc, "secs": secs}


def show(c):
    if c["ok"]:
        return (f"{c['tps']} tok/s" if c["tps"] else "fits (no speed printed)")
    return "does not fit: " + c["why"].replace("|", "/")


def build(root, version):
    classes = sorted(p for p in Path(root).iterdir() if (p / "cells.tsv").exists())
    cols, data, gpus = [], {}, {}
    for d in classes:
        gpus[d.name] = (d / "gpu.txt").read_text().split(",")[0].strip() if (d / "gpu.txt").exists() else "?"
        for line in (d / "cells.tsv").read_text().splitlines():
            row = line.split("\t")
            if len(row) != 6:
                continue
            key = f"{row[0]}:{row[1]}"
            if key not in cols:
                cols.append(key)
            data.setdefault((d.name, key), {})[row[2]] = cell(d, row)
    out = [f"# Release matrix, Glyd {version}", "",
           "`glyd run MODEL:LEVEL --no-think --context 4096|32768` on a fresh install of the release, one pod a GPU class. Each cell: tokens a second at a 4k context, then at 32k, or the engine's own refusal.", "",
           "| GPU | " + " | ".join(cols) + " |", "|" + " --- |" * (len(cols) + 1)]
    for d in classes:
        cells = []
        for k in cols:
            c = data.get((d.name, k), {})
            if not c:
                cells.append("not run")
                continue
            a, b = c.get("4096"), c.get("32768")
            sa, sb = show(a) if a else "not run", show(b) if b else "not run"
            cells.append(sa if sa == sb else f"4k: {sa}<br>32k: {sb}")
        out.append(f"| {d.name} ({gpus[d.name]}) | " + " | ".join(cells) + " |")
    out += ["", "## Load time and first line of the answer", ""]
    for d in classes:
        for k in cols:
            for ctx, c in sorted(data.get((d.name, k), {}).items()):
                if c["ok"]:
                    out.append(f"- {d.name} {k} {ctx}: load {c['load'] or '?'}, {c['secs']} s in all; \"{c['first']}\"")
    out += ["", "Raw logs: `<class>/c<N>.out`, `.err` (N is the line of `cells.tsv`), `install.log`, `steps.txt`."]
    return "\n".join(out) + "\n"


def selftest():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t) / "4090"; d.mkdir()
        (d / "gpu.txt").write_text("NVIDIA GeForce RTX 4090, 580.1, 24564 MiB\n")
        (d / "cells.tsv").write_text("Q/m\tlinnet\t4096\t0\t50\t1\nQ/m\tlinnet\t32768\t3\t4\t2\nQ/m\tpenguin\t4096\t0\t9\t3\n")
        (d / "c1.err").write_text("Ready in 31 s: m on the GPU\n(120 tokens, 41.2 tokens/s)\n"); (d / "c1.out").write_text("\nParis is the capital.\nMore.\n")
        (d / "c2.err").write_text("Loading\nm does not fit this GPU: it needs about 30 GiB.\n")
        (d / "c3.out").write_text("hi\n")
        md = build(t, "0.30.0")
        assert "| 4090 (NVIDIA GeForce RTX 4090) | 4k: 41.2 tok/s<br>32k: does not fit: m does not fit this GPU: it needs about 30 GiB. | 4k: fits (no speed printed)<br>32k: not run |" in md, md
        assert "load 31 s" in md and '"Paris is the capital."' in md and "# Release matrix, Glyd 0.30.0" in md
    print("ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        print(build(sys.argv[1], sys.argv[2]), end="")
