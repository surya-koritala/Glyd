#!/usr/bin/env python3
"""Every codec on this machine against Glyd, on every kind of data:
bytes out, and compress and decompress speed on one thread (MB/s of
the original), each decode compared with its input. Text-like inputs
are the first 32 MB of a file (the long-range levels see less than on
a whole file); containers, pictures and small files are whole.

    scripts/landscape.py [out.jsonl]      # rows as they are measured
    scripts/landscape.py --report out.jsonl [more.jsonl ...] > table.md

Codecs are the CLIs on PATH (zstd, xz, brotli, gzip, bzip2, lz4, zpaq,
cjxl/djxl for JPEG) at one thread; Glyd is ./target/release/glyd with
-s (one core) unless GLYD is set.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLYD = os.environ.get("GLYD", os.path.join(ROOT, "target/release/glyd"))
SCRATCH = os.environ.get("LANDSCAPE_DATA", "")
SAMPLE = 32 << 20

# (label, kind, path, whole file?)
INPUTS = [
    ("Web server log (NASA)", "logs", "corpus/bench/nasa-access-jul95.log", False),
    ("Pageview log (Wikimedia)", "logs", "corpus/bench/pageviews-20240115-10.log", False),
    ("JSON events (GitHub Archive)", "json", "corpus/bench/gharchive-2024-01-15-12.json", False),
    ("SQL dump (Wikipedia page_props)", "sql", "corpus/bench/enwiki-page_props.sql", False),
    ("SQL dump (Simple Wikipedia categorylinks)", "sql", "corpus/bench/simplewiki-categorylinks.sql", False),
    ("Wikipedia XML text (enwik8)", "text", "corpus/enwik8", False),
    ("English text (Silesia dickens)", "text", "corpus/dickens", True),
    ("Source tree tar (Linux 6.10)", "source", "corpus/bucket/linux-6.10.1.tar", False),
    ("OS image tar (Ubuntu root)", "binary", "corpus/bucket/noble-20260705-root.tar", False),
    ("Executable (Silesia mozilla)", "binary", "corpus/mozilla", False),
    ("Database (Silesia osdb)", "binary", "corpus/osdb", True),
    ("Medical image (Silesia x-ray)", "binary", "corpus/x-ray", True),
    ("Parquet (NYC taxi, zstd inside)", "parquet", "corpus/bench/yellow_tripdata_2024-02.parquet", False),
]
if SCRATCH:
    for label, kind, name in [
        ("gzipped log (NASA, gzip -6)", "container", "nasa.gz"),
        ("Jar (Guava, 2,059 entries)", "container", "defl/guava.jar"),
        ("Source zip (GitHub)", "container", "defl/zstd-src.zip"),
        ("PowerPoint, text (.pptx)", "container", "defl/deck.pptx"),
        ("Word, screenshots (.docx)", "container", "defl/screenshots.docx"),
        ("Excel sheet (.xlsx)", "container", "defl/metrics.xlsx"),
        ("PDF paper (pdfTeX)", "container", "defl/attention.pdf"),
        ("PDF paper with figures", "container", "defl/gpt3.pdf"),
        ("PNG photo", "image", "defl/img1.png"),
        ("JPEG photo", "image", "re/img3.jpg"),
        ("tar.gz of mixed objects", "container", "defl/mixed.tar.gz"),
    ]:
        INPUTS.append((label, kind, os.path.join(SCRATCH, name), True))

# (name, compress argv or callable, decompress argv, applies to kinds or None)
def cli(c, d):
    return (c, d)

CODECS = [
    ("lz4 -1", ["lz4", "-1", "-q", "-c"], ["lz4", "-d", "-q", "-c"], None),
    ("lz4 -9", ["lz4", "-9", "-q", "-c"], ["lz4", "-d", "-q", "-c"], None),
    ("gzip -6", ["gzip", "-6", "-c"], ["gzip", "-d", "-c"], None),
    ("gzip -9", ["gzip", "-9", "-c"], ["gzip", "-d", "-c"], None),
    ("bzip2 -9", ["bzip2", "-9", "-c"], ["bzip2", "-d", "-c"], None),
    ("zstd -1", ["zstd", "-1", "-T1", "-q", "-c"], ["zstd", "-d", "-q", "-c"], None),
    ("zstd -3", ["zstd", "-3", "-T1", "-q", "-c"], ["zstd", "-d", "-q", "-c"], None),
    ("zstd -9", ["zstd", "-9", "-T1", "-q", "-c"], ["zstd", "-d", "-q", "-c"], None),
    ("zstd -19", ["zstd", "-19", "-T1", "-q", "-c"], ["zstd", "-d", "-q", "-c"], None),
    ("zstd -22 --long", ["zstd", "--ultra", "-22", "--long=27", "-T1", "-q", "-c"], ["zstd", "-d", "--long=27", "-q", "-c"], None),
    ("brotli -5", ["brotli", "-q", "5", "-c"], ["brotli", "-d", "-c"], None),
    ("brotli -11", ["brotli", "-q", "11", "-w", "24", "-c"], ["brotli", "-d", "-c"], None),
    ("xz -6", ["xz", "-6", "-T1", "-c"], ["xz", "-d", "-c"], None),
    ("xz -9e", ["xz", "-9e", "-T1", "-c"], ["xz", "-d", "-c"], None),
    ("zpaq -m5", "zpaq", None, None),
    ("JPEG XL (lossless JPEG)", "cjxl", None, {"image"}),
    ("Glyd default", ["-s"], None, None),
    ("Glyd --max", ["-s", "--max"], None, None),
    ("Glyd --max -r", ["-s", "--max", "-r"], None, None),
    ("Glyd --ultra", ["-s", "--ultra"], None, None),
    ("Glyd --ultra -r", ["-s", "--ultra", "-r"], None, None),
    ("Glyd --cold -r", ["-s", "--cold", "-r"], None, None),
]


def timed(argv, stdin_path, stdout_path):
    with open(stdin_path, "rb") as i, open(stdout_path, "wb") as o:
        t = time.perf_counter()
        r = subprocess.run(argv, stdin=i, stdout=o, stderr=subprocess.DEVNULL)
        return time.perf_counter() - t, r.returncode


def run_one(codec, src, work):
    name, c, d, kinds = codec
    comp = os.path.join(work, "c")
    back = os.path.join(work, "b")
    for p in (comp, back):
        if os.path.exists(p):
            os.remove(p)
    if isinstance(c, list) and name.startswith("Glyd"):
        t = time.perf_counter()
        r = subprocess.run([GLYD, *c, src, "-o", comp], stderr=subprocess.DEVNULL)
        ct = time.perf_counter() - t
        t = time.perf_counter()
        r2 = subprocess.run([GLYD, "-d", "-s", comp, "-o", back], stderr=subprocess.DEVNULL)
        dt = time.perf_counter() - t
        ok = r.returncode == 0 and r2.returncode == 0
        size = os.path.getsize(comp) if os.path.exists(comp) else 0
    elif c == "zpaq":
        arc = os.path.join(work, "a.zpaq")
        out = os.path.join(work, "x")
        for p in (arc,):
            if os.path.exists(p):
                os.remove(p)
        shutil.rmtree(out, ignore_errors=True)
        t = time.perf_counter()
        r = subprocess.run(["zpaq", "a", arc, os.path.basename(src), "-m5", "-t1"], cwd=os.path.dirname(src), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ct = time.perf_counter() - t
        t = time.perf_counter()
        r2 = subprocess.run(["zpaq", "x", arc, "-to", out, "-t1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dt = time.perf_counter() - t
        size = os.path.getsize(arc) if os.path.exists(arc) else 0
        extracted = os.path.join(out, os.path.basename(src))
        ok = r.returncode == 0 and r2.returncode == 0 and os.path.exists(extracted)
        if ok:
            shutil.move(extracted, back)
        shutil.rmtree(out, ignore_errors=True)
        if os.path.exists(arc):
            os.remove(arc)
    elif c == "cjxl":
        jxl = os.path.join(work, "c.jxl")
        if not src.endswith((".jpg", ".jpeg")):
            return None
        t = time.perf_counter()
        r = subprocess.run(["cjxl", "--lossless_jpeg=1", "-e", "7", "--num_threads=0", src, jxl], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ct = time.perf_counter() - t
        t = time.perf_counter()
        r2 = subprocess.run(["djxl", "--num_threads=0", jxl, back + ".jpg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dt = time.perf_counter() - t
        if os.path.exists(back + ".jpg"):
            shutil.move(back + ".jpg", back)
        size = os.path.getsize(jxl) if os.path.exists(jxl) else 0
        ok = r.returncode == 0 and r2.returncode == 0
        if os.path.exists(jxl):
            os.remove(jxl)
    else:
        ct, rc = timed(c, src, comp)
        dt, rc2 = timed(d, comp, back)
        size = os.path.getsize(comp)
        ok = rc == 0 and rc2 == 0
    if ok:
        with open(src, "rb") as a, open(back, "rb") as b:
            ok = a.read() == b.read()
    n = os.path.getsize(src)
    return {"codec": name, "in": n, "out": size, "ratio": n / size if size else 0, "c_mbps": n / ct / 1e6, "d_mbps": n / dt / 1e6, "exact": ok}


def measure(out_path):
    work = tempfile.mkdtemp(prefix="landscape-")
    # LANDSCAPE_ONLY="2,5,9": these inputs (by position) only, so several
    # lanes can run at once, each codec still on one thread.
    only = os.environ.get("LANDSCAPE_ONLY")
    picked = [INPUTS[int(i)] for i in only.split(",")] if only else INPUTS
    with open(out_path, "a") as out:
        for label, kind, path, whole in picked:
            path = path if os.path.isabs(path) else os.path.join(ROOT, path)
            if not os.path.exists(path):
                print(f"skipping {label}: {path} missing", file=sys.stderr)
                continue
            src = path
            if not whole and os.path.getsize(path) > SAMPLE:
                src = os.path.join(work, os.path.basename(path))
                with open(path, "rb") as f, open(src, "wb") as g:
                    g.write(f.read(SAMPLE))
            codecs = [c for c in CODECS if not os.environ.get("LANDSCAPE_CODECS") or any(c[0].startswith(p) for p in os.environ["LANDSCAPE_CODECS"].split(","))]
            for codec in codecs:
                kinds = codec[3]
                if kinds is not None and kind not in kinds:
                    continue
                row = run_one(codec, src, work)
                if row is None:
                    continue
                row.update({"label": label, "kind": kind})
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(f"{label:44} {row['codec']:26} {row['ratio']:8.3f} {row['c_mbps']:9.1f} {row['d_mbps']:9.1f} {'exact' if row['exact'] else 'MISMATCH'}", file=sys.stderr)
            if src != path:
                os.remove(src)
    shutil.rmtree(work, ignore_errors=True)


FAST = ["lz4 -1", "zstd -1", "zstd -3"]
STRONG = ["zstd -19", "zstd -22 --long", "xz -6", "xz -9e", "brotli -11", "bzip2 -9"]
ARCHIVAL = ["zpaq -m5"]


def report(paths):
    rows = []
    for path in paths:
        rows += [json.loads(l) for l in open(path) if l.strip()]
    labels = []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])
    # A later row for the same data and codec (a re-run) replaces an earlier one.
    latest = {}
    for r in rows:
        latest[(r["label"], r["codec"])] = r
    rows = list(latest.values())
    by = {(r["label"], r["codec"]): r for r in rows if r["exact"]}
    bad = [r for r in rows if not r["exact"]]
    pct = lambda g, o: f"{(o['ratio'] / g['ratio'] - 1) * 100:+.0f}%".replace("+-", "-")
    def best(label, names):
        rs = [by[(label, n)] for n in names if (label, n) in by]
        return max(rs, key=lambda r: r["ratio"]) if rs else None
    glyd_all = ["Glyd default", "Glyd --max", "Glyd --max -r", "Glyd --ultra", "Glyd --ultra -r", "Glyd --cold -r"]
    others = [c[0] for c in CODECS if not c[0].startswith("Glyd")]
    print("Glyd's bytes against the other (minus: fewer bytes than it), and speeds on one thread, MB/s of the original.")
    print()
    print("| Data | Fast: Glyd --max vs zstd -3 (bytes; in; out) | Record mode: Glyd --max -r vs zstd -3 (bytes; in; out) | Strong: best Glyd vs best of zstd -19/-22, xz, brotli -11, bzip2 | Archival: Glyd --cold -r vs zpaq -m5 (bytes; in; out) | Smallest of all |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for label in labels:
        m, z3 = by.get((label, "Glyd --max")), by.get((label, "zstd -3"))
        mr = by.get((label, "Glyd --max -r"))
        cell1 = f"{pct(m, z3)}; {m['c_mbps']/z3['c_mbps']:.2f}×; {m['d_mbps']/z3['d_mbps']:.2f}×" if m and z3 else "–"
        cell1b = f"{pct(mr, z3)}; {mr['c_mbps']/z3['c_mbps']:.2f}×; {mr['d_mbps']/z3['d_mbps']:.2f}×" if mr and z3 and m and mr["ratio"] > m["ratio"] * 1.02 else "–"
        gs, os_ = best(label, ["Glyd --ultra", "Glyd --ultra -r", "Glyd --max -r"]), best(label, STRONG)
        cell2 = f"{pct(gs, os_)} ({gs['codec'][5:]} {gs['ratio']:.2f} vs {os_['codec']} {os_['ratio']:.2f})" if gs and os_ else "–"
        gc, zp = by.get((label, "Glyd --cold -r")), by.get((label, "zpaq -m5"))
        cell3 = f"{pct(gc, zp)}; {gc['c_mbps']/zp['c_mbps']:.0f}×; {gc['d_mbps']/zp['d_mbps']:.0f}×" if gc and zp else "–"
        top = max((r for r in rows if r["label"] == label and r["exact"]), key=lambda r: r["ratio"])
        cell4 = f"{top['codec']} {top['ratio']:.2f}"
        print(f"| {label} | {cell1} | {cell1b} | {cell2} | {cell3} | {cell4} |")
    print()
    for label in labels:
        print(f"<details><summary>{label}</summary>")
        print()
        print("| Codec | Ratio | Compress MB/s | Decompress MB/s |")
        print("| :--- | ---: | ---: | ---: |")
        for r in sorted((r for r in rows if r["label"] == label and r["exact"]), key=lambda r: -r["ratio"]):
            name = f"**{r['codec']}**" if r["codec"].startswith("Glyd") else r["codec"]
            print(f"| {name} | {r['ratio']:.3f} | {r['c_mbps']:.1f} | {r['d_mbps']:.1f} |")
        print()
        print("</details>")
        print()
    if bad:
        print("Decodes that did not match: " + ", ".join(f"{r['label']} / {r['codec']}" for r in bad))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--report":
        report(sys.argv[2:])
    else:
        measure(sys.argv[1] if len(sys.argv) > 1 else "landscape.jsonl")
