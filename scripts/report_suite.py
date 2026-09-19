#!/usr/bin/env python3
"""Markdown tables from the benchmark suite's JSON lines.

    scripts/report_suite.py benchmarks/suite/<machine>/ [...]

Reads bench_suite_*.jsonl and s3_workflow.jsonl in each directory and
prints, per machine: the large-file totals per codec and thread count,
ratios per data kind, the small-object sets, and the S3 workflow with
its monthly cost. Every number is what the harnesses wrote; nothing is
recomputed except sums and ratios of sums.
"""
import glob
import json
import os
import sys

KIND = {"json": "JSON events", "log": "logs", "sql": "SQL dumps", "parquet": "Parquet"}


def rows(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def fmt_bytes(n):
    return f"{n / 1e9:.3f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def large_tables(d):
    large = []
    machine = None
    for path in sorted(glob.glob(os.path.join(d, "bench_suite_*.jsonl"))):
        for r in rows(path):
            if r["kind"] == "machine" and machine is None:
                machine = r
            if r["kind"] == "large":
                large.append(r)
    if not large:
        return
    threads = sorted({r["threads"] for r in large})
    codecs = []
    for r in large:
        if r["codec"] not in codecs:
            codecs.append(r["codec"])
    for t in threads:
        sel = [r for r in large if r["threads"] == t]
        files = sorted({r["file"] for r in sel})
        # Only files every codec covered, so the totals compare like with like.
        complete = [f for f in files if all(any(r["file"] == f and r["codec"] == c for r in sel) for c in codecs)]
        raw = sum(next(r["bytes"] for r in sel if r["file"] == f and r["codec"] == codecs[0]) for f in complete)
        print(f"\n### Large files, {t} thread{'s' if t > 1 else ''} ({len(complete)} files, {fmt_bytes(raw)} in)\n")
        print("| Codec | Ratio | Compressed | Compress MB/s | Decompress MB/s | Peak RSS (64 MB input) |")
        print("| :--- | ---: | ---: | ---: | ---: | ---: |")
        for c in codecs:
            rs = [r for r in sel if r["codec"] == c and r["file"] in complete]
            comp = sum(r["compressed"] for r in rs)
            cs = sum(r["compress_s"] for r in rs)
            ds = sum(r["decompress_s"] for r in rs)
            rss = max(r.get("peak_rss_bytes_64mb", r.get("peak_rss_bytes", 0)) for r in rs)
            bold = c.startswith("glyd")
            name = f"**{c}**" if bold else c
            print(f"| {name} | {raw / comp:.3f} | {fmt_bytes(comp)} | {raw / cs / 1e6:,.0f} | {raw / ds / 1e6:,.0f} | {rss / 2**20:,.0f} MB |")
        # Per kind.
        print(f"\nRatio by data kind ({t} thread{'s' if t > 1 else ''}):\n")
        kinds = []
        for f in complete:
            k = KIND.get(f.rsplit(".", 1)[-1], "other")
            if k not in kinds:
                kinds.append(k)
        print("| Codec | " + " | ".join(kinds) + " |")
        print("| :--- | " + " | ".join("---:" for _ in kinds) + " |")
        for c in codecs:
            cells = []
            for k in kinds:
                rs = [r for r in sel if r["codec"] == c and r["file"] in complete and KIND.get(r["file"].rsplit(".", 1)[-1], "other") == k]
                cells.append(f"{sum(r['bytes'] for r in rs) / sum(r['compressed'] for r in rs):.3f}")
            name = f"**{c}**" if c.startswith("glyd") else c
            print(f"| {name} | " + " | ".join(cells) + " |")
        print(f"\nPer file ({t} thread{'s' if t > 1 else ''}), ratio · compress MB/s · decompress MB/s:\n")
        print("| File | " + " | ".join(codecs) + " |")
        print("| :--- | " + " | ".join("---:" for _ in codecs) + " |")
        for f in complete:
            cells = []
            for c in codecs:
                r = next(r for r in sel if r["file"] == f and r["codec"] == c)
                cells.append(f"{r['ratio']:.3f} · {r['compress_mb_s']:,.0f} · {r['decompress_mb_s']:,.0f}")
            print(f"| {f} | " + " | ".join(cells) + " |")


def small_tables(d):
    small = []
    for path in sorted(glob.glob(os.path.join(d, "bench_suite_*.jsonl"))):
        small += [r for r in rows(path) if r["kind"] == "small"]
    if not small:
        return
    print("\n### Small objects (one thread, dictionaries of 110 KB trained on other days' data)\n")
    sets = []
    for r in small:
        if r["set"] not in sets:
            sets.append(r["set"])
    for s in sets:
        rs = [r for r in small if r["set"] == s]
        r0 = rs[0]
        print(f"\n{s}: {r0['objects']:,} objects, {fmt_bytes(r0['bytes'])}, mean {r0['bytes'] // r0['objects']:,} B\n")
        print("| Codec | Ratio | Compressed | + dictionary | Compress µs p50 / p90 / p99 | Decompress µs p50 / p90 / p99 |")
        print("| :--- | ---: | ---: | ---: | ---: | ---: |")
        for r in rs:
            name = f"**{r['codec']}**" if r["codec"].startswith("glyd") else r["codec"]
            print(f"| {name} | {r['ratio']:.3f} | {fmt_bytes(r['compressed'])} | {fmt_bytes(r['compressed'] + r['dict_bytes'])} | {r['compress_us_p50']:.2f} / {r['compress_us_p90']:.2f} / {r['compress_us_p99']:.2f} | {r['decompress_us_p50']:.2f} / {r['decompress_us_p90']:.2f} / {r['decompress_us_p99']:.2f} |")


def s3_table(d):
    path = os.path.join(d, "s3_workflow.jsonl")
    if not os.path.exists(path):
        return
    rs = list(rows(path))
    if not rs:
        return
    r0 = rs[0]
    p = r0["prices"]
    print(f"\n### S3 workflow ({r0['instance']}, {r0['threads']} threads; {r0['files']} files, {fmt_bytes(r0['raw_bytes'])})\n")
    print(f"Prices (us-east-1, on-demand): S3 Standard ${p['s3_gb_month']}/GB-month, PUT ${p['put_per_1000']}/1,000, GET ${p['get_per_1000']}/1,000, instance ${p['ec2_per_hour']}/hour; EC2-S3 transfer in the region is free, internet egress ${p['egress_per_gb']}/GB.\n")
    print("| Codec | Stored | Ratio | Compress s (wall / CPU) | Upload s | Download s | Decompress s (wall / CPU) | Verified | $/month, 1 read | 10 reads | 100 reads | Egress $/read |")
    print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: |")
    for r in rs:
        name = f"**{r['codec']}**" if r["codec"].startswith("glyd") else r["codec"]
        print(f"| {name} | {fmt_bytes(r['stored_bytes'])} | {r['ratio']:.3f} | {r['compress_wall_s']:.1f} / {r['compress_cpu_s']:.1f} | {r['upload_wall_s']:.1f} | {r['download_wall_s']:.1f} | {r['decompress_wall_s']:.1f} / {r['decompress_cpu_s']:.1f} | {r['verified']} | {r['usd_month_1_read']:.4f} | {r['usd_month_10_reads']:.4f} | {r['usd_month_100_reads']:.4f} | {r['usd_egress_per_read']:.4f} |")


def main():
    for d in sys.argv[1:]:
        machine = None
        for path in sorted(glob.glob(os.path.join(d, "bench_suite_*.jsonl"))):
            for r in rows(path):
                if r["kind"] == "machine":
                    machine = r
                    break
            if machine:
                break
        title = os.path.basename(os.path.normpath(d))
        print(f"\n## {title}" + (f" — {machine['cpu']}, {machine['cores']} cores, {machine['os']}/{machine['arch']}, {machine['date'][:10]}" if machine else ""))
        large_tables(d)
        small_tables(d)
        s3_table(d)


if __name__ == "__main__":
    main()
