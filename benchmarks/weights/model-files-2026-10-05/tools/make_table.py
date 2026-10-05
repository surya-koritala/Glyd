"""The results table (markdown) from the harness's TSVs: baseline.tsv (zstd, xz, glyd v0.28.0), after.tsv (this change), header_sizes.tsv."""
import collections, sys
base, after, hdr = {}, {}, {}
for line in open(sys.argv[1]):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 9 and not p[3].startswith("FAIL"):
        base[(p[0], p[1])] = dict(size=int(p[2]), comp=int(p[3]), cw=float(p[4]), cc=float(p[5]), dw=float(p[6]), dc=float(p[7]))
for line in open(sys.argv[2]):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 9 and p[8] == "ok":
        after[p[0]] = dict(size=int(p[2]), comp=int(p[3]), cw=float(p[4]), cc=float(p[5]), dw=float(p[6]), dc=float(p[7]))
for line in open(sys.argv[3]):
    p = line.rstrip("\n").split("\t")
    hdr[p[0]] = tuple(int(x) for x in p[1:5])
names = sorted(after, key=lambda n: (n.split("-")[0] != "st", n))
sv = lambda size, comp: 100.0 * (1 - comp / size)
mb = lambda size, t: size / 1e6 / t
out = []
out.append("| file | size MB | zstd -19 | zstd -19 --long=27 | xz -9 | glyd --max, v0.28.0 | **glyd --max, now** | vs zstd -19 | vs xz -9 | now: compress MB/s | now: decompress MB/s | v0.28.0: compress MB/s | v0.28.0: decompress MB/s |")
out.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
worst = []
for n in names:
    a = after[n]; z = base[(n, "zstd19")]; zl = base[(n, "zstd19long")]; x = base[(n, "xz9")]; g = base[(n, "glyd-max")]
    size = a["size"]
    sz, szl, sx, sg, sa = sv(size, z["comp"]), sv(size, zl["comp"]), sv(size, x["comp"]), sv(size, g["comp"]), sv(size, a["comp"])
    out.append(f"| {n} | {size/1e6:.1f} | {sz:.2f}% | {szl:.2f}% | {sx:.2f}% | {sg:.2f}% | **{sa:.2f}%** | {sa-sz:+.2f} | {sa-sx:+.2f} | {mb(size, a['cc']):.0f} | {mb(size, a['dc']):.0f} | {mb(size, g['cc']):.0f} | {mb(size, g['dc']):.0f} |")
print("\n".join(out))
print()
# header-excluded
print("| GGUF file | header MB | header: zstd -19 / glyd v0.28.0 / xz -9 (MB) | payload only: zstd -19 | payload only: glyd --max now | difference |")
print("| :--- | ---: | :--- | ---: | ---: | ---: |")
for n in names:
    if n not in hdr: continue
    h, hz, hg, hx = hdr[n]
    a = after[n]; z = base[(n, "zstd19")]
    size = a["size"]
    pz = 100.0 * (1 - (z["comp"] - hz) / (size - h))
    pa = 100.0 * (1 - (a["comp"] - hg) / (size - h))
    print(f"| {n} | {h/1e6:.1f} | {hz/1e6:.2f} / {hg/1e6:.2f} / {hx/1e6:.2f} | {pz:.2f}% | {pa:.2f}% | {pa-pz:+.2f} |")
print()
# summary
d = [after[n]["size"] for n in names]
tot = sum(d)
cc = sum(after[n]["cc"] for n in names); dc = sum(after[n]["dc"] for n in names)
print("files", len(names), "total GB", tot/1e9, "aggregate compress MB/s (CPU)", tot/1e6/cc, "decompress", tot/1e6/dc)
mins = min((mb(after[n]["size"], after[n]["cc"]), n) for n in names); maxs = max((mb(after[n]["size"], after[n]["cc"]), n) for n in names)
print("compress MB/s min/max", mins, maxs)
mind = min((mb(after[n]["size"], after[n]["dc"]), n) for n in names); maxd = max((mb(after[n]["size"], after[n]["dc"]), n) for n in names)
print("decompress MB/s min/max", mind, maxd)
bz = sum(base[(n, "zstd19")]["comp"] for n in names); ba = sum(after[n]["comp"] for n in names); bx = sum(base[(n, "xz9")]["comp"] for n in names); bg = sum(base[(n, "glyd-max")]["comp"] for n in names)
print("whole corpus saved: zstd19 %.2f%% xz9 %.2f%% glyd-before %.2f%% glyd-now %.2f%%" % (100*(1-bz/tot), 100*(1-bx/tot), 100*(1-bg/tot), 100*(1-ba/tot)))
