#!/usr/bin/env python3
"""Charts and the README table from a landscape run (scripts/landscape.py),
Glyd against the codecs in general use: gzip, zstd, xz, brotli, lz4.

    scripts/landscape_charts.py rows.jsonl out_dir     # three SVGs
    scripts/landscape_charts.py rows.jsonl --table     # the markdown table

Plain SVG, no dependencies; every number is read from the rows.
"""
import json
import math
import os
import sys

# The data types in the order shown, grouped.
GROUPS = [
    ("Records (record mode, -r)", [
        ("Web server log (NASA)", "Web server log"),
        ("Pageview log (Wikimedia)", "Pageview log"),
        ("SQL dump (Wikipedia page_props)", "SQL dump, page_props"),
        ("SQL dump (Simple Wikipedia categorylinks)", "SQL dump, categorylinks"),
        ("JSON events (GitHub Archive)", "JSON events"),
    ]),
    ("Containers (opened)", [
        ("gzipped log (NASA, gzip -6)", "gzipped log"),
        ("tar.gz of mixed objects", "tar.gz, mixed"),
        ("Jar (Guava, 2,059 entries)", ".jar (Guava)"),
        ("Source zip (GitHub)", "source .zip"),
        ("PowerPoint, text (.pptx)", ".pptx"),
        ("Word, screenshots (.docx)", ".docx, screenshots"),
        ("Excel sheet (.xlsx)", ".xlsx"),
        ("PDF paper (pdfTeX)", "PDF, text"),
        ("PDF paper with figures", "PDF, figures"),
        ("PNG photo", "PNG photo"),
        ("JPEG photo", "JPEG photo"),
    ]),
    ("Plain data", [
        ("Wikipedia XML text (enwik8)", "Wikipedia text"),
        ("English text (Silesia dickens)", "English text"),
        ("Source tree tar (Linux 6.10)", "Linux source tar"),
        ("OS image tar (Ubuntu root)", "Ubuntu root tar"),
        ("Executable (Silesia mozilla)", "Executable"),
        ("Database (Silesia osdb)", "Database file"),
        ("Medical image (Silesia x-ray)", "X-ray image"),
        ("Parquet (NYC taxi, zstd inside)", "Parquet"),
    ]),
]
RECORD_KINDS = {"logs", "sql"}

FONT = "font-family='-apple-system,system-ui,Segoe UI,Helvetica,Arial,sans-serif'"
GLYD, OTHER, LESS, MORE, GRID, TEXT, MUTED = "#0b5fff", "#8a8a8a", "#2a9d4a", "#d64545", "#e4e4e4", "#111", "#666"


def load(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[(r["label"], r["codec"])] = r
    return rows


def best(rows, label, codecs):
    have = [rows[(label, c)] for c in codecs if (label, c) in rows]
    return min(have, key=lambda r: r["out"]) if have else None


def glyd_fast(rows, label):
    r = rows[(label, "Glyd --max")]
    if r["kind"] in RECORD_KINDS:
        return best(rows, label, ["Glyd --max", "Glyd --max -r"])
    return r


def glyd_strong(rows, label):
    r = rows[(label, "Glyd --ultra")]
    if r["kind"] in RECORD_KINDS:
        return best(rows, label, ["Glyd --ultra", "Glyd --ultra -r"])
    return r


def pct(a, b):
    return 100.0 * (a["out"] / b["out"] - 1)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;")


def bar_chart(rows, title, subtitle, codecs, glyd_of):
    """Horizontal bars, one per data type: Glyd's bytes against the
    smallest output of `codecs` on that data, in percent (below zero:
    Glyd is smaller)."""
    left, right, top = 210, 80, 70
    width, bar_h, pitch, group_gap = 900, 17, 25, 30
    vals = {label: pct(glyd_of(rows, label), best(rows, label, codecs)) for _, items in GROUPS for label, _ in items}
    lo, hi = min(min(vals.values()), -10), max(max(vals.values()), 10)
    lo, hi = math.floor(lo / 10) * 10 - 6, math.ceil(hi / 10) * 10 + 6
    x0, x1 = left, width - right
    x_of = lambda v: x0 + (v - lo) * (x1 - x0) / (hi - lo)
    height = top + sum(group_gap + pitch * len(items) for _, items in GROUPS) + 44
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' {FONT} font-size='13'>",
           f"<rect width='{width}' height='{height}' fill='white'/>",
           f"<text x='16' y='28' font-size='18' font-weight='600' fill='{TEXT}'>{esc(title)}</text>",
           f"<text x='16' y='50' fill='{MUTED}'>{esc(subtitle)}</text>"]
    y_end = height - 34
    v = math.ceil((lo + 6) / 10) * 10
    while v <= hi - 6:
        x = x_of(v)
        out.append(f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{y_end}' stroke='{GRID if v else '#999'}' stroke-width='{1 if v else 1.5}'/>")
        out.append(f"<text x='{x:.1f}' y='{height - 16}' text-anchor='middle' fill='{MUTED}'>{v:+d}%</text>")
        v += 10
    out.append(f"<text x='{(x0 + x1) / 2:.0f}' y='{height - 2}' text-anchor='middle' fill='{MUTED}' font-size='12'>← Glyd's file is smaller · larger →</text>")
    y = top
    for group, items in GROUPS:
        y += group_gap
        out.append(f"<text x='{left - 8}' y='{y - 9}' text-anchor='end' font-weight='600' fill='{TEXT}'>{esc(group)}</text>")
        for label, short in items:
            p = vals[label]
            col = LESS if p < -0.5 else (MORE if p > 0.5 else OTHER)
            xa, xb = sorted((x_of(0), x_of(p)))
            by = y + (pitch - bar_h) / 2
            out.append(f"<text x='{left - 8}' y='{y + pitch / 2 + 4.5:.1f}' text-anchor='end' fill='{TEXT}'>{esc(short)}</text>")
            out.append(f"<rect x='{xa:.1f}' y='{by:.1f}' width='{max(xb - xa, 1):.1f}' height='{bar_h}' rx='2' fill='{col}'/>")
            tx, anchor = (xa - 6, "end") if p < 0 else (xb + 6, "start")
            out.append(f"<text x='{tx:.1f}' y='{y + pitch / 2 + 4.5:.1f}' text-anchor='{anchor}' fill='{TEXT}' font-weight='600'>{'0%' if abs(p) < 0.5 else f'{p:+.0f}%'}</text>")
            y += pitch
    out.append("</svg>")
    return "\n".join(out)


def scatter(rows, panels, title, subtitle):
    """Ratio against read speed (one thread, log scale), one panel per input."""
    width, height, top = 900, 420, 70
    pw = (width - 40) // len(panels)
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' {FONT} font-size='12'>",
           f"<rect width='{width}' height='{height}' fill='white'/>",
           f"<text x='16' y='26' font-size='17' font-weight='600' fill='{TEXT}'>{esc(title)}</text>",
           f"<text x='16' y='46' fill='{MUTED}'>{esc(subtitle)}</text>"]
    for pi, (label, name, codecs) in enumerate(panels):
        px = 20 + pi * pw
        x0, x1, y0, y1 = px + 44, px + pw - 16, height - 44, top + 24
        pts = [rows[(label, c)] for c in codecs if (label, c) in rows]
        xs = [p["d_mbps"] for p in pts]
        lo = 10 ** math.floor(math.log10(max(min(xs), 0.1)))
        hi = 10 ** math.ceil(math.log10(max(xs)))
        ymax = max(p["ratio"] for p in pts) * 1.08
        step = 1 if ymax < 8 else (2 if ymax < 16 else 5)
        X = lambda v: x0 + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)
        Y = lambda v: y0 - v / ymax * (y0 - y1)
        out.append(f"<text x='{(x0 + x1) / 2:.0f}' y='{top + 12}' text-anchor='middle' font-weight='600' fill='{TEXT}'>{esc(name)}</text>")
        v = lo
        while v <= hi:
            out.append(f"<line x1='{X(v):.1f}' y1='{y1}' x2='{X(v):.1f}' y2='{y0}' stroke='{GRID}'/>")
            out.append(f"<text x='{X(v):.1f}' y='{y0 + 16}' text-anchor='middle' fill='{MUTED}'>{v:g}</text>")
            v *= 10
        t = 0
        while t <= ymax:
            out.append(f"<line x1='{x0}' y1='{Y(t):.1f}' x2='{x1}' y2='{Y(t):.1f}' stroke='{GRID}'/>")
            out.append(f"<text x='{x0 - 6}' y='{Y(t) + 4:.1f}' text-anchor='end' fill='{MUTED}'>{t:g}×</text>")
            t += step
        out.append(f"<text x='{(x0 + x1) / 2:.0f}' y='{height - 8}' text-anchor='middle' fill='{MUTED}' font-size='11'>decompress MB/s, one thread (log scale)</text>")
        out.append(f"<text transform='translate({px + 12},{(y0 + y1) / 2:.0f}) rotate(-90)' text-anchor='middle' fill='{MUTED}' font-size='11'>ratio</text>")
        # Labels: the first spot of right, left, above, below, then right
        # at growing offsets, that overlaps no point and no earlier label.
        boxes = []
        pts_xy = [(X(p["d_mbps"]), Y(p["ratio"])) for p in pts]
        for p, (cx, cy) in sorted(zip(pts, pts_xy), key=lambda t: (-t[0]["ratio"], t[1][0])):
            is_glyd = p["codec"].startswith("Glyd")
            out.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{5 if is_glyd else 4}' fill='{GLYD if is_glyd else OTHER}'/>")
            w, h = 6.3 * len(p["codec"]), 12
            spots = [(cx + 8, cy - 6), (cx - 8 - w, cy - 6), (cx - w / 2, cy - 18), (cx - w / 2, cy + 7)]
            for d in (12, -12, 24, -24, 36, -36, 48, -48, 60, -60):
                spots += [(cx + 8, cy - 6 + d), (cx - 8 - w, cy - 6 + d), (cx - w / 2, cy - 6 + d)]

            def free(bx, by):
                if bx < x0 or bx + w > x1 + 12 or by < y1 - 14 or by + h > y0:
                    return False
                for (ox, oy, ow, oh) in boxes:
                    if bx < ox + ow and bx + w > ox and by < oy + oh and by + h > oy:
                        return False
                for (qx, qy) in pts_xy:
                    if bx - 3 < qx < bx + w + 3 and by - 3 < qy < by + h + 3:
                        return False
                return True
            bx, by = next(((sx, sy) for sx, sy in spots if free(sx, sy)), spots[0])
            boxes.append((bx, by, w, h))
            out.append(f"<text x='{bx:.1f}' y='{by + 10:.1f}' fill='{GLYD if is_glyd else TEXT}' font-size='11' font-weight='{600 if is_glyd else 400}'>{esc(p['codec'])}</text>")
    out.append("</svg>")
    return "\n".join(out)


def table(rows):
    print("| Data | vs gzip -6 | vs zstd -3 | vs zstd -19/-22 | vs xz -9e | vs brotli -11 | Read MB/s, Glyd --max · zstd -3 |")
    print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for group, items in GROUPS:
        print(f"| **{group}** | | | | | | |")
        for label, short in items:
            f, s = glyd_fast(rows, label), glyd_strong(rows, label)
            z3 = rows[(label, "zstd -3")]
            cells = [pct(f, rows[(label, "gzip -6")]), pct(f, z3),
                     pct(s, best(rows, label, ["zstd -19", "zstd -22 --long"])),
                     pct(s, best(rows, label, ["xz -6", "xz -9e"])), pct(s, rows[(label, "brotli -11")])]
            fmt = lambda p: f"**{p:+.0f}%**" if p < -0.5 else f"{p:+.0f}%"
            m = rows[(label, "Glyd --max")]
            print(f"| {short} | " + " | ".join(fmt(p) for p in cells) + f" | {m['d_mbps']:,.0f} · {z3['d_mbps']:,.0f} |")


def main():
    rows = load(sys.argv[1])
    if sys.argv[2:] == ["--table"]:
        table(rows)
        return
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    fast = bar_chart(rows, "Glyd against zstd -3: bytes stored",
                     "Glyd --max (-r on records) against zstd -3 on the same data; one thread, M1 Max. Green: Glyd's file is smaller.",
                     ["zstd -3"], glyd_fast)
    strong = bar_chart(rows, "Glyd against the strongest of zstd -19/-22, xz -9e and brotli -11",
                       "Glyd --ultra (-r on records) against whichever of the three is smallest on that data. Green: Glyd's file is smaller.",
                       ["zstd -19", "zstd -22 --long", "xz -6", "xz -9e", "brotli -11"], glyd_strong)
    codecs = ["lz4 -1", "gzip -6", "zstd -1", "zstd -3", "zstd -9", "zstd -19", "zstd -22 --long", "brotli -5", "brotli -11", "xz -6", "xz -9e",
              "Glyd default", "Glyd --max", "Glyd --max -r", "Glyd --ultra", "Glyd --ultra -r", "Glyd --cold -r"]
    speed = scatter(rows, [("Source tree tar (Linux 6.10)", "Linux source tar (plain data)", [c for c in codecs if not c.endswith("-r")]),
                           ("Web server log (NASA)", "Web server log (records)", codecs)],
                    "Ratio against read speed", "Every codec's level on the same 32 MB, one thread, M1 Max; up and to the right is better")
    for name, svg in [("bytes-fast-tier.svg", fast), ("bytes-strong-tier.svg", strong), ("ratio-vs-read-speed.svg", speed)]:
        open(os.path.join(out, name), "w").write(svg + "\n")


if __name__ == "__main__":
    main()
