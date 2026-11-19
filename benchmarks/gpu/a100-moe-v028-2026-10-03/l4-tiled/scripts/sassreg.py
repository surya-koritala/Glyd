"""mma_gemm_big_kernel's stall samples by SASS instruction (`ncu --page source --csv --print-source sass`: the job's prof/sass-T.csv.gz, or one launch's
file), told apart by where the instructions sit: the consumers' main loop (up to the last HMMA), the producers' loop (back from the first LDGSTS, the
instructions that run a hundred thousand times), and what lies between (the epilogue, and the paths a step rarely takes). A file may hold several
launches, each printed twice by ncu: the distinct ones are taken, in order (the job's captures: the gate and up product, then the down).

    python sassreg.py sass-2048.csv.gz [more files]

Four producer warps and four consumer warps a block (mma_gemm_big_kernel<Fmt, 4, 4, 3, 2>): a warp's samples are its time resident, so the last column,
the producers' share of a consumer warp's life, is the producers' samples over the consumers' (loop and epilogue; the producers' rare paths are counted
as the consumers'). The regions are told apart by instruction, not by source line."""
import collections
import csv
import gzip
import io
import re
import sys


def num(x):
    try:
        return float(x.replace(",", ""))
    except ValueError:
        return 0.0


def sections(path):
    """[rows of each distinct launch's table], from a file of one or more."""
    text = (gzip.open(path, "rt") if path.endswith(".gz") else open(path)).read()
    parts = ['"Kernel Name"' + t for t in text.split('"Kernel Name"') if t.strip()]
    out, seen = [], None
    for t in parts:
        rows = list(csv.reader(io.StringIO(t)))
        hdr = rows[1]
        ix = {h: i for i, h in enumerate(hdr)}
        d = []
        for r in rows[2:]:
            if len(r) < 10:
                continue
            m = re.match(r"\s*(@!?U?P\d+\s+)?([A-Z0-9_]+(?:\.[A-Z0-9_]+)*)", r[ix["Source"]])
            d.append(dict(op=m.group(2).split(".")[0] if m else "?", samp=num(r[ix["Warp Stall Sampling (All Samples)"]]), ex=num(r[ix["Instructions Executed"]]),
                          stalls={h[6:]: num(r[ix[h]]) for h in hdr if h.startswith("stall_") and "Not Issued" not in h}))
        key = (round(sum(x["ex"] for x in d)), round(sum(x["samp"] for x in d)))
        if key != seen:
            out.append(d)
        seen = key
    return out


def regions(d):
    hm1 = max(i for i, x in enumerate(d) if x["op"] == "HMMA")
    lg0 = next(i for i, x in enumerate(d) if x["op"] == "LDGSTS")
    p0 = lg0
    while p0 > hm1 + 40 and d[p0 - 1]["ex"] > 1e5:
        p0 -= 1
    return {"consumers' main loop": d[: hm1 + 40], "epilogue and rare code": d[hm1 + 40 : p0], "producers": d[p0:]}


def main(path):
    for k, d in enumerate(sections(path)):
        tot, tex = sum(x["samp"] for x in d), sum(x["ex"] for x in d)
        print(f"{path.split('/')[-1]}, launch {k}: {len(d)} static instructions, {tex / 1e6:.1f} M executed")
        reg = regions(d)
        for name, seg in reg.items():
            s, e = sum(x["samp"] for x in seg), sum(x["ex"] for x in seg)
            st = collections.Counter()
            for x in seg:
                for r, v in x["stalls"].items():
                    st[r] += v
            print(f"  {name:24s} static {len(seg):6d}  executed {e / 1e6:7.2f} M ({100 * e / tex:4.1f}%)  samples {100 * s / tot:5.1f}%  [{', '.join(f'{r} {100 * v / max(s, 1):.0f}%' for r, v in st.most_common(4))}]")
        cons = sum(x["samp"] for n, seg in reg.items() if n != "producers" for x in seg)
        prod = sum(x["samp"] for x in reg["producers"])
        mix = collections.Counter()
        for x in reg["producers"]:
            mix[x["op"]] += x["ex"]
        print(f"  producers' share of a consumer warp's life {100 * prod / cons:.0f}%; the epilogue's share of it {100 * sum(x['samp'] for x in reg['epilogue and rare code']) / cons:.0f}%; "
              f"the producers' instructions: {', '.join(f'{o} {100 * v / sum(mix.values()):.0f}%' for o, v in mix.most_common(6))}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
