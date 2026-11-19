"""Per-SASS-instruction stall samples from an ncu --page source --print-source cuda,sass export (.src2): the instructions with the
most samples and their stall reasons, and the samples by opcode and by CUDA source line.   python sass_stalls.py FILE.src2 [N]"""
import csv, sys, collections, re

def rows(path):
    rd = csv.reader(open(path))
    cols = None
    cur_file = None
    cur_line = None
    for r in rd:
        if not r: continue
        if r[0] in ("File Path", "File Name"):
            cur_file = r[1]; continue
        if r[0] == "Function Name":
            continue
        if r[0] == "Line No":
            cols = r; continue
        if cols is None: continue
        if r[0].isdigit():
            cur_line = (cur_file, int(r[0]), r[1]); continue
        if r[0] == "" and len(r) > 3 and r[2].startswith("0x"):
            d = dict(zip(cols, r))
            yield cur_line, d

def num(d, k):
    try: return float(d[k])
    except Exception: return 0.0

if __name__ == "__main__":
    path = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    data = list(rows(path))
    tot = sum(num(d, "Warp Stall Sampling (All Samples)") for _, d in data)
    ex = sum(num(d, "Instructions Executed") for _, d in data)
    print(f"{len(data)} SASS rows, {tot:.0f} stall samples, {ex:.0f} instructions executed")
    stall_cols = [c for c in data[0][1] if c.startswith("stall_") and "(Not Issued)" not in c]
    bytall = collections.Counter()
    for _, d in data:
        for c in stall_cols: bytall[c] += num(d, c)
    print("stall reasons (share of samples):", ", ".join(f"{c[6:]} {100 * v / tot:.1f}%" for c, v in bytall.most_common(10)))
    byop = collections.defaultdict(lambda: [0.0, 0.0])
    for _, d in data:
        op = d["Source"].split()[0] if d["Source"].split() else "?"
        if op.startswith("@"): op = d["Source"].split()[1]
        op = op.split(".")[0]
        byop[op][0] += num(d, "Warp Stall Sampling (All Samples)")
        byop[op][1] += num(d, "Instructions Executed")
    print("by opcode (samples, executed):", ", ".join(f"{k} {v[0] / tot * 100:.1f}% ({v[1] / ex * 100:.1f}%)" for k, v in sorted(byop.items(), key=lambda kv: -kv[1][0])[:14]))
    byline = collections.defaultdict(float)
    for (f, ln, src), d in data:
        byline[(f.split("/")[-1], ln, src[:90])] += num(d, "Warp Stall Sampling (All Samples)")
    print("source lines by stall samples:")
    for (f, ln, src), v in sorted(byline.items(), key=lambda kv: -kv[1])[:N]:
        print(f"  {100 * v / tot:5.1f}%  {f}:{ln}  {src}")
