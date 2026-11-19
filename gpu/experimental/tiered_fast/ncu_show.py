import csv, sys, re
def load(path):
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith('"ID","Process ID"'))
    rows = list(csv.reader(lines[i:]))
    hdr, units, vals = rows[0], rows[1], rows[2]
    return {h: (v, u) for h, u, v in zip(hdr, units, vals)}
def num(d, k):
    try: return float(d[k][0].replace(",", ""))
    except Exception: return None
if __name__ == "__main__":
    pats = sys.argv[2:]
    d = load(sys.argv[1])
    for k, (v, u) in d.items():
        if any(re.search(p, k) for p in pats):
            print(f"{k:90s} {v:>20s} {u}")
