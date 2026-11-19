import csv, sys, re
def mix(path):
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith('"ID"'))
    rows = list(csv.reader(lines[i:]))
    hdr, vals = rows[0], rows[2]
    out = {}
    for h, v in zip(hdr, vals):
        if h.startswith("sass__"):
            tot, _, rest = v.partition(" (")
            d = {}
            for part in rest.rstrip(")").split("; "):
                if ":" in part:
                    k, n = part.split(": ")
                    d[k] = int(n)
            out[h] = (int(tot), d)
    return out
if __name__ == "__main__":
    steps = float(sys.argv[3])
    A, B = mix(sys.argv[1]), mix(sys.argv[2])
    for key in ["sass__inst_executed_per_opcode", "sass__inst_executed_per_opcode_pipeline"]:
        a, b = A[key], B[key]
        print(key, "total/step:", f"{a[0] / steps:.1f}", f"{b[0] / steps:.1f}")
        ks = sorted(set(a[1]) | set(b[1]), key=lambda k: -(a[1].get(k, 0) + b[1].get(k, 0)))
        for k in ks[:26]:
            print(f"  {k:10s} {a[1].get(k, 0) / steps:8.1f} {b[1].get(k, 0) / steps:8.1f}")
