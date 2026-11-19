import csv, sys, collections
def parse(path):
    """-> {file: {line: (instr executed, source text)}}"""
    out = {}
    cur = None
    rd = csv.reader(open(path))
    cols = None
    for row in rd:
        if not row: continue
        if row[0] == "File Path" or row[0] == "File Name":
            cur = row[1]; out[cur] = {}; cols = None; continue
        if row[0] == "Line No":
            cols = row; continue
        if cols and row[0].isdigit():
            try:
                i = cols.index("Instructions Executed")
                n = int(row[i]) if row[i].isdigit() else 0
            except ValueError:
                n = 0
            out[cur][int(row[0])] = (n, row[1])
    return out
if __name__ == "__main__":
    steps = float(sys.argv[2])
    d = parse(sys.argv[1])
    for f, lines in d.items():
        tot = sum(n for n, _ in lines.values())
        print(f"{f.split('/')[-1]:30s} {tot / steps:8.1f}/step")
    f = [k for k in d if k.endswith("glyd_gpu.cu")][0]
    rows = sorted(((n, ln, s) for ln, (n, s) in d[f].items() if n), reverse=True)
    for n, ln, s in rows[:int(sys.argv[3]) if len(sys.argv) > 3 else 40]:
        print(f"{n / steps:8.1f}  L{ln:5d}  {s[:150]}")
