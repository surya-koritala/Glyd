import collections, glob, os, re, sys
root = sys.argv[1]
def stamp(f):
    return re.search(r"(\d{8}-\d{6})", os.path.basename(f)).group(1)
for run in sorted(glob.glob(os.path.join(root, "runs", "*"))):
    logs = sorted(glob.glob(os.path.join(run, "glyd-logs", "*.txt")), key=stamp)
    checked, purposeful = logs[:4], logs[4:]  # the acceptance checks the first four (run, serve, serve with a key twice); the rest stop a server on purpose (Ctrl-C while loading, a killed engine)
    kinds = collections.Counter()
    tb = {"checked": 0, "stopped on purpose": 0}
    oom = 0
    for group, files in (("checked", checked), ("stopped on purpose", purposeful)):
        for f in files:
            text = open(f, errors="replace").read()
            tb[group] += text.split("[shutdown]")[0].count("Traceback")
            oom += text.count("with OOM")
            if group == "checked":
                for line in text.splitlines():
                    m = re.search(r"\bWARNING\b[^\[]*\[[^\]]*\]\s*(.*)", line)
                    if m:
                        kinds[re.sub(r"\d+(\.\d+)?", "N", m.group(1))[:100]] += 1
    inst = [open(f, errors="replace").read() for f in glob.glob(os.path.join(run, "logs", "install*.txt"))]
    uvw = sum(t.lower().count("warning:") for t in inst)
    print(f"{os.path.basename(run)}: {len(checked)} server logs checked + {len(purposeful)} of cases that stop a server on purpose. Tracebacks in the checked: {tb['checked']}; in the others: {tb['stopped on purpose']}; allocator OOM warnings in all: {oom}; 'warning:' lines in the install logs: {uvw}")
    for msg, n in kinds.most_common(8):
        print(f"    {n:4d} x WARNING {msg}   (checked logs)")
