import re, sys
t = open(sys.argv[1]).read().split("\n")
want = set(int(x) for x in sys.argv[2].split(","))
rows = []
for i, l in enumerate(t):
    m = re.search(r"Compiling entry function .(\S+). for", l)
    if not m or "step_kernel" not in m.group(1):
        continue
    nm = m.group(1)
    mm = re.search(r"Tiered2oILi(\d+)EELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)(?:ELb(\d))?E", nm)
    if not mm:
        continue
    opt, mt, unr, nw, mb, sw = mm.groups()
    if mt != "1" or int(opt) not in want:
        continue
    j = i + 1
    regs = None
    stack = None
    while j < len(t) and "Compiling entry" not in t[j]:
        r = re.search(r"Used (\d+) registers", t[j])
        if r:
            regs = r.group(1)
        s2 = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", t[j])
        if s2:
            stack = s2.groups()
        j += 1
    rows.append((int(opt), sw or "0", unr, nw, int(mb), regs, stack))
for r in sorted(rows):
    print("opt=%s SW=%s unr=%s nw=%s mb=%s regs=%s stack/spill=%s" % r)
