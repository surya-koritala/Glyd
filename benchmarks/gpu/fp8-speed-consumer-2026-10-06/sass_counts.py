"""sass_counts.py SASS.txt: per f8_linear_kernel<NTL, KIND, 0, 0> (cuobjdump -sass of the library's cubin for one architecture): the instructions of the
kernel, of its step loop (the code between the step's barrier and its branch back, every path counted once: the plain patches', the literals' and
the unstaged variable parts' too), the MMAs, and the step loop's instructions by opcode."""
import collections, re, sys

fn, body = None, collections.defaultdict(list)
for line in open(sys.argv[1]):
    m = re.search(r"Function : (\S+)", line)
    if m:
        fn = m.group(1)
        continue
    m = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;", line)
    if m and fn:
        body[fn].append((int(m.group(1), 16), m.group(2)))
for f, ins in sorted(body.items()):
    m = re.search(r"f8_linear_kernelILi(\d+)ELi(\d+)ELi0ELi0E", f)
    if not m:
        continue
    ntl, kind = int(m.group(1)), int(m.group(2))
    ops = [re.sub(r"^@!?U?P\w+\s+", "", t).split()[0] for _, t in ins]
    mma = [i for i, o in enumerate(ops) if "MMA" in o]
    bars = [i for i, o in enumerate(ops) if o.startswith("BAR.SYNC")]
    # the step loop: from the barrier before the first MMA to the last backward branch that jumps over it
    head = max(b for b in bars if b < mma[0])
    addr = {a: i for i, (a, _) in enumerate(ins)}
    back = [i for i, (a, t) in enumerate(ins) if ops[i].startswith("BRA") and (mt := re.search(r"`\(\.L_x_\d+\)|0x([0-9a-f]+)", t)) and mt.group(1) and int(mt.group(1), 16) <= ins[head][0] and i > mma[-1]]
    tail = max(back) if back else len(ins) - 1
    loop = ops[head:tail + 1]
    c = collections.Counter(o.split(".")[0] for o in loop)
    print(f"NTL={ntl} KIND={kind:<2} instructions {len(ins):5d}  step loop {len(loop):5d}  MMAs {len(mma):3d} ({ops[mma[0]].split('.')[0]}.{ops[mma[0]].split('.')[1]})  "
          + ", ".join(f"{k} {v}" for k, v in c.most_common(10)))
