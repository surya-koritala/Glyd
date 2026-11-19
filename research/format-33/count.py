"""Static SASS counts of f33_dec.cu's decode kernels (cuobjdump -sass of the binary), with gpu/experimental/tiered_fast/sassstat.py's
parser: each k_hot_* / k_dec_* kernel's instructions, its loops (back edges) and the opcode and pipe mix of its main loop.
    python count.py FILE.sass"""
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gpu", "experimental", "tiered_fast"))
import sassstat

text = open(sys.argv[1]).read()
parts = re.split(r"\n\s*Function : (\S+)\n", text)
for name, body in sorted(zip(parts[1::2], parts[2::2])):
    if not re.match(r"k_(hot|dec)_", name):
        continue
    tmp = sys.argv[1] + "." + name
    open(tmp, "w").write(body)
    ins = sassstat.parse(tmp)
    os.remove(tmp)
    ls = sorted(sassstat.loops(ins), key=lambda tb: tb[1] - tb[0])
    print(f"== {name}: {len(ins)} instructions, loops (instructions between target and back edge): " + ", ".join(f"{sum(1 for i in ins if t <= i[0] <= b)}" for t, b in ls))
    if ls:
        # the main loop: the outer one (a step a turn) for the tiered kernels; for rANS the loop over a strip's steps (the second largest)
        t, b = ls[-1] if not name.endswith("rans") else ls[-2] if len(ls) > 1 else ls[-1]
        sassstat.show(f"   loop {t:#06x}-{b:#06x}", [i for i in ins if t <= i[0] <= b])
