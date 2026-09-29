"""Two builds' SASS for one architecture (cuobjdump -sass of main's library and of this tree's), kernel by kernel. The
12-bit layout's kernels (those of Nib): their instructions by kind, main's -> this tree's, and whether their memory,
tensor-core, barrier and branch instructions are the same ones in the same order (the schedule), the decode's integer
instructions and constant loads apart; every other kernel: the same instructions in the same order, or listed.

    python sass_diff.py MAIN.sass TREE.sass [MAIN.res TREE.res]      (.res: cuobjdump -res-usage, for the registers)"""
import re, subprocess, sys
from collections import Counter

MMA = {"HMMA", "HGMMA", "IMMA", "WARPGROUP"}
CONST = {"LDC", "ULDC", "S2R", "S2UR", "CS2R"}  # constants and special registers
MEM = {"LDG", "STG", "LDS", "STS", "LDSM", "LDGSTS", "LDGDEPBAR", "LD", "ST", "LDL", "STL", "ATOM", "ATOMG", "ATOMS", "RED", "REDG", "UTMALDG", "UTMASTG",
       "UTMACCTL", "UBLKCP", "SYNCS", "CCTL", "MEMBAR", "FENCE", "ERRBAR", "STSM"}
CTL = {"BAR", "DEPBAR", "WARPSYNC", "BSSY", "BSYNC", "BRA", "BRX", "JMP", "EXIT", "CALL", "RET", "YIELD", "NANOSLEEP", "BREAK", "KILL", "BPT", "ELECT", "VOTE",
       "VOTEU", "UCGABAR_ARV", "UCGABAR_WAIT", "ACQBULK", "CGAERRBAR", "PREEXIT", "WARPGROUP.ARRIVE", "WARPGROUP.DEPBAR"}
FMA = {"IMAD", "IMUL", "FFMA", "FADD", "FMUL", "HFMA2", "HADD2", "HMUL2", "FSEL", "FMNMX", "MUFU", "F2F", "F2I", "I2F", "FRND", "FSETP", "F2FP", "FCHK"}


def kind(op):
    b = op.split(".")[0]
    if b in MMA:
        return "mma"
    if b in CONST:
        return "const"
    if b in MEM:
        return "mem"
    if b in CTL or op in CTL:
        return "ctl"
    if b == "NOP":
        return "nop"
    if b in FMA:
        return "fma"
    if b == "SHFL":
        return "shfl"
    return "uni" if b.startswith("U") else "alu"


def functions(path):
    fns, name = {}, None
    for line in open(path):
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            name = m.group(1)
            fns[name] = []
            continue
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
        if m and name:
            fns[name].append(m.group(1))
    return fns


def registers(path):
    out = {}
    for line in open(path):
        m = re.search(r"Function (\S+):\s*REG:(\d+)", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


a, b = functions(sys.argv[1]), functions(sys.argv[2])
ra, rb = (registers(sys.argv[3]), registers(sys.argv[4])) if len(sys.argv) > 4 else ({}, {})
assert set(a) == set(b), ("kernels differ", sorted(set(a) ^ set(b))[:10])
names = sorted(a)
try:
    short = dict(zip(names, subprocess.run(["c++filt", "-p"], input="\n".join(names), capture_output=True, text=True).stdout.split("\n")))
except OSError:
    short = {n: n for n in names}
KINDS = ("alu", "fma", "uni", "const", "shfl", "mem", "mma", "ctl")
twelve = [n for n in names if "3Nib" in n]
same = sum(a[n] == b[n] for n in names if n not in twelve)
print(f"{sys.argv[1]} -> {sys.argv[2]}: {len(names)} kernels; the {len(names) - len(twelve)} not of the 12-bit layout: {same} the same instructions in the same order")
for n in names:
    if n not in twelve and a[n] != b[n]:
        print(f"  DIFFERS: {short.get(n, n)}")
print(f"the 12-bit layout's {len(twelve)} kernels, main's -> this tree's (instructions by kind; schedule: memory, tensor-core, barrier and branch instructions in order):")
changed = 0
for n in twelve:
    ca, cb = Counter(kind(o) for o in a[n]), Counter(kind(o) for o in b[n])
    sched = [o for o in a[n] if kind(o) in ("mem", "mma", "ctl")] == [o for o in b[n] if kind(o) in ("mem", "mma", "ctl")]
    changed += not sched
    regs = f"regs {ra.get(n, '?')}->{rb.get(n, '?')} " if ra else ""
    print(f"  {short.get(n, n)[:90]:90s} {regs}" + " ".join(f"{k} {ca[k]}->{cb[k]}" if ca[k] != cb[k] else f"{k} {ca[k]}" for k in KINDS)
          + f" | total {sum(ca.values()) - ca['nop']}->{sum(cb.values()) - cb['nop']} | schedule {'the same' if sched else 'CHANGED'}")
print(f"schedule changed in {changed} of {len(twelve)} kernels of the 12-bit layout")
