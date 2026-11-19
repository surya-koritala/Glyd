#!/usr/bin/env python3
"""Static SASS statistics of one kernel (cuobjdump -sass output): the instructions, the loops (back edges), and per loop the
mix by opcode and by the pipe class it runs on.   python sassstat.py FILE.sass [--loops] [--all]"""
import re, sys, collections

INS = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;\s*/\*")
BRA_T = re.compile(r"0x([0-9a-f]+)")

# the pipe an opcode runs on (Ampere/Hopper; by Nsight Compute's names), coarse
ALU = {"LOP3", "SHF", "PRMT", "IADD3", "ISETP", "SEL", "POPC", "FLO", "LEA", "IABS", "IMNMX", "BMSK", "P2R", "R2P", "SGXT", "VOTE", "VIADD", "VIMNMX", "FSEL", "FMNMX", "LOP", "ISET", "ICMP", "BREV", "I2I", "F2F", "MOV", "SHL", "SHR"}
FMA = {"FFMA", "FADD", "FMUL", "FSETP", "FSET", "HFMA2", "FCHK"}
FMAH = {"IMAD", "IMUL", "IMAD.MOV"}
LSU = {"LDS", "STS", "LDG", "STG", "LDSM", "LDGSTS", "ATOM", "ATOMS", "RED", "LD", "ST", "ATOMG", "LDC", "LDL", "STL", "MEMBAR", "CCTL"}
MIO = {"SHFL", "LDS", "STS", "LDSM", "BAR", "ATOMS", "REDUX"}
TENSOR = {"HMMA", "IMMA", "HGMMA", "BMMA", "DMMA"}
UNI = {"UIADD3", "ULOP3", "USHF", "UMOV", "USEL", "UISETP", "ULEA", "UPRMT", "UFLO", "UPOPC", "UIMAD", "UP2UR", "UCLEA", "UMOV", "USGXT", "UBREV", "VOTEU"}
CTRL = {"BRA", "BSSY", "BSYNC", "EXIT", "WARPSYNC", "BAR", "DEPBAR", "NOP", "BREAK", "CALL", "RET", "YIELD", "SYNCS", "ELECT", "WARPGROUP", "BPT", "KILL", "NANOSLEEP"}

def root(op):
    return op.split(".")[0]

def pipe(op):
    r = root(op)
    if op.startswith("IMAD.MOV") or op.startswith("IMAD.SHL") or op.startswith("IMAD.U32") or op.startswith("IMAD.WIDE") or r == "IMAD":
        return "imad"
    if r in TENSOR: return "tensor"
    if r == "SHFL": return "shfl"
    if r in ("LDS", "STS", "LDSM"): return "smem"
    if r in ("LDG", "STG", "LDGSTS", "LD", "ST", "ATOM", "RED", "ATOMG", "LDL", "STL", "LDC", "CCTL"): return "gmem"
    if r == "REDUX": return "redux"
    if r in ("POPC", "FLO"): return "popc"  # XU/ALU quarter rate on some parts
    if r in ALU: return "alu"
    if r in FMA: return "fma"
    if r in CTRL or r.startswith("U"): return "ctrl"
    if r in ("MUFU", "F2I", "I2F", "F2F", "I2I", "FRND", "F2FP"): return "xu"
    if r == "S2R" or r == "CS2R" or r == "R2UR": return "ctrl"
    return "other"

def parse(path):
    ins = []
    for line in open(path):
        m = INS.match(line)
        if not m:
            continue
        a, pred, op, rest = int(m.group(1), 16), (m.group(2) or "").strip(), m.group(3), m.group(4)
        ins.append((a, pred, op, rest))
    return ins

def loops(ins):
    """back edges: a BRA whose target is at an earlier address -> (target, branch)."""
    out = []
    for a, pred, op, rest in ins:
        if root(op) == "BRA":
            m = BRA_T.findall(rest)
            if m:
                t = int(m[-1], 16)
                if t <= a:
                    out.append((t, a))
    return out

def mix(rng_ins):
    c = collections.Counter(root(op) if not op.startswith("IMAD") else op.split(".")[0] for _, _, op, _ in rng_ins)
    p = collections.Counter(pipe(op) for _, _, op, _ in rng_ins)
    return c, p

def show(name, rng_ins):
    c, p = mix(rng_ins)
    n = len(rng_ins)
    print(f"{name}: {n} instructions")
    print("  by pipe  :", ", ".join(f"{k} {v}" for k, v in p.most_common()))
    print("  by opcode:", ", ".join(f"{k} {v}" for k, v in c.most_common(30)))

if __name__ == "__main__":
    path = sys.argv[1]
    ins = parse(path)
    show("whole kernel", ins)
    for t, b in sorted(loops(ins)):
        rng = [i for i in ins if t <= i[0] <= b]
        show(f"loop {t:#07x}-{b:#07x}", rng)
