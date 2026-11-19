"""Count SASS instructions by kind, per kernel, from `cuobjdump -sass` on stdin.

    cuobjdump -sass decode.cubin | python sass_count.py [NAME_SUBSTRING=WEIGHTS ...]   # weights a thread, e.g. scales=512 decodeILi6=64

ALU = everything except memory (LD*, ST*, ATOM*, RED*), barriers, branches and returns (BRA, BRX, BAR, EXIT, RET, CALL,
BSSY, BSYNC, WARPSYNC, NOP), and special-register reads (S2R, S2UR, CS2R). LDS is counted on its own.
"""
import re, sys

kern, cur = {}, None
for line in sys.stdin:
    m = re.match(r"\s*Function : (\S+)", line)
    if m:
        cur = kern.setdefault(m.group(1), {"alu": 0, "lds": 0, "ldg": 0, "stg": 0, "ctl": 0, "ops": {}})
        continue
    m = re.match(r"\s*/\*[0-9a-f]+\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)(\.[A-Z0-9_.]+)?", line)
    if not m or cur is None:
        continue
    op = m.group(2)
    cur["ops"][op] = cur["ops"].get(op, 0) + 1
    if op.startswith("LDS"):
        cur["lds"] += 1
    elif op.startswith(("LDG", "LD", "LDC", "ULDC")):
        cur["ldg"] += 1
    elif op.startswith(("STG", "STS", "ST", "ATOM", "RED")):
        cur["stg"] += 1
    elif op in ("BRA", "BRX", "BAR", "EXIT", "RET", "CALL", "BSSY", "BSYNC", "WARPSYNC", "NOP", "BREAK", "S2R", "S2UR", "CS2R", "DEPBAR", "ERRBAR", "MEMBAR", "SYNCS", "ELECT", "VOTEU", "UBLKCP"):
        cur["ctl"] += 1
    else:
        cur["alu"] += 1
want = dict(a.split("=") for a in sys.argv[1:])
for name, k in kern.items():
    w = next((float(v) for key, v in want.items() if key in name), 1.0)
    top = sorted(k["ops"].items(), key=lambda kv: -kv[1])[:12]
    print(f"{name[:60]:60s} ALU {k['alu']:5d} ({k['alu'] / w:6.2f} a weight)  LDS {k['lds']:4d} LD {k['ldg']:4d} ST {k['stg']:4d} ctl {k['ctl']:4d}   {top}")
