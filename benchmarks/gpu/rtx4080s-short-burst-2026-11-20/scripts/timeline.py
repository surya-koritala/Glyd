#!/usr/bin/env python3
"""The timeline of bursts of short requests: the client's sends and first tokens (load.py's JSON) with the server's submits and prompt passes (GLYD_ENGINE_TRACE_SCHED=1 on stderr).

    python3 timeline.py load.json server.log [--round N]

Times in ms from the burst's first send. One line a event; at the end, per request the terms of its first token time: send to submit (HTTP, tokenizing, the chat template), submit to the pass that
began its prompt (waiting), that pass, the pass's end to the client's first token.
"""
import json, re, sys

load = json.load(open(sys.argv[1]))
log = open(sys.argv[2]).read().splitlines()
subs = [(float(m.group(1)), int(m.group(2))) for l in log if (m := re.match(r"sched ([\d.]+) submit (\d+)", l))]
passes = []
for l in log:
    m = re.match(r"sched ([\d.]+) pass from ([\d.]+) of (\d+) sequences \[([\d, ]*)\], (\d+) tokens, ([\d.]+) ms", l)
    if m:
        passes.append((float(m.group(2)), float(m.group(1)), [int(x) for x in m.group(4).split(",") if x.strip()], int(m.group(5)), float(m.group(6))))
cells = [c for c in load["cells"] if c["prompt"] == "short" and c["conc"] >= 4]
for c in cells[-1:]:
    for r_i, rd in enumerate(c["rounds"]):
        reqs = sorted(rd["requests"], key=lambda r: r["sent"])
        t0 = reqs[0]["sent"]
        # (the server's submits of this burst: the ones within 0.5 s of the first send)
        ss = sorted((t, i) for t, i in subs if t0 - 0.01 <= t <= t0 + 0.5)
        print(f"== burst {r_i} of {len(reqs)} (first send at {t0:.3f})")
        ev = [(r["sent"] - t0, f"client sends #{k}") for k, r in enumerate(reqs)]
        ev += [(t - t0, f"server submit {i}") for t, i in ss]
        ev += [(a - t0, f"  prompt pass of {len(ids)} {ids} ({tok} tokens, {ms:.1f} ms), ends at {(b - t0) * 1e3:.1f}") for a, b, ids, tok, ms in passes if t0 - 0.01 <= a <= t0 + 0.5]
        ev += [(r["first"] - t0, f"client first token of #{k} (ttft {r['ttft'] * 1e3:.1f} ms)") for k, r in enumerate(reqs)]
        for t, what in sorted(ev):
            print(f"{t * 1e3:8.1f}  {what}")
        break
