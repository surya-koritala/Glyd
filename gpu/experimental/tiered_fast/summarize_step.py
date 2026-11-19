"""Summarize a step_job.sh results directory: the clk/step tables (ratios to the 12-bit and the 8-warp tiered kernels) and the gates' lines.
python3 summarize_step.py RESULTS_DIR"""
import csv
import glob
import os
import re
import sys

ROW = re.compile(r"^(\S.*?)\s+(\d+\.\d)\s+(\d+\.\d)\s+(\d\.\d{3})\s+(ok|TOO FAR)(.*)$")


def table(path):
    head, rows = None, []
    for line in open(path):
        if re.match(r"^\S+: \d+x\d+, M=", line):
            head = line.strip()
        m = ROW.match(line.rstrip("\n"))
        if m:
            rows.append((m.group(1).strip(), float(m.group(2)), float(m.group(3)), float(m.group(4)), m.group(5) + m.group(6)))
    return head, rows


def find(rows, name):
    return next((r for r in rows if r[0] == name), None)


def show(d):
    gate = []
    for path in sorted(glob.glob(os.path.join(d, "*.txt"))):
        if os.path.basename(path) in ("summary.txt", "nvidia-smi-q.txt"):
            continue
        head, rows = table(path)
        if not rows:
            continue
        ref12 = find(rows, "12-bit lib")
        old = find(rows, "tiered, 8 warps")
        print(f"\n{os.path.basename(path)}: {head}")
        print(f"  {'variant':26s} {'clk/step':>9s} {'GHz':>6s} {'vs 12-bit':>10s} {'vs 8 warps':>11s}  check")
        for name, ns, clk, ghz, chk in rows:
            a = f"{clk / ref12[2]:.3f}x" if ref12 else "-"
            b = f"{clk / old[2]:.3f}x" if old else "-"
            print(f"  {name:26s} {clk:9.1f} {ghz:6.3f} {a:>10s} {b:>11s}  {chk}")
        s = find(rows, "tiered lib")
        n = ref12
        base = os.path.basename(path)
        if s and n and (base.startswith("stream_u10_m") or base.startswith("hot_o10_m")):
            gate.append((base[:-4], s[2], n[2], s[2] / n[2]))
    if gate:
        print("\nthe tiered library kernel against the 12-bit one (clk/step):")
        for name, a, b, r in gate:
            print(f"  {name:20s} {a:8.1f} {b:8.1f}  {r:.3f}x" + ("  (the A100 gate: within 1.10x: " + ("met" if r <= 1.10 else "MISSED") + ")" if name in ("stream_u10_m1", "stream_u10_m8") else ""))


def load(path):
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith('"ID","Process ID"'))
    rows = list(csv.reader(lines[i:]))
    return {h: v for h, v in zip(rows[0], rows[2])}


def num(d, k):
    try:
        return float(d[k].replace(",", ""))
    except Exception:
        return None


def show_ncu(d):
    files = sorted(glob.glob(os.path.join(d, "ncu", "*.raw")))
    if not files:
        return
    keys = [("duration us", "gpu__time_duration.sum", 1e-3), ("registers", "launch__registers_per_thread", 1),
            ("warps active/SMSP", "smsp__warps_active.avg.per_cycle_active", 1), ("eligible/SMSP", "smsp__warps_eligible.avg.per_cycle_active", 1),
            ("issue active %", "smsp__issue_active.avg.pct_of_peak_sustained_active", 1), ("ALU pipe %", "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active", 1),
            ("FMA-heavy pipe %", "sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_elapsed", 1), ("tensor pipe %", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", 1),
            ("LSU wavefronts %", "l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed", 1), ("DRAM %", "dram__throughput.avg.pct_of_peak_sustained_elapsed", 1),
            ("smem wavefronts", "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum", 1), ("smem bank conflicts", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", 1),
            ("warp instr", "smsp__inst_executed.sum", 1)]
    stalls = ["wait", "short_scoreboard", "long_scoreboard", "not_selected", "math_pipe_throttle", "barrier", "mio_throttle", "branch_resolving", "no_instruction", "dispatch_stall", "lg_throttle"]
    data = {os.path.basename(f)[:-4]: load(f) for f in files}
    names = list(data)
    print("\nncu (one launch each; per-warp CPI = 1 + the stall columns):")
    print(f"  {'':22s}" + "".join(f"{n:>13s}" for n in names))
    for label, k, scale in keys:
        vals = [num(data[n], k) for n in names]
        print(f"  {label:22s}" + "".join(f"{(v * scale if v is not None else float('nan')):13.2f}" for v in vals))
    for s in stalls:
        k = f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio"
        vals = [num(data[n], k) for n in names]
        if any(v for v in vals):
            print(f"  {'stall ' + s:22s}" + "".join(f"{(v if v is not None else float('nan')):13.2f}" for v in vals))


if __name__ == "__main__":
    show(sys.argv[1])
    show_ncu(sys.argv[1])
