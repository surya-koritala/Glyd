"""Key metrics of each kernel launch in an `ncu --page raw --csv` dump (the job's prof/raw-T.csv): the duration, the throughput percentages (SM, memory, L1,
L2, DRAM), the tensor pipe (wgmma on Hopper) and issue slot utilisation, the launch's registers and shared memory, and the warp stall reasons (the sampled
ones), one block a launch. Nsight Compute writes a launch a row and a metric a column, the first row the units (older versions: a row a metric).

    python ncu_summary.py raw-8192.csv [raw-232.csv ...]
"""
import csv
import re
import sys

KEYS = [  # (label, patterns: the first that matches a column of the dump)
    ("duration", [r"^gpu__time_duration\.sum$"]),
    ("SM throughput %", [r"^sm__throughput\.avg\.pct_of_peak_sustained_elapsed$"]),
    ("memory throughput %", [r"^gpu__compute_memory_throughput\.avg\.pct_of_peak_sustained_elapsed$"]),
    ("L1/TEX throughput %", [r"^l1tex__throughput\.avg\.pct_of_peak_sustained_elapsed$"]),
    ("L2 throughput %", [r"^lts__throughput\.avg\.pct_of_peak_sustained_elapsed$"]),
    ("DRAM throughput %", [r"^gpu__dram_throughput\.avg\.pct_of_peak_sustained_elapsed$", r"^dram__throughput\.avg\.pct_of_peak_sustained_elapsed$"]),
    ("tensor gmma pipe %", [r"^sm__pipe_tensor_op_gmma_cycles_active\.avg\.pct_of_peak_sustained_(active|elapsed)$"]),
    ("tensor pipe %", [r"^sm__pipe_tensor_cycles_active\.avg\.pct_of_peak_sustained_active$", r"^sm__pipe_tensor_op_hmma_cycles_active\.avg\.pct_of_peak_sustained_active$"]),
    ("tensor bf16 ops % of peak", [r"^sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off\.avg\.pct_of_peak_sustained_elapsed$"]),
    ("hmma issue % of peak", [r"^sm__inst_executed_pipe_tensor_op_hmma_v2\.avg\.pct_of_peak_sustained_active$"]),
    ("issue slots active %", [r"^smsp__issue_active\.avg\.pct_of_peak_sustained_active$"]),
    ("warp instructions", [r"^smsp__inst_executed\.sum$"]),
    ("local loads (spills)", [r"^sass__inst_executed_local_loads$"]),
    ("local stores (spills)", [r"^sass__inst_executed_local_stores$"]),
    ("L2 hit rate %", [r"^lts__t_sector_hit_rate\.pct$"]),
    ("L2 read sectors", [r"^lts__t_sectors_srcunit_tex_op_read\.sum$"]),
    ("DRAM bytes", [r"^dram__bytes\.sum$", r"^dram__bytes_read\.sum$"]),
    ("registers a thread", [r"^launch__registers_per_thread$"]),
    ("dynamic smem a block", [r"^launch__shared_mem_per_block_dynamic$"]),
    ("warps active %", [r"^sm__warps_active\.avg\.pct_of_peak_sustained_active$"]),
]
STALL = re.compile(r"^smsp__pcsamp_warps_issue_stalled_(\w+?)$")


def number(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


def launches_of(text):
    """[(id, kernel, grid, block, {metric: (value, unit)})] of the dump (either layout)."""
    start = text.find('"ID"')
    if start < 0:
        return []
    rows = list(csv.DictReader(text[start:].splitlines()))
    if rows and "Metric Name" in rows[0]:  # (a row a metric)
        out = {}
        for r in rows:
            d = out.setdefault((r["ID"], r.get("Kernel Name", "")), [r["ID"], r.get("Kernel Name", ""), r.get("Grid Size", ""), r.get("Block Size", ""), {}])
            d[4][r["Metric Name"]] = (number(r["Metric Value"]), r.get("Metric Unit", ""))
        return [tuple(v) for v in out.values()]
    units, res = (rows[0] if rows else {}), []
    for r in rows[1:] if units.get("ID", "x") == "" else rows:
        if not r.get("ID"):
            continue
        res.append((r["ID"], r.get("Kernel Name", ""), r.get("Grid Size", ""), r.get("Block Size", ""), {k: (number(v), units.get(k, "")) for k, v in r.items() if k not in ("ID", "Kernel Name", "Grid Size", "Block Size") and v not in ("", None)}))
    return res


def main(paths):
    for path in paths:
        found = launches_of(open(path, errors="replace").read())
        if not found:
            print(f"{path}: no launches in it")
        for i, name, grid, block, m in found:
            print(f"\n{path}: launch {i}: {name[:100]}  grid {grid or m.get('launch__grid_size', ('', ''))[0]} block {block or m.get('launch__block_size', ('', ''))[0]}")
            for label, pats in KEYS:
                hit = next((c for p in pats for c in m if re.match(p, c) and m[c][0] is not None), None)
                if hit:
                    v, u = m[hit]
                    print(f"  {label:24s} {v:14,.2f} {u}")
            stalls = sorted(((v[0], STALL.match(c).group(1)) for c, v in m.items() if STALL.match(c) and not c.endswith("_not_issued") and v[0]), reverse=True)
            tot = sum(v for v, _ in stalls) or 1
            for v, n in stalls[:6]:
                print(f"  stall {n:30s} {100 * v / tot:6.1f}% of the sampled stalls")


if __name__ == "__main__":
    main(sys.argv[1:])
