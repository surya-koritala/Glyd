import sys, ncu_show as n
steps = float(sys.argv[3]) if len(sys.argv) > 3 else 16384
files = sys.argv[1:3]
D = [n.load(f) for f in files]
def g(d, k):
    try: return float(d[k][0].replace(",", ""))
    except Exception: return float("nan")
sms = 58
rows = [
 ("sm cycles elapsed avg", "sm__cycles_elapsed.avg", 1),
 ("sm cycles active avg", "sm__cycles_active.avg", 1),
 ("warp instr executed", "smsp__inst_executed.sum", 1),
 ("ALU inst", "sm__inst_executed_pipe_alu.sum", 1),
 ("FMA inst", "sm__inst_executed_pipe_fma.sum", 1),
 ("FMAheavy inst", "sm__inst_executed_pipe_fmaheavy.sum", 1),
 ("FMAlite inst", "sm__inst_executed_pipe_fmalite.sum", 1),
 ("LSU inst", "sm__inst_executed_pipe_lsu.sum", 1),
 ("XU inst", "sm__inst_executed_pipe_xu.sum", 1),
 ("uniform inst", "sm__inst_executed_pipe_uniform.sum", 1),
 ("CBU (branch) inst", "sm__inst_executed_pipe_cbu.sum", 1),
 ("tensor inst", "sm__inst_executed_pipe_tensor.sum", 1),
 ("shared wavefronts", "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum", 1),
 ("shared conflicts", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", 1),
 ("shared ld wavefronts", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", 1),
 ("shared st wavefronts", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum", 1),
 ("lg (global) wavefronts", "l1tex__data_pipe_lsu_wavefronts_mem_lg.sum", 1),
]
print(f"{'':28s} " + "  ".join(f"{f.split('/')[-1][:22]:>22s}" for f in files) + "   (per step: / steps)")
for name, k, _ in rows:
    vals = [g(d, k) for d in D]
    if k.startswith("sm__cycles"):
        print(f"{name:28s} " + "  ".join(f"{v:12.0f} {v / (steps / sms):9.1f}/st" for v in vals))
    else:
        print(f"{name:28s} " + "  ".join(f"{v:12.0f} {v / steps:9.1f}/st" for v in vals))
for name, k in [("issue active %", "smsp__issue_active.avg.pct_of_peak_sustained_active"), ("ALU pipe % active", "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active"), ("warps/SMSP", "smsp__warps_active.avg.per_cycle_active"), ("eligible/SMSP", "smsp__warps_eligible.avg.per_cycle_active"), ("regs", "launch__registers_per_thread")]:
    print(f"{name:28s} " + "  ".join(f"{g(d, k):22.2f}" for d in D))
