#!/usr/bin/env python3
"""The GPU check that the plan is what loads: `glyd-engine fit MODEL --json` (the plan, before anything is loaded) against the report a run of the same model
wrote (GLYD_PLAN_REPORT: the plan the loader took and what the load used of the GPU).

usage: ci_plan_check.py FIT.json LOADED.json
"""
import json
import sys

fit = json.load(open(sys.argv[1]))
loaded = json.load(open(sys.argv[2]))
plan, used = loaded["plan"], loaded["loaded"]
bad = []

# the plan fit printed and the plan run loaded: the same level, context, users and placement (the weights within 3%: fit sizes a model it has not
# downloaded from its config, run from the files it has), the same reserve and work
for key in ("schema", "family", "level", "context", "users", "chunk", "reserve_bytes", "work_bytes"):
    if fit[key] != plan[key]:
        bad.append(f"{key}: fit {fit[key]}, run {plan[key]}")
for key in ("weights_bytes", "kv_bytes"):
    if abs(fit[key] / plan[key] - 1) > 0.03:
        bad.append(f"{key}: fit {fit[key]}, run {plan[key]}")
if fit["experts"] != plan["experts"]:
    bad.append(f"experts: fit {fit['experts']}, run {plan['experts']}")
if fit.get("spread") != plan.get("spread"):
    bad.append(f"spread over the GPUs: fit {fit.get('spread')}, run {plan.get('spread')}")

# what the loader allocated against the plan: the weights the engine counts, and the memory the load took (the plan's total without its reserve)
# (the multi-GPU route and Condor do not count their weights apart: 0)
if used["weights_bytes"] and abs(used["weights_bytes"] / plan["weights_on_gpu_bytes"] - 1) > 0.03:
    bad.append(f"weights: planned {plan['weights_on_gpu_bytes']}, loaded {used['weights_bytes']}")
if used["gpu_used_bytes"] > plan["gpu_total_bytes"] - plan["reserve_bytes"]:
    bad.append(f"the load took {used['gpu_used_bytes']} of the GPU, the plan has {plan['gpu_total_bytes'] - plan['reserve_bytes']} besides its reserve")
if plan["gpu_total_bytes"] > plan["free_bytes"]:
    bad.append(f"the plan's {plan['gpu_total_bytes']} are more than the {plan['free_bytes']} free")

gib = 1 << 30
print(
    f"{plan['family']} {plan['level']}, {plan['context']} tokens: planned {plan['weights_on_gpu_bytes'] / gib:.2f} GiB of weights and "
    f"{(plan['gpu_total_bytes'] - plan['reserve_bytes']) / gib:.2f} GiB of GPU memory; loaded {used['weights_bytes'] / gib:.2f} GiB of weights, "
    f"the GPU lost {used['gpu_used_bytes'] / gib:.2f} GiB ({plan['free_bytes'] / gib:.2f} GiB were free)"
)
if bad:
    print("the plan is not what loaded:\n  " + "\n  ".join(bad))
    sys.exit(1)
