#!/usr/bin/env python3
"""Inspect AI's gsm8k and mmlu_pro (inspect_evals) against an OpenAI-compatible server, non-thinking, over one shard of each dataset: eval_run.py BASE_URL MODEL SHARD(i/n) LIMIT(0: all) OUTDIR
Writes OUTDIR/evals.json ([{dataset, task_id, value, n, shard, harness, notes}]) and Inspect's logs in OUTDIR/logs. The shard is every n-th question from the i-th (1-based)."""
import json, os, sys
from importlib.metadata import version
from pathlib import Path
from inspect_ai import eval as inspect_eval
from inspect_evals.gsm8k import gsm8k
from inspect_evals.mmlu_pro import mmlu_pro

base, model, shard, limit, out = sys.argv[1:6]
i, n = map(int, shard.split("/"))
harness = {"inspect_ai": version("inspect-ai"), "inspect_evals": version("inspect-evals")}
# (the HF eval.yaml's own setup is 0-shot with a model grader; these are inspect_evals' tasks, as they ship)
TASKS = [("openai/gsm8k", "gsm8k", gsm8k(), "10-shot, numeric match, non-thinking"), ("TIGER-Lab/MMLU-Pro", "mmlu_pro", mmlu_pro(shuffle=False), "0-shot chain of thought, non-thinking")]
want = os.environ.get("TASKS", "").split(",") if os.environ.get("TASKS") else None   # (TASKS=gsm8k: one benchmark only)
rows = []
for dataset, tid, task, notes in TASKS:
    if want and tid not in want:
        continue
    task.dataset = task.dataset[i - 1::n]
    log = inspect_eval(task, model=f"openai-api/glyd/{model}", model_base_url=base, limit=int(limit) or None, log_dir=f"{out}/logs", temperature=0, max_tokens=3072,
                       max_connections=8, extra_body={"chat_template_kwargs": {"enable_thinking": False}}, display="plain")[0]
    assert log.status == "success", f"{tid}: {log.status} {log.error}"
    rows.append({"dataset": dataset, "task_id": tid, "value": log.results.scores[0].metrics["accuracy"].value, "n": log.results.completed_samples, "shard": shard, "harness": harness, "notes": notes})
Path(out, "evals.json").write_text(json.dumps(rows, indent=2) + "\n")
