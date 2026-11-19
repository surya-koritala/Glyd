#!/usr/bin/env python3
"""The slices of a benchmark merged into one score (Inspect AI's reader where installed, else the standard library): merge_slices.py OUT.json SLICE_DIR [SLICE_DIR ...]
Reads every Inspect .eval log under the slice directories (OUTDIR/logs/*.eval of eval_slice.py), takes each task's samples by id (a question answered in two slices counts once, the later log wins), and prints and writes
for each task: questions answered, correct ("C"), accuracy, how many hit the 3,072-token cap, mean output tokens, and the slices it was made of. A running score is the same call on the slices finished so far."""
import json
import sys
import zipfile
from pathlib import Path


def read(path):
    """The task's name and each sample's score, stop reason, output tokens and time. Through Inspect AI's own reader where it is installed:
    recent Inspect AI writes .eval logs with zstd compression, which Python's zipfile reads only from 3.14 on."""
    try:
        from inspect_ai.log import read_eval_log
    except ImportError:
        return read_zip(path)
    log = read_eval_log(path)
    rows = {}
    for s in log.samples or []:
        sc = next(iter((s.scores or {}).values()), None)
        ch = (s.output.choices or [None])[0] if s.output else None
        rows[s.id] = {"value": sc.value if sc else None, "error": s.error, "stop": ch.stop_reason if ch else None,
                      "tokens": s.output.usage.output_tokens if s.output and s.output.usage else None, "secs": s.total_time}
    return log.eval.task, rows


def read_zip(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    head = json.loads(z.read("header.json" if "header.json" in names else "_journal/start.json"))
    rows = {}
    for n in names:
        if n.startswith("samples/") and n.endswith(".json"):
            s = json.loads(z.read(n))
            sc = next(iter((s.get("scores") or {"": {"value": None}}).values()))
            ch = ((s.get("output") or {}).get("choices") or [{}])[0]
            rows[s["id"]] = {"value": sc.get("value"), "error": s.get("error"), "stop": ch.get("stop_reason"), "tokens": ((s.get("output") or {}).get("usage") or {}).get("output_tokens"),
                             "secs": s.get("total_time")}
    return head["eval"]["task"], rows


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    tasks = {}
    for d in dirs:
        for p in sorted(Path(d).glob("logs/*.eval")):
            t, rows = read(p)
            t = "gsm8k" if "gsm8k" in t else "mmlu_pro"
            tasks.setdefault(t, {"rows": {}, "slices": []})
            tasks[t]["rows"].update(rows)
            tasks[t]["slices"].append(str(d))
    res = {}
    for t, v in tasks.items():
        r = v["rows"]
        c = sum(1 for x in r.values() if x["value"] == "C")
        res[t] = {"answered": len(r), "correct": c, "accuracy": round(c / max(1, len(r)), 4), "hit_max_tokens": sum(1 for x in r.values() if x["stop"] == "max_tokens"),
                  "no_answer_errors": sum(1 for x in r.values() if x["error"]), "mean_output_tokens": round(sum(x["tokens"] or 0 for x in r.values()) / max(1, len(r)), 1),
                  "mean_secs": round(sum(x["secs"] or 0 for x in r.values()) / max(1, len(r)), 1), "slices": v["slices"], "ids": sorted(r, key=str), "values": {str(k): x["value"] for k, x in r.items()}}
        print(f"{t}: {c} / {len(r)} correct ({100 * c / max(1, len(r)):.1f}%), {res[t]['hit_max_tokens']} hit the 3,072-token cap, mean {res[t]['mean_output_tokens']} output tokens, {res[t]['mean_secs']} s a question, {len(v['slices'])} slices")
    Path(out).write_text(json.dumps(res, indent=1))


main()
